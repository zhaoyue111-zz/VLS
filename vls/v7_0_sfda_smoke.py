from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import (
    prepare_functional_seg_head,
    resolve_device,
    select_patch_slicers,
    state_to_intermediate_prediction,
    visual_action,
)
from vls.v6_0_imagined_world_reliability import (
    SOURCE_PROMPT,
    flatten_prompt_embedding,
    load_world_model,
    pad_label_like_image,
)
from vls.v6_1_unified_reliability_fusion import percentile_rank
from vls.voxtell_states import VoxTellStateInterface


VARIANTS = {
    "A0_confidence_rank": "confidence_rank",
    "A1_world": "world_stability",
    "A2_joint_product": "joint_product",
}


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V7.0c CUDA controlled confidence/world/joint SFDA confirmation.")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v7_0c_sfda_smoke")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resize_logits(logits: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(logits.shape[-3:]) == shape:
        return logits.float()
    return F.interpolate(logits.float(), size=shape, mode="trilinear", align_corners=False)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    return {
        "dice": 1.0 if tp + fp + fn == 0 else 2.0 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


@torch.inference_mode()
def build_cache(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    cases: list[Any],
    selected_stage: str,
    prompt_embedding: torch.Tensor,
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
    prediction_threshold: float,
    label_value: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    for case in cases:
        print(f"[V7.0c] cache case={case.case}", flush=True)
        image, label, _ = read_image_and_label(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface, image, prompt_embedding, patches_per_case,
            foreground_patches_per_case, foreground_candidate_patches, foreground_threshold,
        )
        label_padded = pad_label_like_image(interface, label)
        embedding = prompt_embedding
        for patch_index, slicer in enumerate(slicers):
            print(f"[V7.0c] cache patch case={case.case} index={patch_index} kind={patch_kinds[patch_index]}", flush=True)
            patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            teacher_result = interface.forward_with_states(patch, embedding)
            source_state = teacher_result["decoder_states"][selected_stage][:, 0].detach().float().to(device)
            source_final_logits = teacher_result["final_prediction"][:, 0:1].detach().float().to(device)
            source_probability = torch.sigmoid(source_final_logits)
            final_shape = tuple(int(size) for size in source_probability.shape[-3:])
            gamma_state = world_model(source_state, action=visual_action("gamma", 0.30, device))
            blur_state = world_model(source_state, action=visual_action("blur", 1.5, device))
            imagined = {
                "original": source_state,
                "gamma": gamma_state,
                "blur": blur_state,
            }
            imagined_probabilities = [
                torch.sigmoid(resize_logits(
                    state_to_intermediate_prediction(interface, selected_stage, state), final_shape,
                ))
                for state in imagined.values()
            ]
            stack = torch.cat(imagined_probabilities, dim=0)
            pairwise = torch.stack([
                (stack[0] - stack[1]).abs(), (stack[0] - stack[2]).abs(), (stack[1] - stack[2]).abs(),
            ], dim=0).mean(dim=0)
            confidence = torch.maximum(source_probability, 1.0 - source_probability)
            c_rank = percentile_rank(confidence.flatten().cpu().numpy())
            d_rank = percentile_rank(pairwise.flatten().cpu().numpy())
            c_rank = c_rank.astype(np.float32)
            world_stability = (1.0 - d_rank).astype(np.float32)
            reliability = {
                "confidence_rank": c_rank,
                "world_stability": world_stability,
                "joint_product": (c_rank * world_stability).astype(np.float32),
                "raw_confidence": confidence.flatten().cpu().numpy().astype(np.float32),
            }
            gt = (label_padded[slicer][None].to(device) == label_value).float()
            if tuple(gt.shape[-3:]) != final_shape:
                gt = F.interpolate(gt, size=final_shape, mode="nearest")
            pseudo = (source_probability > prediction_threshold).float()
            gt_np = gt.flatten().cpu().numpy().astype(bool)
            pseudo_np = pseudo.flatten().cpu().numpy().astype(bool)
            cache.append({
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "image": patch.detach().cpu(),
                "embedding": embedding.detach().cpu(),
                "pseudo": pseudo.detach().cpu(),
                "gt": gt.detach().cpu(),
                "gt_np": gt_np,
                "pseudo_np": pseudo_np,
                "weights": reliability,
                "teacher_metrics": binary_metrics(pseudo_np, gt_np),
                "has_foreground": bool(np.count_nonzero(pseudo_np)),
            })
    return cache


def trainable_parameters(network: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = []
    for name, parameter in network.named_parameters():
        parameter.requires_grad = name.startswith("decoder.") or name.startswith("project_to_decoder_channels.")
        if parameter.requires_grad:
            parameters.append(parameter)
    return parameters


def train_variant(
    variant: str,
    source: str,
    base_network: torch.nn.Module,
    cache: list[dict[str, Any]],
    train_samples: list[dict[str, Any]],
    interface: VoxTellStateInterface,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    student = copy.deepcopy(base_network).to(device)
    student.train()
    parameters = trainable_parameters(student)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    loss_rows, pseudo_rows, metric_rows = [], [], []
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    for step, sample in enumerate(train_samples, start=1):
        # build_cache runs under inference_mode; clone targets and inputs before
        # they participate in an autograd graph.
        patch = sample["image"].to(device).clone()
        embedding = sample["embedding"].to(device).clone().float()
        pseudo = sample["pseudo"].to(device).clone()
        weight = torch.from_numpy(sample["weights"][source]).to(device=device, dtype=torch.float32).view_as(pseudo)
        optimizer.zero_grad(set_to_none=True)
        # Keep the adaptation path in FP32.  The teacher cache uses VoxTell's
        # existing CUDA autocast, but FP16 here can underflow tiny pseudo-label
        # BCE gradients for confidence-heavy maps.
        with nullcontext():
            student_result = interface._network_forward_with_states(student, patch, embedding)
            logits = student_result["final_prediction"][:, 0:1].float()
            per_voxel = F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
            loss = (per_voxel * weight).sum() / weight.sum().clamp_min(1e-6)
        loss.backward()
        gradient_norm = float(torch.sqrt(sum(
            parameter.grad.detach().float().pow(2).sum() for parameter in parameters if parameter.grad is not None
        )).detach().cpu())
        before_update = [parameter.detach().clone() for parameter in parameters]
        optimizer.step()
        parameter_delta = float(torch.sqrt(sum(
            (parameter.detach() - initial).float().pow(2).sum()
            for parameter, initial in zip(parameters, initial_parameters, strict=True)
        )).detach().cpu())
        update_delta = float(torch.sqrt(sum(
            (parameter.detach() - before).float().pow(2).sum()
            for parameter, before in zip(parameters, before_update, strict=True)
        )).detach().cpu())
        if float(loss.detach().cpu()) <= 1e-12 and gradient_norm <= 1e-10 and update_delta > 1e-9:
            raise AssertionError(
                f"zero-loss/zero-gradient update changed parameters: {variant} step={step} delta={update_delta}"
            )
        loss_rows.append({
            "variant": variant, "step": step, "case": sample["case"], "patch_index": sample["patch_index"],
            "loss": float(loss.detach().cpu()), "gradient_norm": gradient_norm,
            "student_parameter_delta_norm": parameter_delta, "update_delta_norm": update_delta,
            "weight_decay": 0.0, "training_patch_kind": sample["patch_kind"],
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "peak_cuda_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
            "peak_cuda_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0,
        })
    student.eval()
    for sample in cache:
        patch = sample["image"].to(device).clone()
        embedding = sample["embedding"].to(device).clone().float()
        with torch.inference_mode():
            with nullcontext():
                result = interface._network_forward_with_states(student, patch, embedding)
            prediction = (torch.sigmoid(result["final_prediction"][:, 0:1]) > args.prediction_threshold).flatten().cpu().numpy()
        metrics = binary_metrics(prediction, sample["gt_np"])
        metric_rows.append({
            "variant": variant, "case": sample["case"], "patch_index": sample["patch_index"],
            **metrics,
        })
        weights = sample["weights"][source]
        correct = (sample["pseudo_np"] == sample["gt_np"])
        pseudo_rows.append({
            "variant": variant, "case": sample["case"], "patch_index": sample["patch_index"],
            "reliability_source": source, "training_used": int(sample.get("training_used", False)),
            "patch_kind": sample["patch_kind"], "pseudo_label_voxels": int(correct.size),
            "nonzero_weight_voxels": int(np.count_nonzero(weights > 0)),
            "weight_sum": float(weights.sum()), "weight_mean": float(weights.mean()),
            "weighted_correct_mass": float(weights[correct].sum()),
            "weighted_error_mass": float(weights[~correct].sum()),
            "teacher_pseudo_accuracy": float(correct.mean()),
            "teacher_weighted_accuracy": float(weights[correct].sum() / max(weights.sum(), 1e-12)),
        })
    del student, optimizer, parameters, initial_parameters
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_rows, pseudo_rows, metric_rows


@torch.inference_mode()
def evaluate_network(
    variant: str,
    network: torch.nn.Module,
    cache: list[dict[str, Any]],
    interface: VoxTellStateInterface,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    network.eval()
    rows = []
    for sample in cache:
        patch = sample["image"].to(device)
        embedding = sample["embedding"].to(device).float()
        with nullcontext():
            result = interface._network_forward_with_states(network, patch, embedding)
        prediction = (torch.sigmoid(result["final_prediction"][:, 0:1]) > args.prediction_threshold).flatten().cpu().numpy()
        rows.append({
            "variant": variant, "case": sample["case"], "patch_index": sample["patch_index"],
            "patch_kind": sample["patch_kind"], **binary_metrics(prediction, sample["gt_np"]),
        })
    return rows


def pooled_rows(patch_rows: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in patch_rows:
        key = (row["variant"], "__global__" if level == "global" else row["case"])
        groups.setdefault(key, []).append(row)
    output = []
    for (variant, case), rows in sorted(groups.items()):
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        tn = sum(int(row["tn"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        output.append({
            "variant": variant, "case": case, "patches": len(rows),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "dice": 1.0 if tp + fp + fn == 0 else 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(
            "V7.0c requires CUDA: --device cuda was requested but resolve_device returned "
            f"{device}. CUDA availability={torch.cuda.is_available()}"
        )
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        gpu_name = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root), voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root), split_json=Path(args.split_json),
    )
    teacher = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
    prepare_functional_seg_head(teacher, args.selected_stage)
    # The text encoder is only needed once.  Precompute while the large VoxTell
    # network is on CPU so CUDA can hold the network without a text-model peak.
    prompt_embedding = teacher.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    cases = iter_cases(paths, split="test", limit=args.val_cases)
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(
        checkpoint_path, int(checkpoint["state_dict"]["output_projection.bias"].shape[0]), device, args.hidden_channels,
    )
    cache = build_cache(
        teacher, world_model, cases, args.selected_stage, prompt_embedding,
        args.patches_per_case,
        args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
        args.prediction_threshold, args.label_value, device,
    )
    print(f"[V7.0c] cache complete patches={len(cache)} device={device} gpu={gpu_name}", flush=True)
    teacher.network.to("cpu")
    if hasattr(teacher, "functional_seg_head"):
        teacher.functional_seg_head.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_samples = []
    skipped = []
    for case in cases:
        candidates = [
            sample for sample in cache
            if sample["case"] == case.case
            and sample["patch_kind"] == "foreground"
            and sample["has_foreground"]
        ]
        if candidates:
            chosen = sorted(candidates, key=lambda sample: sample["patch_index"])[0]
            chosen["training_used"] = True
            train_samples.append(chosen)
        else:
            skipped.append({"case": case.case, "reason": "no_nonempty_foreground_patch"})
    train_samples.sort(key=lambda sample: (sample["case"], sample["patch_index"]))
    base_network = copy.deepcopy(teacher.network).cpu().eval()
    for parameter in base_network.parameters():
        parameter.requires_grad = False
    base_network.to(device)
    initial_patch_rows = evaluate_network("A_init_no_adaptation", base_network, cache, teacher, args, device)
    base_network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[V7.0c] initial baseline complete", flush=True)
    loss_rows, pseudo_rows, patch_rows = [], [], list(initial_patch_rows)
    for variant, source in VARIANTS.items():
        print(f"[V7.0c] train/evaluate variant={variant} updates={len(train_samples)}", flush=True)
        variant_loss, variant_pseudo, variant_metrics = train_variant(
            variant, source, base_network, cache, train_samples, teacher, args, device,
        )
        loss_rows.extend(variant_loss)
        pseudo_rows.extend(variant_pseudo)
        patch_rows.extend(variant_metrics)
        print(f"[V7.0c] variant complete={variant}", flush=True)
    initial_baseline = pooled_rows(initial_patch_rows, "global") + pooled_rows(initial_patch_rows, "case")
    pooled_global = pooled_rows(patch_rows, "global")
    pooled_by_case = pooled_rows(patch_rows, "case")
    initial_lookup = {(row["case"], row["variant"]): row for row in initial_baseline}
    for row in pooled_global + pooled_by_case:
        baseline = initial_lookup.get((row["case"], "A_init_no_adaptation"))
        if baseline and row["variant"] != "A_init_no_adaptation":
            row["delta_dice_vs_A_init"] = row["dice"] - baseline["dice"]
            row["delta_precision_vs_A_init"] = row["precision"] - baseline["precision"]
            row["delta_recall_vs_A_init"] = row["recall"] - baseline["recall"]
    global_by_variant = {row["variant"]: row for row in pooled_global}
    case_names = sorted({row["case"] for row in pooled_by_case})
    case_lookup = {(row["case"], row["variant"]): row for row in pooled_by_case}
    joint_tradeoff = {
        "cases_with_precision_nonworse_than_A0": sum(
            case_lookup[(case, "A2_joint_product")]["precision"] >= case_lookup[(case, "A0_confidence_rank")]["precision"]
            for case in case_names if (case, "A2_joint_product") in case_lookup and (case, "A0_confidence_rank") in case_lookup
        ),
        "cases_with_recall_nonworse_than_A0": sum(
            case_lookup[(case, "A2_joint_product")]["recall"] >= case_lookup[(case, "A0_confidence_rank")]["recall"]
            for case in case_names if (case, "A2_joint_product") in case_lookup and (case, "A0_confidence_rank") in case_lookup
        ),
        "cases_with_both_precision_and_recall_nonworse_than_A0": sum(
            case_lookup[(case, "A2_joint_product")]["precision"] >= case_lookup[(case, "A0_confidence_rank")]["precision"]
            and case_lookup[(case, "A2_joint_product")]["recall"] >= case_lookup[(case, "A0_confidence_rank")]["recall"]
            for case in case_names if (case, "A2_joint_product") in case_lookup and (case, "A0_confidence_rank") in case_lookup
        ),
        "case_count": len(case_names),
    }
    write_csv(output_dir / "training_loss.csv", loss_rows)
    write_csv(output_dir / "pseudo_label_stats.csv", pseudo_rows)
    write_csv(output_dir / "patch_metrics.csv", patch_rows)
    write_csv(output_dir / "initial_baseline.csv", initial_baseline)
    write_csv(output_dir / "pooled_by_case.csv", pooled_by_case)
    write_csv(output_dir / "pooled_global.csv", pooled_global)
    resolved_device = str(device)
    peak_allocated = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0
    summary = {
        "stage": "V7.0c CUDA controlled confirmation",
        "source_checkpoint": str(checkpoint_path),
        "voxtell_model_dir": str(paths.voxtell_model_dir),
        "selected_stage": args.selected_stage,
        "seed": args.seed,
        "resolved_device": resolved_device,
        "gpu_name": gpu_name,
        "peak_cuda_allocated_mb": peak_allocated,
        "peak_cuda_reserved_mb": peak_reserved,
        "config": vars(args),
        "cases": [case.case for case in cases],
        "patches": len(cache),
        "training_updates": len(train_samples),
        "training_order": [{"case": sample["case"], "patch_index": sample["patch_index"], "patch_kind": sample["patch_kind"]} for sample in train_samples],
        "effective_update_counts": {variant: len(train_samples) for variant in VARIANTS},
        "effective_case_coverage": {variant: sorted({sample["case"] for sample in train_samples}) for variant in VARIANTS},
        "skipped_cases": skipped,
        "variants": VARIANTS,
        "training": {
            "loss": "weighted pseudo-label BCE with fixed VoxTell teacher labels",
            "student_trainable_scope": "decoder.* and project_to_decoder_channels.* only",
            "same_initialization_data_steps_and_lr": True,
            "weight_decay": 0.0,
            "reliability_maps": {
                "A0_confidence_rank": "patch-wise tie-aware percentile rank of max(p,1-p)",
                "A1_world": "1 - patch-wise tie-aware percentile rank of visual-only pairwise_abs",
                "A2_joint_product": "A0_confidence_rank * A1_world",
            },
            "world_predictor_updated": False,
            "new_loss_or_module": False,
            "ema_or_sfda_extra": False,
        },
        "outputs": {
            "training_loss": str(output_dir / "training_loss.csv"),
            "pseudo_label_stats": str(output_dir / "pseudo_label_stats.csv"),
            "patch_metrics": str(output_dir / "patch_metrics.csv"),
            "initial_baseline": str(output_dir / "initial_baseline.csv"),
            "pooled_by_case": str(output_dir / "pooled_by_case.csv"),
            "pooled_global": str(output_dir / "pooled_global.csv"),
            "summary": str(output_dir / "summary.json"),
        },
        "comparison": "Compare A0/A1/A2 against A_init_no_adaptation using pooled TP/FP/TN/FN metrics and deltas; no patch averaging is used for primary case/global results.",
        "global_pooled_metrics": global_by_variant,
        "joint_product_precision_recall_tradeoff": joint_tradeoff,
        "interpretation": "Controlled smoke only; joint-product stability is assessed descriptively from pooled and per-case precision/recall, not used to select future routing or hyperparameters.",
        "status": "smoke_complete; no V7 hyperparameter search or full training performed",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
