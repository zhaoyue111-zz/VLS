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
from vls.data import CaseRecord, iter_cases, iter_image_cases, read_image, read_image_and_label
from vls.v2_experiment import (
    padded_image_and_slicers,
    padded_visual_action_and_slicers,
    prepare_functional_seg_head,
    resolve_device,
    select_patch_slicers,
    state_to_intermediate_prediction,
    visual_action,
)
from vls.v6_0_imagined_world_reliability import (
    SOURCE_PROMPT,
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
ORDERS = ("forward", "reverse")
STRONG_ACTIONS = (("gamma", 0.30), ("blur", 1.5))


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V7.0d train/eval-separated protocol sanity.")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v7_0d_protocol_sanity")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--train-cases", type=int, default=4)
    parser.add_argument("--evaluation-cases", type=int, default=4)
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


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "dice": 1.0 if tp + fp + fn == 0 else 2.0 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def pooled_rows(rows: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        case = "__global__" if level in ("global", "global_step") else row["case"]
        step = int(row["step"]) if level in ("step", "global_step") else -1
        key = (row["variant"], row["order"], case, step)
        grouped.setdefault(key, []).append(row)
    pooled = []
    for (variant, order, case, step), group in sorted(grouped.items()):
        tp = sum(int(row["tp"]) for row in group)
        fp = sum(int(row["fp"]) for row in group)
        tn = sum(int(row["tn"]) for row in group)
        fn = sum(int(row["fn"]) for row in group)
        pooled.append({
            "variant": variant,
            "order": order,
            **({"step": step} if level in ("step", "global_step") else {}),
            "case": case,
            "patches": len(group),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "dice": 1.0 if tp + fp + fn == 0 else 2.0 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        })
    return pooled


def trainable_parameters(network: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = []
    for name, parameter in network.named_parameters():
        parameter.requires_grad = name.startswith("decoder.") or name.startswith("project_to_decoder_channels.")
        if parameter.requires_grad:
            parameters.append(parameter)
    return parameters


def resize_logits(logits: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(logits.shape[-3:]) == shape:
        return logits.float()
    return F.interpolate(logits.float(), size=shape, mode="trilinear", align_corners=False)


def reliability_from_source(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    source_state: torch.Tensor,
    source_probability: torch.Tensor,
    selected_stage: str,
    final_shape: tuple[int, int, int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    gamma_state = world_model(source_state, action=visual_action("gamma", 0.30, device))
    blur_state = world_model(source_state, action=visual_action("blur", 1.5, device))
    probabilities = [
        torch.sigmoid(resize_logits(
            state_to_intermediate_prediction(interface, selected_stage, state), final_shape,
        ))
        for state in (source_state, gamma_state, blur_state)
    ]
    stack = torch.cat(probabilities, dim=0)
    pairwise = torch.stack([
        (stack[0] - stack[1]).abs(),
        (stack[0] - stack[2]).abs(),
        (stack[1] - stack[2]).abs(),
    ], dim=0).mean(dim=0)
    confidence = torch.maximum(source_probability, 1.0 - source_probability)
    c_rank = percentile_rank(confidence.flatten().cpu().numpy()).astype(np.float32)
    d_rank = percentile_rank(pairwise.flatten().cpu().numpy()).astype(np.float32)
    world_stability = 1.0 - d_rank
    return {
        "confidence_rank": c_rank,
        "world_stability": world_stability,
        "joint_product": (c_rank * world_stability).astype(np.float32),
    }


def strong_padded_image(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    action_family: str,
    strength: float,
    original_padded: torch.Tensor,
) -> torch.Tensor:
    padded, _ = padded_visual_action_and_slicers(
        interface.predictor, image, action_family, strength,
    )
    if tuple(padded.shape) != tuple(original_padded.shape):
        raise AssertionError(f"strong-view padded shape differs for {action_family} {strength}")
    return padded


@torch.inference_mode()
def build_train_cache(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    cases: list[CaseRecord],
    prompt_embedding: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    cache = []
    for case_index, case in enumerate(cases):
        image, _ = read_image(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface, image, prompt_embedding, args.patches_per_case,
            args.foreground_patches_per_case, args.foreground_candidate_patches,
            args.foreground_threshold,
        )
        action_family, strength = STRONG_ACTIONS[case_index % len(STRONG_ACTIONS)]
        strong_padded = strong_padded_image(interface, image, action_family, strength, original_padded)
        for patch_index, slicer in enumerate(slicers):
            patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            strong_patch = torch.clone(strong_padded[slicer][None], memory_format=torch.contiguous_format)
            result = interface.forward_with_states(patch, prompt_embedding)
            source_state = result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
            source_probability = torch.sigmoid(result["final_prediction"][:, 0:1].detach().float().to(device))
            final_shape = tuple(int(size) for size in source_probability.shape[-3:])
            weights = reliability_from_source(
                interface, world_model, source_state, source_probability,
                args.selected_stage, final_shape, device,
            )
            pseudo = (source_probability > args.prediction_threshold).float()
            cache.append({
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "image": strong_patch.detach().cpu(),
                "embedding": prompt_embedding.detach().cpu(),
                "pseudo": pseudo.detach().cpu(),
                "weights": weights,
                "has_foreground": bool(torch.count_nonzero(pseudo)),
                "augmentation": f"{action_family}:{strength}",
                "case_index": case_index,
            })
    return cache


@torch.inference_mode()
def build_evaluation_cache(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    cases: list[CaseRecord],
    prompt_embedding: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    cache = []
    for case in cases:
        image, label, _ = read_image_and_label(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface, image, prompt_embedding, args.patches_per_case,
            args.foreground_patches_per_case, args.foreground_candidate_patches,
            args.foreground_threshold,
        )
        label_padded = pad_label_like_image(interface, label)
        for patch_index, slicer in enumerate(slicers):
            patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            result = interface.forward_with_states(patch, prompt_embedding)
            source_state = result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
            source_probability = torch.sigmoid(result["final_prediction"][:, 0:1].detach().float().to(device))
            final_shape = tuple(int(size) for size in source_probability.shape[-3:])
            weights = reliability_from_source(
                interface, world_model, source_state, source_probability,
                args.selected_stage, final_shape, device,
            )
            gt = (label_padded[slicer][None].to(device) == args.label_value).float()
            if tuple(gt.shape[-3:]) != final_shape:
                gt = F.interpolate(gt, size=final_shape, mode="nearest")
            cache.append({
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "image": patch.detach().cpu(),
                "embedding": prompt_embedding.detach().cpu(),
                "gt_np": gt.flatten().cpu().numpy().astype(bool),
                "weights": weights,
            })
    return cache


def evaluate_network(
    variant: str,
    order: str,
    step: int,
    network: torch.nn.Module,
    cache: list[dict[str, Any]],
    interface: VoxTellStateInterface,
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    network.eval()
    rows = []
    with torch.inference_mode():
        for sample in cache:
            image = sample["image"].to(device).clone()
            embedding = sample["embedding"].to(device).clone().float()
            result = interface._network_forward_with_states(network, image, embedding)
            prediction = (
                torch.sigmoid(result["final_prediction"][:, 0:1]) > threshold
            ).flatten().cpu().numpy()
            rows.append({
                "variant": variant,
                "order": order,
                "step": step,
                "case": sample["case"],
                "patch_index": sample["patch_index"],
                "patch_kind": sample["patch_kind"],
                **binary_metrics(prediction, sample["gt_np"]),
            })
            del result, image, embedding, prediction
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def train_one_order(
    variant: str,
    source: str,
    order: str,
    base_network: torch.nn.Module,
    train_cache: list[dict[str, Any]],
    evaluation_cache: list[dict[str, Any]],
    interface: VoxTellStateInterface,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    student = copy.deepcopy(base_network).to(device)
    student.train()
    parameters = trainable_parameters(student)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    training_rows = []
    pseudo_rows = []
    eval_rows = []
    ordered_cache = train_cache if order == "forward" else list(reversed(train_cache))
    for step, sample in enumerate(ordered_cache, start=1):
        image = sample["image"].to(device).clone()
        embedding = sample["embedding"].to(device).clone().float()
        pseudo = sample["pseudo"].to(device).clone()
        weight = torch.from_numpy(sample["weights"][source]).to(device=device, dtype=torch.float32).view_as(pseudo)
        optimizer.zero_grad(set_to_none=True)
        with nullcontext():
            result = interface._network_forward_with_states(student, image, embedding)
            logits = result["final_prediction"][:, 0:1].float()
            per_voxel = F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
            loss = (per_voxel * weight).sum() / weight.sum().clamp_min(1e-6)
        loss.backward()
        gradient_norm = float(torch.sqrt(sum(
            parameter.grad.detach().float().pow(2).sum()
            for parameter in parameters if parameter.grad is not None
        )).detach().cpu())
        before_update = [parameter.detach().clone() for parameter in parameters]
        optimizer.step()
        update_delta = float(torch.sqrt(sum(
            (parameter.detach() - before).float().pow(2).sum()
            for parameter, before in zip(parameters, before_update, strict=True)
        )).detach().cpu())
        if float(loss.detach().cpu()) <= 1e-12 and gradient_norm <= 1e-10 and update_delta > 1e-9:
            raise AssertionError(f"zero-loss/zero-gradient update changed parameters: {variant}/{order}/{step}")
        training_rows.append({
            "variant": variant,
            "order": order,
            "step": step,
            "case": sample["case"],
            "patch_index": sample["patch_index"],
            "augmentation": sample["augmentation"],
            "patch_kind": sample["patch_kind"],
            "loss": float(loss.detach().cpu()),
            "gradient_norm": gradient_norm,
            "update_delta_norm": update_delta,
            "weight_decay": 0.0,
            "learning_rate": args.learning_rate,
        })
        eval_rows.extend(evaluate_network(variant, order, step, student, evaluation_cache, interface, device, args.prediction_threshold))
        pseudo_rows.append({
            "variant": variant,
            "order": order,
            "step": step,
            "case": sample["case"],
            "patch_index": sample["patch_index"],
            "augmentation": sample["augmentation"],
            "reliability_source": source,
            "pseudo_positive_voxels": int(np.count_nonzero(sample["pseudo"].numpy())),
            "weight_sum": float(sample["weights"][source].sum()),
            "weight_mean": float(sample["weights"][source].mean()),
            "weight_min": float(sample["weights"][source].min()),
            "weight_max": float(sample["weights"][source].max()),
        })
        del result, logits, per_voxel, loss, image, embedding, pseudo, weight, before_update
    del student, optimizer, parameters
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return training_rows, pseudo_rows, eval_rows


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
    print(f"[V7.0d] cuda_probe_before_seed={torch.cuda.is_available()} requested_device={args.device}", flush=True)
    set_seed(args.seed)
    device = resolve_device(args)
    print(f"[V7.0d] cuda_probe_after_seed={torch.cuda.is_available()} resolved_device={device}", flush=True)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V7.0d requires CUDA, resolve_device returned {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    train_cases = iter_image_cases(paths, split="train", limit=args.train_cases)
    evaluation_cases = iter_cases(paths, split="test", limit=args.evaluation_cases)
    train_names = [case.case for case in train_cases]
    evaluation_names = [case.case for case in evaluation_cases]
    overlap = sorted(set(train_names) & set(evaluation_names))
    if overlap:
        raise AssertionError(f"train/evaluation case overlap: {overlap}")
    (output_dir / "adaptation_cases.json").write_text(json.dumps(train_names, indent=2))
    (output_dir / "evaluation_cases.json").write_text(json.dumps(evaluation_names, indent=2))

    teacher = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
    prepare_functional_seg_head(teacher, args.selected_stage)
    prompt_embedding = teacher.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(
        checkpoint_path,
        int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
        device,
        args.hidden_channels,
    )
    print("[V7.0d] building train cache", flush=True)
    train_cache = build_train_cache(teacher, world_model, train_cases, prompt_embedding, args, device)
    print("[V7.0d] building evaluation cache", flush=True)
    evaluation_cache = build_evaluation_cache(teacher, world_model, evaluation_cases, prompt_embedding, args, device)
    world_model.to("cpu")
    teacher.network.to("cpu")
    if hasattr(teacher, "functional_seg_head"):
        teacher.functional_seg_head.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_samples = []
    skipped = []
    for case in train_cases:
        candidates = [
            sample for sample in train_cache
            if sample["case"] == case.case and sample["patch_kind"] == "foreground" and sample["has_foreground"]
        ]
        if candidates:
            train_samples.append(sorted(candidates, key=lambda sample: sample["patch_index"])[0])
        else:
            skipped.append({"case": case.case, "reason": "no_nonempty_foreground_patch"})
    train_samples.sort(key=lambda sample: (sample["case_index"], sample["patch_index"]))
    if len(train_samples) != len(train_cases):
        raise RuntimeError(f"V7.0d requires one effective update per train case, got {len(train_samples)}/{len(train_cases)}")

    base_network = copy.deepcopy(teacher.network).cpu().eval()
    for parameter in base_network.parameters():
        parameter.requires_grad = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base_network.to(device)
    initial_patch_rows = evaluate_network(
        "A_init_no_adaptation", "forward", 0, base_network,
        evaluation_cache, teacher, device, args.prediction_threshold,
    )
    base_network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    training_rows = []
    pseudo_rows = []
    per_step_patch_rows = list(initial_patch_rows)
    final_patch_rows = list(initial_patch_rows)
    final_rows = []
    for variant, source in VARIANTS.items():
        for order in ORDERS:
            print(f"[V7.0d] {variant} {order}", flush=True)
            variant_training, variant_pseudo, variant_eval = train_one_order(
                variant, source, order, base_network, train_samples, evaluation_cache,
                teacher, args, device,
            )
            training_rows.extend(variant_training)
            pseudo_rows.extend(variant_pseudo)
            per_step_patch_rows.extend(variant_eval)
            final_step = len(train_samples)
            final_rows.extend(row for row in variant_eval if int(row["step"]) == final_step)

    # The initial model is identical for both order labels; keep one baseline
    # row in final pooled outputs and use forward for its canonical identity.
    final_pooled_rows = pooled_rows(final_patch_rows, "global")
    final_pooled_rows.extend(pooled_rows(final_rows, "global"))
    final_case_rows = pooled_rows(final_patch_rows, "case")
    final_case_rows.extend(pooled_rows(final_rows, "case"))
    init_by_order = {
        row["order"]: row for row in final_pooled_rows
        if row["variant"] == "A_init_no_adaptation"
    }
    for row in final_pooled_rows + final_case_rows:
        baseline = init_by_order.get(row["order"])
        if baseline is not None and row["variant"] != "A_init_no_adaptation":
            row["delta_dice_vs_A_init"] = row["dice"] - baseline["dice"]
            row["delta_precision_vs_A_init"] = row["precision"] - baseline["precision"]
            row["delta_recall_vs_A_init"] = row["recall"] - baseline["recall"]
    per_step_eval = pooled_rows(per_step_patch_rows, "global_step")

    global_lookup = {(row["variant"], row["order"]): row for row in final_pooled_rows}
    order_sensitivity = {}
    for variant in VARIANTS:
        forward = global_lookup[(variant, "forward")]
        reverse = global_lookup[(variant, "reverse")]
        order_sensitivity[variant] = {
            "forward_minus_reverse_dice": forward["dice"] - reverse["dice"],
            "forward_minus_reverse_precision": forward["precision"] - reverse["precision"],
            "forward_minus_reverse_recall": forward["recall"] - reverse["recall"],
        }
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    peak_allocated = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0
    write_csv(output_dir / "training_loss.csv", training_rows)
    write_csv(output_dir / "per_step_eval.csv", per_step_eval)
    write_csv(output_dir / "pooled_by_case.csv", final_case_rows)
    write_csv(output_dir / "pooled_global.csv", final_pooled_rows)
    write_csv(output_dir / "pseudo_label_stats.csv", pseudo_rows)
    summary = {
        "stage": "V7.0d protocol sanity",
        "source_checkpoint": str(checkpoint_path),
        "selected_stage": args.selected_stage,
        "seed": args.seed,
        "resolved_device": str(device),
        "gpu_name": gpu_name,
        "peak_cuda_allocated_mb": peak_allocated,
        "peak_cuda_reserved_mb": peak_reserved,
        "config": vars(args),
        "adaptation_cases": train_names,
        "evaluation_cases": evaluation_names,
        "case_overlap": overlap,
        "case_overlap_count": len(overlap),
        "effective_updates_per_order_variant": len(train_samples),
        "skipped_train_cases": skipped,
        "training_order_cases": [sample["case"] for sample in train_samples],
        "strong_view_sequence": [
            {"case": sample["case"], "augmentation": sample["augmentation"]}
            for sample in train_samples
        ],
        "variants": VARIANTS,
        "orders": list(ORDERS),
        "training": {
            "loss": "weighted pseudo-label BCE with fixed teacher pseudo-labels",
            "student_trainable_scope": "decoder.* and project_to_decoder_channels.* only",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "teacher_train_gt_read": False,
            "student_view": "fixed label-preserving gamma(+0.30)/blur(1.5) alternating by train-case index",
            "same_cache_initialization_and_student_view_sequence": True,
            "world_predictor_updated": False,
            "new_loss_or_module": False,
        },
        "reliability_maps": {
            "A0_confidence_rank": "patch-wise tie-aware percentile rank of max(p,1-p)",
            "A1_world": "1 - patch-wise tie-aware percentile rank of visual-only pairwise_abs",
            "A2_joint_product": "A0_confidence_rank * A1_world",
        },
        "final_global_metrics": final_pooled_rows,
        "order_sensitivity": order_sensitivity,
        "outputs": {
            "adaptation_cases": str(output_dir / "adaptation_cases.json"),
            "evaluation_cases": str(output_dir / "evaluation_cases.json"),
            "training_loss": str(output_dir / "training_loss.csv"),
            "per_step_eval": str(output_dir / "per_step_eval.csv"),
            "pooled_by_case": str(output_dir / "pooled_by_case.csv"),
            "pooled_global": str(output_dir / "pooled_global.csv"),
            "pseudo_label_stats": str(output_dir / "pseudo_label_stats.csv"),
            "summary": str(output_dir / "summary.json"),
        },
        "status": "protocol_sanity_complete; no 10-50 step training, LoRA, EMA, gating, or new loss",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
