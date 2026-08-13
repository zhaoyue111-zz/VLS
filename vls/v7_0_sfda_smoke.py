from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import random
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
    "A0_confidence": "confidence",
    "A1_world": "world_stability",
    "A2_joint_product": "joint_product",
}


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V7.0 minimal confidence/world/joint SFDA smoke.")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v7_0_sfda_smoke")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--steps", type=int, default=2)
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
        image, label, _ = read_image_and_label(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface, image, [SOURCE_PROMPT], patches_per_case,
            foreground_patches_per_case, foreground_candidate_patches, foreground_threshold,
        )
        label_padded = pad_label_like_image(interface, label)
        embedding = interface.embed_text_prompts([SOURCE_PROMPT])
        for patch_index, slicer in enumerate(slicers):
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
            reliability = {
                "confidence": confidence.flatten().cpu().numpy().astype(np.float32),
                "world_stability": (1.0 - d_rank).astype(np.float32),
                "joint_product": (c_rank * (1.0 - d_rank)).astype(np.float32),
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
    interface: VoxTellStateInterface,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    student = copy.deepcopy(base_network).to(device)
    student.train()
    parameters = trainable_parameters(student)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    loss_rows, pseudo_rows, metric_rows = [], [], []
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    for step in range(1, args.steps + 1):
        sample = cache[(step - 1) % len(cache)]
        # build_cache runs under inference_mode; clone targets and inputs before
        # they participate in an autograd graph.
        patch = sample["image"].to(device).clone()
        embedding = sample["embedding"].to(device).clone()
        pseudo = sample["pseudo"].to(device).clone()
        weight = torch.from_numpy(sample["weights"][source]).to(device=device, dtype=torch.float32).view_as(pseudo)
        optimizer.zero_grad(set_to_none=True)
        student_result = interface._network_forward_with_states(student, patch, embedding)
        logits = student_result["final_prediction"][:, 0:1].float()
        per_voxel = F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
        loss = (per_voxel * weight).sum() / weight.sum().clamp_min(1e-6)
        loss.backward()
        gradient_norm = float(torch.sqrt(sum(
            parameter.grad.detach().float().pow(2).sum() for parameter in parameters if parameter.grad is not None
        )).detach().cpu())
        optimizer.step()
        parameter_delta = float(torch.sqrt(sum(
            (parameter.detach() - initial).float().pow(2).sum()
            for parameter, initial in zip(parameters, initial_parameters, strict=True)
        )).detach().cpu())
        loss_rows.append({
            "variant": variant, "step": step, "case": sample["case"], "patch_index": sample["patch_index"],
            "loss": float(loss.detach().cpu()), "gradient_norm": gradient_norm,
            "student_parameter_delta_norm": parameter_delta,
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "max_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        })
    student.eval()
    for sample in cache:
        patch = sample["image"].to(device)
        embedding = sample["embedding"].to(device)
        with torch.inference_mode():
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
            "reliability_source": source, "pseudo_label_voxels": int(correct.size),
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root), voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root), split_json=Path(args.split_json),
    )
    teacher = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
    prepare_functional_seg_head(teacher, args.selected_stage)
    cases = iter_cases(paths, split="test", limit=args.val_cases)
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(
        checkpoint_path, int(checkpoint["state_dict"]["output_projection.bias"].shape[0]), device, args.hidden_channels,
    )
    cache = build_cache(
        teacher, world_model, cases, args.selected_stage, args.patches_per_case,
        args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
        args.prediction_threshold, args.label_value, device,
    )
    base_network = copy.deepcopy(teacher.network).to(device).eval()
    for parameter in base_network.parameters():
        parameter.requires_grad = False
    loss_rows, pseudo_rows, metric_rows = [], [], []
    for variant, source in VARIANTS.items():
        variant_loss, variant_pseudo, variant_metrics = train_variant(
            variant, source, base_network, cache, teacher, args, device,
        )
        loss_rows.extend(variant_loss)
        pseudo_rows.extend(variant_pseudo)
        metric_rows.extend(variant_metrics)
    write_csv(output_dir / "training_loss.csv", loss_rows)
    write_csv(output_dir / "pseudo_label_stats.csv", pseudo_rows)
    write_csv(output_dir / "metrics.csv", metric_rows)
    by_case = []
    for variant in VARIANTS:
        selected = [row for row in metric_rows if row["variant"] == variant]
        for case in sorted({row["case"] for row in selected}):
            case_rows = [row for row in selected if row["case"] == case]
            by_case.append({
                "variant": variant, "case": case, "patches": len(case_rows),
                "dice": float(np.mean([row["dice"] for row in case_rows])),
                "precision": float(np.mean([row["precision"] for row in case_rows])),
                "recall": float(np.mean([row["recall"] for row in case_rows])),
                "tp": int(sum(row["tp"] for row in case_rows)), "fp": int(sum(row["fp"] for row in case_rows)),
                "tn": int(sum(row["tn"] for row in case_rows)), "fn": int(sum(row["fn"] for row in case_rows)),
            })
    write_csv(output_dir / "by_case.csv", by_case)
    summary = {
        "stage": "V7.0 minimal SFDA smoke",
        "source_checkpoint": str(checkpoint_path),
        "voxtell_model_dir": str(paths.voxtell_model_dir),
        "selected_stage": args.selected_stage,
        "seed": args.seed,
        "config": vars(args),
        "cases": [case.case for case in cases],
        "patches": len(cache),
        "variants": VARIANTS,
        "training": {
            "loss": "weighted pseudo-label BCE with fixed VoxTell teacher labels",
            "student_trainable_scope": "decoder.* and project_to_decoder_channels.* only",
            "same_initialization_data_steps_and_lr": True,
            "world_predictor_updated": False,
            "new_loss_or_module": False,
            "ema_or_sfda_extra": False,
        },
        "outputs": {
            "training_loss": str(output_dir / "training_loss.csv"),
            "pseudo_label_stats": str(output_dir / "pseudo_label_stats.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "by_case": str(output_dir / "by_case.csv"),
        },
        "status": "smoke_complete; no V7 hyperparameter search or scale-up performed",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
