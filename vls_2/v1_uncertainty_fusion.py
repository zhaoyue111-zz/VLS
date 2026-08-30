"""Uncertainty-guided pseudo-label weighting over frozen multi-action WP predictions.

This runner keeps the existing VoxTell encoder -> World Predictor -> native
decoder -> LoRA adaptation path intact. It does not train a World Predictor
and does not create real image augmentations for uncertainty estimation.
Uncertainty is computed only from multiple decoded predictions produced by
feeding source encoder features through the frozen V9.3 hierarchical WP with
different visual action vectors.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths


OUTPUT_DIR = Path("output_1/v1_uncertainty_fusion")
WEIGHT_METHODS = {
    "A_confidence_only": "confidence",
    "B_wp_uncertainty_only": "wp_stability",
    "C_confidence_wp_uncertainty": "fusion",
}
DEFAULT_WP_ACTIONS = (
    "gamma:0.30",
    "blur:1.50",
    "gamma:0.15",
    "blur:0.75",
    "gamma:-0.15",
)


def load_runtime_dependencies() -> None:
    """Import VoxTell/nnUNet-dependent helpers only when an experiment runs."""
    global SOURCE_PROMPT, LEVEL_NAMES
    global VoxTellStateInterface
    global build_text_delta, class_balanced_loss, evaluate_full_volume, inject_lora_qkv
    global iter_cases, level_features, load_v93_world_predictor, lora_parameters
    global native_decoder_from_predicted_skips, pad_label_like_image, pool_full_volume
    global read_image_and_label, resolve_device, select_patch_slicers, set_seed, visual_action, write_csv

    from vls.data import iter_cases, read_image_and_label
    from vls.v2_experiment import resolve_device, select_patch_slicers, visual_action
    from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, pad_label_like_image
    from vls.v7_0d_protocol_sanity import set_seed
    from vls.v7_1a_lora_qkv_smoke import inject_lora_qkv, lora_parameters
    from vls.v7_1b_protocol_consolidation import evaluate_full_volume, pool_full_volume, write_csv
    from vls.v7_1c_class_balanced_loss_sanity import class_balanced_loss
    if sys.platform == "win32" and "resource" not in sys.modules:
        resource_shim = types.ModuleType("resource")
        resource_shim.RUSAGE_SELF = 0
        resource_shim.getrusage = lambda _who: types.SimpleNamespace(ru_maxrss=0)
        sys.modules["resource"] = resource_shim
    from vls.v9_3_hierarchical_residual_world_predictor import (
        LEVEL_NAMES,
        level_features,
        native_decoder_from_predicted_skips,
    )
    from vls.v9_4_hierarchical_wp_matched_diagnostic import build_text_delta, load_v93_world_predictor
    from vls.voxtell_states import VoxTellStateInterface


@dataclass(frozen=True)
class WpAction:
    family: str
    strength: float

    @property
    def label(self) -> str:
        return f"{self.family}:{self.strength:g}"


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(
        description="Confidence + frozen multi-action WP uncertainty pseudo-label weighting."
    )
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/v9_3_hierarchical_residual_world_predictor/best_hierarchical_world_predictor.pt",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--train-cases", type=int, default=0, help="0 uses all train cases")
    parser.add_argument("--evaluation-cases", type=int, default=0, help="0 uses all test cases")
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--training-rounds", type=int, default=5)
    parser.add_argument("--wp-sample-counts", default="1,3,5")
    parser.add_argument("--wp-actions", default=",".join(DEFAULT_WP_ACTIONS))
    parser.add_argument("--uncertainty-formula", choices=("variance", "entropy"), default="variance")
    parser.add_argument("--fusion-mode", choices=("alpha", "product"), default="alpha")
    parser.add_argument("--fusion-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run one train/eval case, one patch, one round, and N=1/3 only.",
    )
    return parser.parse_args()


def make_paths(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )


def parse_sample_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError(f"--wp-sample-counts must contain positive integers: {value}")
    return sorted(dict.fromkeys(counts))


def parse_wp_actions(value: str) -> list[WpAction]:
    actions: list[WpAction] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        family, sep, raw_strength = item.partition(":")
        if sep != ":":
            raise ValueError(f"WP action must be family:strength, got {item!r}")
        family = family.strip()
        if family not in {"gamma", "blur"}:
            raise ValueError(f"V9.3 visual WP supports only gamma/blur actions, got {family!r}")
        actions.append(WpAction(family, float(raw_strength)))
    if not actions:
        raise ValueError("--wp-actions cannot be empty")
    return actions


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    low = float(values[finite].min())
    high = float(values[finite].max())
    if high - low <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    out = (values - low) / (high - low)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def binary_iou_from_counts(metrics: dict[str, Any]) -> float:
    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    return 1.0 if tp + fp + fn == 0 else float(tp / max(tp + fp + fn, 1))


def enrich_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    iou = binary_iou_from_counts(row)
    row["iou"] = iou
    row["foreground_dice"] = row["dice"]
    row["foreground_iou"] = iou
    return row


def array_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(values, dtype=np.float64)),
        f"{prefix}_std": float(np.std(values, dtype=np.float64)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_max": float(np.max(values)),
    }


def uncertainty_map(probabilities: torch.Tensor, formula: str) -> torch.Tensor:
    if probabilities.ndim < 5:
        raise AssertionError(f"Expected WP probability stack with N,B,C,D,H,W axes, got {probabilities.shape}")
    if int(probabilities.shape[0]) == 1:
        return torch.zeros_like(probabilities[0])
    if formula == "variance":
        return torch.var(probabilities.float(), dim=0, unbiased=False)
    if formula == "entropy":
        mean_probability = probabilities.float().mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
        return -(
            mean_probability * torch.log(mean_probability)
            + (1.0 - mean_probability) * torch.log(1.0 - mean_probability)
        )
    raise ValueError(f"Unsupported uncertainty formula: {formula}")


def fusion_weights(
    confidence: np.ndarray,
    wp_stability: np.ndarray,
    mode: str,
    alpha: float,
) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float32)
    wp_stability = np.asarray(wp_stability, dtype=np.float32)
    if mode == "product":
        return np.clip(confidence * wp_stability, 0.0, 1.0).astype(np.float32)
    if mode == "alpha":
        return np.clip(alpha * confidence + (1.0 - alpha) * wp_stability, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported fusion mode: {mode}")


@torch.inference_mode()
def wp_probabilities_from_source_features(
    interface: VoxTellStateInterface,
    world_predictor: torch.nn.Module,
    source_context: dict[str, Any],
    actions: list[WpAction],
    text_delta: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    source_features = level_features(source_context["decoder_audit"]["skips"], LEVEL_NAMES)
    probabilities: list[torch.Tensor] = []
    for action in actions:
        predicted_features = world_predictor(
            source_features,
            visual_action(action.family, action.strength, device),
            text_delta,
        )
        logits = native_decoder_from_predicted_skips(
            interface, source_context, predicted_features, LEVEL_NAMES,
        )
        probabilities.append(torch.sigmoid(logits.detach().float()))
        del predicted_features, logits
    return torch.stack(probabilities, dim=0)


def pseudo_accuracy(pseudo: np.ndarray, gt: np.ndarray) -> float:
    pseudo = np.asarray(pseudo, dtype=bool).reshape(-1)
    gt = np.asarray(gt, dtype=bool).reshape(-1)
    if pseudo.shape != gt.shape:
        raise AssertionError(f"Pseudo/GT shape mismatch: {pseudo.shape} vs {gt.shape}")
    return float(np.mean(pseudo == gt))


@torch.inference_mode()
def build_train_cache(
    interface: VoxTellStateInterface,
    world_predictor: torch.nn.Module,
    cases: list[CaseRecord],
    prompt_embedding: torch.Tensor,
    text_delta: torch.Tensor,
    wp_actions: list[WpAction],
    sample_counts: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max(sample_counts) > len(wp_actions):
        raise AssertionError(f"Need at least {max(sample_counts)} WP actions, got {len(wp_actions)}")
    cache: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        image, label, _ = read_image_and_label(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface,
            image,
            prompt_embedding,
            args.patches_per_case,
            args.foreground_patches_per_case,
            args.foreground_candidate_patches,
            args.foreground_threshold,
        )
        label_padded = pad_label_like_image(interface, label)
        for patch_index, slicer in enumerate(slicers):
            patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            source_context = interface.forward_with_audit_context(patch, prompt_embedding)
            source_probability = torch.sigmoid(source_context["final_prediction"][:, :1].detach().float())
            source_probability_np = source_probability.flatten().cpu().numpy().astype(np.float32)
            confidence = np.maximum(source_probability_np, 1.0 - source_probability_np).astype(np.float32)
            pseudo = (source_probability > args.prediction_threshold).float()
            gt = (label_padded[slicer][None].to(device) == args.label_value).float()
            if tuple(gt.shape[-3:]) != tuple(source_probability.shape[-3:]):
                import torch.nn.functional as F

                gt = F.interpolate(gt, size=tuple(int(size) for size in source_probability.shape[-3:]), mode="nearest")
            pseudo_np = pseudo.flatten().cpu().numpy().astype(bool)
            gt_np = gt.flatten().cpu().numpy().astype(bool)
            full_wp_stack = wp_probabilities_from_source_features(
                interface, world_predictor, source_context, wp_actions[: max(sample_counts)], text_delta, device
            )
            weights: dict[str, np.ndarray] = {}
            uncertainty_stats: dict[str, dict[str, Any]] = {}
            for sample_count in sample_counts:
                active = wp_actions[:sample_count]
                stack = full_wp_stack[:sample_count]
                raw_uncertainty = uncertainty_map(stack, args.uncertainty_formula)
                raw_uncertainty_np = raw_uncertainty.flatten().cpu().numpy().astype(np.float32)
                normalized_uncertainty = minmax_normalize(raw_uncertainty_np)
                wp_stability = (1.0 - normalized_uncertainty).astype(np.float32)
                suffix = f"N{sample_count}"
                weights[f"confidence:{suffix}"] = confidence
                weights[f"wp_stability:{suffix}"] = wp_stability
                weights[f"fusion:{suffix}"] = fusion_weights(
                    confidence, wp_stability, args.fusion_mode, args.fusion_alpha
                )
                uncertainty_stats[suffix] = {
                    "wp_sample_count": sample_count,
                    "wp_actions": [action.label for action in active],
                    "pseudo_label_accuracy": pseudo_accuracy(pseudo_np, gt_np),
                    **array_stats("confidence", confidence),
                    **array_stats("uncertainty_raw", raw_uncertainty_np),
                    **array_stats("uncertainty_normalized", normalized_uncertainty),
                    **array_stats("wp_stability", wp_stability),
                }
            cache.append({
                "case": case.case,
                "case_index": case_index,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "image": patch.detach().cpu(),
                "embedding": prompt_embedding.detach().cpu(),
                "pseudo": pseudo.detach().cpu(),
                "pseudo_np": pseudo_np,
                "gt_np": gt_np,
                "weights": weights,
                "weight_stats": uncertainty_stats,
                "has_foreground": bool(np.count_nonzero(pseudo_np)),
            })
            manifest.append({
                "case": case.case,
                "case_index": case_index,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "slicer": [[item.start, item.stop, item.step] for item in slicer],
            })
            del patch, source_context, source_probability, pseudo, gt, full_wp_stack
        del image, label, original_padded, label_padded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return cache, manifest


def train_variant(
    variant: str,
    weight_key: str,
    wp_sample_count: int,
    method: str,
    base_network: torch.nn.Module,
    train_samples: list[dict[str, Any]],
    eval_cases: list[CaseRecord],
    full_data: dict[str, tuple[np.ndarray, np.ndarray]],
    interface: VoxTellStateInterface,
    embedding: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    base_total: int,
    target_names: list[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(args.seed)
    student = copy.deepcopy(base_network)
    targets = inject_lora_qkv(student, args.lora_rank, args.lora_alpha, args.lora_dropout)
    if target_names is not None and targets != target_names:
        raise AssertionError("LoRA target modules differ across variants")
    student = student.to(device)
    trainable = lora_parameters(student)
    base_trainable = sum(
        parameter.numel()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    if base_trainable != 0:
        raise AssertionError(f"Base VoxTell parameters must remain frozen, got {base_trainable} trainable")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    ordered = train_samples * args.training_rounds
    for step, sample in enumerate(ordered, start=1):
        image = sample["image"].to(device).clone()
        embedding_batch = sample["embedding"].to(device).clone().float()
        pseudo = sample["pseudo"].to(device).clone()
        weight = torch.from_numpy(sample["weights"][weight_key]).to(device=device, dtype=torch.float32).view_as(pseudo)
        optimizer.zero_grad(set_to_none=True)
        result = interface._network_forward_with_states(student, image, embedding_batch)
        logits = result["final_prediction"][:, :1].float()
        loss, stats = class_balanced_loss(logits, pseudo, weight)
        loss.backward()
        gradient_norm = float(torch.sqrt(sum(
            parameter.grad.detach().float().pow(2).sum()
            for parameter in trainable
            if parameter.grad is not None
        )).detach().cpu())
        before = [parameter.detach().clone() for parameter in trainable]
        optimizer.step()
        update_delta = float(torch.sqrt(sum(
            (parameter.detach() - previous).float().pow(2).sum()
            for parameter, previous in zip(trainable, before, strict=True)
        )).detach().cpu())
        round_index = (step - 1) // len(train_samples) + 1
        loss_rows.append({
            "variant": variant,
            "weight_method": method,
            "wp_sample_count": wp_sample_count,
            "step": step,
            "round": round_index,
            "case": sample["case"],
            "patch_index": sample["patch_index"],
            "patch_kind": sample["patch_kind"],
            "loss": float(loss.detach().cpu()),
            "gradient_norm": gradient_norm,
            "update_delta_norm": update_delta,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "base_trainable_parameters": base_trainable,
            "lora_trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        })
        stats_key = f"N{wp_sample_count}"
        weight_np = np.asarray(sample["weights"][weight_key], dtype=np.float32)
        correct = sample["pseudo_np"] == sample["gt_np"]
        pseudo_rows.append({
            "variant": variant,
            "weight_method": method,
            "wp_sample_count": wp_sample_count,
            "step": step,
            "round": round_index,
            "case": sample["case"],
            "patch_index": sample["patch_index"],
            "patch_kind": sample["patch_kind"],
            "weight_key": weight_key,
            "pseudo_label_accuracy": float(np.mean(correct)),
            "weighted_pseudo_label_accuracy": float(weight_np[correct].sum() / max(float(weight_np.sum()), 1e-12)),
            "pseudo_positive_voxels": int(np.count_nonzero(sample["pseudo_np"])),
            "weight_sum": float(weight_np.sum()),
            **array_stats("weight", weight_np),
            **sample["weight_stats"][stats_key],
            **{f"loss_{key}": value for key, value in stats.items()},
        })
        del image, embedding_batch, pseudo, weight, result, logits, loss, before
    student.eval()
    final_step = len(ordered)
    full_rows = evaluate_full_volume(
        interface,
        student,
        eval_cases,
        full_data,
        embedding,
        variant,
        "fixed_order",
        final_step,
        args.label_value,
        args.prediction_threshold,
    )
    for row in full_rows:
        row["weight_method"] = method
        row["wp_sample_count"] = wp_sample_count
        enrich_metric_row(row)
    stats = {
        "variant": variant,
        "weight_method": method,
        "wp_sample_count": wp_sample_count,
        "target_modules": targets,
        "base_trainable_parameters": base_trainable,
        "base_total_parameters": base_total,
        "lora_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
    }
    del student, optimizer, trainable
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_rows, pseudo_rows, full_rows, stats


def select_train_samples(cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in cache:
        by_case[sample["case"]].append(sample)
    selected: list[dict[str, Any]] = []
    for case, samples in sorted(by_case.items(), key=lambda item: min(s["case_index"] for s in item[1])):
        foreground = [
            sample for sample in samples
            if sample["patch_kind"] == "foreground" and sample["has_foreground"]
        ]
        candidates = foreground or samples
        selected.append(sorted(candidates, key=lambda sample: sample["patch_index"])[0])
    return selected


def aggregate_metrics(
    full_rows: list[dict[str, Any]],
    pseudo_rows: list[dict[str, Any]],
    initial_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    curve = pool_full_volume(full_rows)
    initial_curve = next(row for row in curve if row["variant"] == "A_init_no_adaptation")
    pseudo_by_variant: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pseudo_rows:
        pseudo_by_variant[(row["variant"], int(row["wp_sample_count"]))].append(row)

    rows: list[dict[str, Any]] = []
    for row in curve:
        variant = row["variant"]
        if variant == "A_init_no_adaptation":
            rows.append({
                **row,
                "weight_method": "none",
                "wp_sample_count": 0,
                "mean_iou": row["mean_foreground_iou"],
                "mean_foreground_dice": row["mean_dice"],
                "delta_mean_dice_vs_A_init": 0.0,
                "delta_mean_iou_vs_A_init": 0.0,
            })
            continue
        sample_count = int(row.get("wp_sample_count", str(variant).split("_N")[-1]))
        pseudo = pseudo_by_variant[(variant, sample_count)]
        pseudo_mean = lambda field: float(np.mean([float(item[field]) for item in pseudo], dtype=np.float64)) if pseudo else None
        rows.append({
            **row,
            "weight_method": next((item["weight_method"] for item in pseudo), row.get("weight_method")),
            "wp_sample_count": sample_count,
            "mean_iou": row["mean_foreground_iou"],
            "mean_foreground_dice": row["mean_dice"],
            "delta_mean_dice_vs_A_init": row["mean_dice"] - initial_curve["mean_dice"],
            "delta_mean_iou_vs_A_init": row["mean_foreground_iou"] - initial_curve["mean_foreground_iou"],
            "pseudo_label_accuracy_mean": pseudo_mean("pseudo_label_accuracy"),
            "weighted_pseudo_label_accuracy_mean": pseudo_mean("weighted_pseudo_label_accuracy"),
            "confidence_mean": pseudo_mean("confidence_mean"),
            "confidence_std": pseudo_mean("confidence_std"),
            "uncertainty_raw_mean": pseudo_mean("uncertainty_raw_mean"),
            "uncertainty_raw_std": pseudo_mean("uncertainty_raw_std"),
            "uncertainty_normalized_mean": pseudo_mean("uncertainty_normalized_mean"),
            "uncertainty_normalized_std": pseudo_mean("uncertainty_normalized_std"),
            "wp_stability_mean": pseudo_mean("wp_stability_mean"),
            "wp_stability_std": pseudo_mean("wp_stability_std"),
            "weight_mean": pseudo_mean("weight_mean"),
            "weight_std": pseudo_mean("weight_std"),
        })
    for row in rows:
        row.setdefault("case_count", len({item["case"] for item in initial_rows}))
    return rows


def add_baseline_deltas(full_rows: list[dict[str, Any]], initial_rows: list[dict[str, Any]]) -> None:
    baseline_by_case = {row["case"]: row for row in initial_rows}
    for row in full_rows:
        if row["variant"] == "A_init_no_adaptation":
            row["delta_dice_vs_A_init"] = 0.0
            row["delta_iou_vs_A_init"] = 0.0
            row["delta_foreground_dice_vs_A_init"] = 0.0
            continue
        baseline = baseline_by_case[row["case"]]
        row["delta_dice_vs_A_init"] = row["dice"] - baseline["dice"]
        row["delta_iou_vs_A_init"] = row["iou"] - baseline["iou"]
        row["delta_foreground_dice_vs_A_init"] = row["foreground_dice"] - baseline["foreground_dice"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def run(args: argparse.Namespace) -> None:
    if args.smoke_only:
        args.train_cases = 1
        args.evaluation_cases = 1
        args.patches_per_case = 1
        args.foreground_patches_per_case = 1
        args.training_rounds = 1
        args.wp_sample_counts = "1,3"
    if not 0.0 <= args.fusion_alpha <= 1.0:
        raise AssertionError("--fusion-alpha must be in [0, 1]")
    if args.training_rounds <= 0:
        raise AssertionError("--training-rounds must be positive")

    sample_counts = parse_sample_counts(args.wp_sample_counts)
    wp_actions = parse_wp_actions(args.wp_actions)
    if max(sample_counts) > len(wp_actions):
        raise AssertionError(f"Requested N={max(sample_counts)} but only {len(wp_actions)} --wp-actions were provided")

    load_runtime_dependencies()
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"VLS uncertainty fusion requires CUDA, resolved {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = make_paths(args)
    train_cases = iter_cases(paths, split="train", limit=args.train_cases if args.train_cases else None)
    eval_cases = iter_cases(paths, split="test", limit=args.evaluation_cases if args.evaluation_cases else None)
    train_names = [case.case for case in train_cases]
    eval_names = [case.case for case in eval_cases]
    overlap = sorted(set(train_names) & set(eval_names))
    if overlap:
        raise AssertionError(f"train/evaluation case overlap: {overlap}")
    if not train_cases or not eval_cases:
        raise AssertionError("Non-empty train and evaluation case sets are required")

    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir,
        device=device,
        voxtell_root=paths.voxtell_root,
    )
    world_predictor, checkpoint_metadata = load_v93_world_predictor(Path(args.world_checkpoint), device)
    prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    text_delta = build_text_delta(interface, device)

    train_cache, manifest = build_train_cache(
        interface,
        world_predictor,
        train_cases,
        prompt_embedding,
        text_delta,
        wp_actions,
        sample_counts,
        args,
        device,
    )
    train_samples = select_train_samples(train_cache)
    if len(train_samples) != len(train_cases):
        raise AssertionError("Expected exactly one effective train sample per train case")
    full_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in eval_cases:
        image, label, _ = read_image_and_label(case)
        full_data[case.case] = (image, label)

    world_predictor.to("cpu")
    interface.network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    base_network = copy.deepcopy(interface.network).cpu().eval()
    for parameter in base_network.parameters():
        parameter.requires_grad = False
    base_total = sum(parameter.numel() for parameter in base_network.parameters())
    base_network.to(device)
    initial_rows = evaluate_full_volume(
        interface,
        base_network,
        eval_cases,
        full_data,
        prompt_embedding,
        "A_init_no_adaptation",
        "fixed_order",
        0,
        args.label_value,
        args.prediction_threshold,
    )
    for row in initial_rows:
        row["weight_method"] = "none"
        row["wp_sample_count"] = 0
        enrich_metric_row(row)
    base_network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_loss_rows: list[dict[str, Any]] = []
    all_pseudo_rows: list[dict[str, Any]] = []
    all_full_rows: list[dict[str, Any]] = list(initial_rows)
    parameter_stats: list[dict[str, Any]] = []
    target_names: list[str] | None = None
    for sample_count in sample_counts:
        for method_variant, source in WEIGHT_METHODS.items():
            weight_key = f"{source}:N{sample_count}"
            variant = f"{method_variant}_N{sample_count}"
            print(f"[vX_uncertainty_fusion] training {variant}", flush=True)
            losses, pseudo_rows, full_rows, stats = train_variant(
                variant,
                weight_key,
                sample_count,
                source,
                base_network,
                train_samples,
                eval_cases,
                full_data,
                interface,
                prompt_embedding,
                args,
                device,
                base_total,
                target_names,
            )
            target_names = stats["target_modules"] if target_names is None else target_names
            all_loss_rows.extend(losses)
            all_pseudo_rows.extend(pseudo_rows)
            all_full_rows.extend(full_rows)
            parameter_stats.append(stats)

    add_baseline_deltas(all_full_rows, initial_rows)
    metrics_rows = aggregate_metrics(all_full_rows, all_pseudo_rows, initial_rows)

    write_csv(output_dir / "training_loss.csv", all_loss_rows)
    write_csv(output_dir / "pseudo_label_stats.csv", all_pseudo_rows)
    write_csv(output_dir / "per_case.csv", all_full_rows)
    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_json(output_dir / "train_patch_manifest.json", {
        "stage": "vX uncertainty fusion train patch manifest",
        "selection_source": "source teacher prediction only; no GT-based selection",
        "records": manifest,
    })
    write_json(output_dir / "parameter_stats.json", {"variants": parameter_stats})

    final_by_variant = [
        row for row in metrics_rows
        if row["variant"] != "A_init_no_adaptation"
    ]
    summary = {
        "stage": "vX uncertainty-guided pseudo-label weighting",
        "smoke_only": bool(args.smoke_only),
        "output_dir": str(output_dir),
        "source_model_modified": False,
        "world_predictor_trained": False,
        "world_predictor_modified": False,
        "real_image_augmentation_generated_for_uncertainty": False,
        "tta_implementation": False,
        "train_cases": train_names,
        "evaluation_cases": eval_names,
        "case_overlap": overlap,
        "world_checkpoint": checkpoint_metadata,
        "wp_actions": [action.label for action in wp_actions],
        "wp_sample_counts": sample_counts,
        "weight_methods": WEIGHT_METHODS,
        "formula": {
            "source_probability": "p = sigmoid(source_decoder_logits)",
            "confidence": "confidence = max(p, 1-p), used directly without percentile ranking",
            "wp_prediction_path": "image -> source encoder/skips -> frozen V9.3 WP(z,a_i) -> unchanged native decoder -> pred_i",
            "uncertainty_variance": "U = var_i(pred_i) with unbiased=False",
            "uncertainty_entropy": "U = binary entropy(mean_i(pred_i))",
            "normalization": "U_norm = per-patch minmax(U); constant U maps to zeros",
            "wp_uncertainty_only_weight": "w = 1 - U_norm",
            "fusion_alpha": f"w = {args.fusion_alpha:g} * confidence + {1.0 - args.fusion_alpha:g} * (1 - U_norm)",
            "fusion_product": "w = confidence * (1 - U_norm)",
            "active_fusion_mode": args.fusion_mode,
        },
        "training": {
            "loss": "existing class_balanced_loss from vls.v7_1c_class_balanced_loss_sanity",
            "student": "fresh LoRA-injected copy of frozen VoxTell network per variant",
            "base_voxtell_parameters_frozen": True,
            "lora": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": target_names,
            },
            "training_rounds": args.training_rounds,
            "updates_per_variant": len(train_samples) * args.training_rounds,
            "sample_order": "same selected source patches for all variants and N values",
        },
        "final_metrics": final_by_variant,
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "metrics": str(output_dir / "metrics.csv"),
            "per_case": str(output_dir / "per_case.csv"),
            "pseudo_label_stats": str(output_dir / "pseudo_label_stats.csv"),
            "training_loss": str(output_dir / "training_loss.csv"),
            "train_patch_manifest": str(output_dir / "train_patch_manifest.json"),
            "parameter_stats": str(output_dir / "parameter_stats.json"),
        },
        "status": "complete" if not args.smoke_only else "smoke_complete",
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(output_dir),
        "metrics_rows": len(metrics_rows),
        "per_case_rows": len(all_full_rows),
        "variants": [row["variant"] for row in final_by_variant],
    }, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
