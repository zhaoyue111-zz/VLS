"""Frozen VoxTell MRI-domain action sensitivity screening.

This is a diagnostic-only experiment.  It evaluates real intensity-domain
actions against the frozen VoxTell model and never imports or trains a World
Predictor, runs SFDA, or changes the source model.  Spatial geometry and
labels are kept unchanged by construction: every action changes intensities
only, while all actions use the source case's patch slicers.

The default action grid has two strengths.  Gamma and the signed actions keep
their direction in the per-sample output, while action/strength summaries
remain separate from one another.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from vls.config import DEFAULT_LABEL_VALUE, DEFAULT_PROMPTS, ProjectPaths


OUTPUT_DIR = Path("output_2/v0_mri_action_screening")
EPS = 1e-8


@dataclass(frozen=True)
class ActionVariant:
    action_family: str
    strength: str
    direction: str
    parameter_name: str
    parameter_value: float

    @property
    def variant_id(self) -> str:
        return f"{self.action_family}:{self.strength}:{self.direction}"


DEFAULT_ACTION_VARIANTS: tuple[ActionVariant, ...] = (
    ActionVariant("gamma", "mild", "brighten", "gamma", 0.90),
    ActionVariant("gamma", "mild", "darken", "gamma", 1.10),
    ActionVariant("gamma", "moderate", "brighten", "gamma", 0.70),
    ActionVariant("gamma", "moderate", "darken", "gamma", 1.30),
    ActionVariant("intensity_scale", "mild", "down", "scale", 0.90),
    ActionVariant("intensity_scale", "mild", "up", "scale", 1.10),
    ActionVariant("intensity_scale", "moderate", "down", "scale", 0.70),
    ActionVariant("intensity_scale", "moderate", "up", "scale", 1.30),
    ActionVariant("intensity_shift", "mild", "down", "range_fraction", -0.05),
    ActionVariant("intensity_shift", "mild", "up", "range_fraction", 0.05),
    ActionVariant("intensity_shift", "moderate", "down", "range_fraction", -0.15),
    ActionVariant("intensity_shift", "moderate", "up", "range_fraction", 0.15),
    ActionVariant("gaussian_noise", "mild", "fixed", "std_range_fraction", 0.02),
    ActionVariant("gaussian_noise", "moderate", "fixed", "std_range_fraction", 0.05),
    ActionVariant("bias_field", "mild", "fixed", "amplitude", 0.05),
    ActionVariant("bias_field", "moderate", "fixed", "amplitude", 0.15),
)

SMOKE_ACTION_VARIANTS: tuple[ActionVariant, ...] = (
    DEFAULT_ACTION_VARIANTS[0],
    DEFAULT_ACTION_VARIANTS[4],
    DEFAULT_ACTION_VARIANTS[8],
    DEFAULT_ACTION_VARIANTS[12],
    DEFAULT_ACTION_VARIANTS[14],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V0 frozen VoxTell MRI action screening; intensity-only actions, "
            "no blur, World Predictor, or SFDA."
        )
    )
    parser.add_argument("--voxtell-root", default=str(ProjectPaths().voxtell_root))
    parser.add_argument("--model-dir", default=str(ProjectPaths().voxtell_model_dir))
    parser.add_argument("--data-root", default=str(ProjectPaths().data_root))
    parser.add_argument("--split-json", default=str(ProjectPaths().split_json))
    parser.add_argument("--split", default="train", choices=["train", "test", "train_cases", "test_cases"])
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--nonzero-delta-threshold", type=float, default=1e-4)
    parser.add_argument("--severe-performance-drop", type=float, default=0.10)
    parser.add_argument(
        "--meaningful-negative-drop",
        type=float,
        default=0.005,
        help="Count a Dice/mIoU decrease as negative only when change <= -this value.",
    )
    parser.add_argument("--persistent-negative-rate", type=float, default=0.50)
    parser.add_argument("--bias-sigma-fraction", type=float, default=0.12)
    parser.add_argument("--max-patches-per-case", type=int, default=0)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run exactly 1 case, 1 encoder patch, and 5 representative mild actions.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def stable_seed(case_name: str, variant: ActionVariant, seed: int) -> int:
    token = f"{seed}|{case_name}|{variant.variant_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def support_mask(image: np.ndarray) -> np.ndarray:
    """Return the exact non-zero support used to protect the preprocessing bbox."""
    return np.asarray(image) != 0


def robust_range(image: np.ndarray, support: np.ndarray | None = None) -> float:
    values = np.asarray(image, dtype=np.float32)
    if support is None:
        support = support_mask(values)
    supported_values = values[support]
    if supported_values.size == 0:
        return 1.0
    low, high = np.percentile(supported_values, [5.0, 95.0])
    result = float(high - low)
    if not math.isfinite(result) or result < EPS:
        result = float(np.max(supported_values) - np.min(supported_values))
    return max(result, 1.0)


def preserve_original_support(
    original: np.ndarray,
    transformed: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Keep zeros outside support and prevent an action from creating/removing support."""
    result = np.asarray(transformed, dtype=np.float32).copy()
    result[~support] = original[~support]
    newly_zero = support & (result == 0)
    if np.any(newly_zero):
        result[newly_zero] = np.copysign(np.finfo(np.float32).tiny, original[newly_zero])
    return result


def apply_action(
    image: np.ndarray,
    variant: ActionVariant,
    *,
    case_name: str,
    seed: int,
    bias_sigma_fraction: float,
) -> np.ndarray:
    """Apply one intensity-only action without changing array geometry."""
    x = np.asarray(image, dtype=np.float32)
    support = support_mask(x)
    if not np.any(support):
        return x.copy()
    value_range = robust_range(x, support)

    if variant.action_family == "gamma":
        supported_values = x[support]
        lo = float(np.min(supported_values))
        hi = float(np.max(supported_values))
        if hi - lo < EPS:
            return x.copy()
        normalized = (x - lo) / (hi - lo)
        transformed = np.power(np.clip(normalized, 0.0, 1.0), variant.parameter_value)
        return preserve_original_support(x, transformed * (hi - lo) + lo, support)

    if variant.action_family == "intensity_scale":
        return preserve_original_support(x, x * variant.parameter_value, support)

    if variant.action_family == "intensity_shift":
        return preserve_original_support(x, x + variant.parameter_value * value_range, support)

    rng = np.random.default_rng(stable_seed(case_name, variant, seed))
    if variant.action_family == "gaussian_noise":
        noise = rng.normal(0.0, variant.parameter_value * value_range, size=x.shape)
        return preserve_original_support(x, x + noise.astype(np.float32), support)

    if variant.action_family == "bias_field":
        from scipy.ndimage import gaussian_filter

        spatial_shape = x.shape[-3:]
        noise = rng.normal(0.0, 1.0, size=spatial_shape).astype(np.float32)
        sigma = max(1.0, float(min(spatial_shape)) * bias_sigma_fraction)
        field = gaussian_filter(noise, sigma=sigma, mode="reflect")
        field = field - float(field.mean())
        field = field / max(float(field.std()), EPS)
        multiplier = np.clip(1.0 + variant.parameter_value * field, 0.5, 1.5)
        if x.ndim == 4:
            multiplier = multiplier[None, ...]
        return preserve_original_support(x, x * multiplier, support)

    raise ValueError(f"Unsupported action family: {variant.action_family}")


def action_variants() -> tuple[ActionVariant, ...]:
    return DEFAULT_ACTION_VARIANTS


def padded_preprocessed_and_slicers(
    predictor: Any,
    preprocessed: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple], tuple[int, ...]]:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image

    padded, _ = pad_nd_image(
        preprocessed,
        predictor.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    slicers = predictor._internal_get_sliding_window_slicers(padded.shape[1:])
    return padded, slicers, tuple(int(value) for value in preprocessed.shape)


def preprocess_image(
    predictor: Any,
    image: np.ndarray,
) -> tuple[torch.Tensor, list[tuple], tuple[int, ...]]:
    preprocessed, _, _ = predictor.preprocess(image)
    preprocessed = torch.as_tensor(preprocessed).float()
    return padded_preprocessed_and_slicers(predictor, preprocessed)


def prepare_action_input(
    predictor: Any,
    image: np.ndarray,
    variant: ActionVariant,
    *,
    case_name: str,
    seed: int,
    bias_sigma_fraction: float,
) -> tuple[torch.Tensor, list[tuple]]:
    action_image = apply_action(
        image,
        variant,
        case_name=case_name,
        seed=seed,
        bias_sigma_fraction=bias_sigma_fraction,
    )
    padded, slicers, _ = preprocess_image(predictor, action_image)
    return padded, slicers


def clone_patch(padded: torch.Tensor, slicer: tuple) -> torch.Tensor:
    return torch.clone(padded[slicer][None], memory_format=torch.contiguous_format)


def assert_action_geometry_unchanged(
    source_image: np.ndarray,
    action_image: np.ndarray,
    source_preprocessed_shape: tuple[int, ...],
    action_preprocessed_shape: tuple[int, ...],
    source_padded: torch.Tensor,
    action_padded: torch.Tensor,
    source_slicers: list[tuple],
    action_slicers: list[tuple],
) -> dict[str, Any]:
    """Fail fast if an intensity action can alter preprocessing geometry."""
    if source_image.shape != action_image.shape:
        raise ValueError(
            f"Action changed raw image shape: {source_image.shape} vs {action_image.shape}"
        )
    source_support = support_mask(source_image)
    action_support = support_mask(action_image)
    raw_support_identical = bool(np.array_equal(source_support, action_support))
    raw_outside_values_identical = bool(
        np.array_equal(action_image[~source_support], source_image[~source_support])
    )
    if not raw_support_identical or not raw_outside_values_identical:
        raise ValueError("Action changed raw non-zero support or values outside support")

    preprocessed_shape_identical = source_preprocessed_shape == action_preprocessed_shape
    padded_shape_identical = tuple(source_padded.shape) == tuple(action_padded.shape)
    sliding_window_topology_identical = source_slicers == action_slicers
    if not preprocessed_shape_identical or not padded_shape_identical or not sliding_window_topology_identical:
        raise ValueError(
            "Action changed preprocessing/sliding-window geometry: "
            f"preprocessed={source_preprocessed_shape}/{action_preprocessed_shape}, "
            f"padded={tuple(source_padded.shape)}/{tuple(action_padded.shape)}, "
            f"topology={sliding_window_topology_identical}"
        )
    return {
        "raw_support_voxels_before": int(np.count_nonzero(source_support)),
        "raw_support_voxels_after": int(np.count_nonzero(action_support)),
        "raw_support_identical": raw_support_identical,
        "raw_outside_support_values_identical": raw_outside_values_identical,
        "preprocessed_shape_before": json.dumps([int(value) for value in source_preprocessed_shape]),
        "preprocessed_shape_after": json.dumps([int(value) for value in action_preprocessed_shape]),
        "preprocessed_shape_identical": preprocessed_shape_identical,
        "padded_shape_identical": padded_shape_identical,
        "sliding_window_count_before": len(source_slicers),
        "sliding_window_count_after": len(action_slicers),
        "sliding_window_topology_identical": sliding_window_topology_identical,
    }


def tensor_relative_delta(source: torch.Tensor, action: torch.Tensor) -> float:
    source_f = source.detach().float()
    action_f = action.detach().float()
    numerator = torch.linalg.vector_norm((action_f - source_f).reshape(-1), ord=2)
    denominator = torch.linalg.vector_norm(source_f.reshape(-1), ord=2).clamp_min(EPS)
    return float((numerator / denominator).cpu())


@torch.inference_mode()
def encoder_layer_deltas(
    interface: VoxTellStateInterface,
    source_padded: torch.Tensor,
    action_padded: torch.Tensor,
    slicers: list[tuple],
    max_patches: int,
) -> dict[str, list[float]]:
    if tuple(source_padded.shape) != tuple(action_padded.shape):
        raise ValueError(
            "Action changed preprocessed spatial shape; geometry must remain "
            f"identical, got {tuple(source_padded.shape)} and {tuple(action_padded.shape)}"
        )
    selected_slicers = slicers if max_patches <= 0 else slicers[:max_patches]
    if not selected_slicers:
        raise ValueError("No sliding-window patches were produced")

    network = interface.network.to(interface.device).eval()
    per_layer: dict[str, list[float]] = defaultdict(list)
    for slicer in selected_slicers:
        source_patch = clone_patch(source_padded, slicer).to(interface.device)
        action_patch = clone_patch(action_padded, slicer).to(interface.device)
        source_layers = network.encoder(source_patch)
        action_layers = network.encoder(action_patch)
        if len(source_layers) != len(action_layers):
            raise RuntimeError("Source/action encoder returned different layer counts")
        for layer_index, (source_layer, action_layer) in enumerate(zip(source_layers, action_layers)):
            if tuple(source_layer.shape) != tuple(action_layer.shape):
                raise RuntimeError("Source/action encoder layer shapes differ")
            per_layer[f"encoder_layer_{layer_index}"] .append(
                tensor_relative_delta(source_layer, action_layer)
            )
        del source_patch, action_patch, source_layers, action_layers
    return dict(per_layer)


def as_probability_array(value: Any, prompt_index: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 4:
        raise ValueError(f"Expected a 3D probability map with prompt axis, got shape {array.shape}")
    return np.asarray(array[prompt_index], dtype=np.float32)


def foreground_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    intersection = np.logical_and(pred_b, gt_b).sum(dtype=np.int64)
    union = np.logical_or(pred_b, gt_b).sum(dtype=np.int64)
    return float((intersection + EPS) / (union + EPS))


def mean_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    foreground = foreground_iou(pred_b, gt_b)
    background = foreground_iou(~pred_b, ~gt_b)
    return float((foreground + background) / 2.0)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = np.asarray(pred, dtype=bool)
    target_b = np.asarray(target, dtype=bool)
    intersection = np.logical_and(pred_b, target_b).sum(dtype=np.int64)
    denominator = pred_b.sum(dtype=np.int64) + target_b.sum(dtype=np.int64)
    return float((2.0 * intersection + EPS) / (denominator + EPS))


def probability_metrics(source: np.ndarray, action: np.ndarray) -> dict[str, float]:
    delta = np.asarray(action, dtype=np.float32) - np.asarray(source, dtype=np.float32)
    return {
        "prediction_probability_mse": float(np.mean(np.square(delta))),
        "prediction_probability_mae": float(np.mean(np.abs(delta))),
    }


def variant_case_row(
    *,
    case_name: str,
    prompt: str,
    variant: ActionVariant,
    source_probability: np.ndarray,
    action_probability: np.ndarray,
    gt: np.ndarray,
    layer_values: dict[str, list[float]],
    mask_threshold: float,
    geometry_checks: dict[str, Any],
    nonzero_delta_threshold: float,
) -> list[dict[str, Any]]:
    source_mask = source_probability >= mask_threshold
    action_mask = action_probability >= mask_threshold
    source_dice = dice_score(source_mask, gt)
    action_dice = dice_score(action_mask, gt)
    source_miou = mean_iou(source_mask, gt)
    action_miou = mean_iou(action_mask, gt)
    consistency_dice = dice_score(source_mask, action_mask)
    consistency_iou = foreground_iou(source_mask, action_mask)
    prob = probability_metrics(source_probability, action_probability)
    rows = []
    for layer_name in sorted(layer_values, key=lambda name: int(name.rsplit("_", 1)[1])):
        values = np.asarray(layer_values[layer_name], dtype=np.float64)
        rows.append({
            "case": case_name,
            "prompt": prompt,
            "action_family": variant.action_family,
            "strength": variant.strength,
            "direction": variant.direction,
            "variant_id": variant.variant_id,
            "parameter_name": variant.parameter_name,
            "parameter_value": variant.parameter_value,
            "encoder_layer": layer_name,
            "encoder_relative_delta": float(np.mean(values)),
            "encoder_relative_delta_std": float(np.std(values)),
            "encoder_nonzero_patch_fraction": float(np.mean(values > nonzero_delta_threshold)),
            "patch_count": int(values.size),
            **prob,
            "mask_consistency_dice": consistency_dice,
            "mask_consistency_iou": consistency_iou,
            "source_case_dice": source_dice,
            "action_case_dice": action_dice,
            "case_dice_change": action_dice - source_dice,
            "source_case_miou": source_miou,
            "action_case_miou": action_miou,
            "case_miou_change": action_miou - source_miou,
            "source_foreground_iou": foreground_iou(source_mask, gt),
            "action_foreground_iou": foreground_iou(action_mask, gt),
            **geometry_checks,
            "labels_reused_without_change": True,
        })
    return rows


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def std_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.std(values)) if values else 0.0


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    nonzero_delta_threshold: float,
    severe_performance_drop: float,
    meaningful_negative_drop: float,
    persistent_negative_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return action/strength summaries, family descriptions, and concrete rankings."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["prompt"]), str(row["action_family"]), str(row["strength"]))].append(row)

    summary_rows = []
    for (prompt, action_family, strength), group in sorted(grouped.items()):
        cases = {(row["case"], row["variant_id"]) for row in group}
        unique_case_rows = {}
        for row in group:
            unique_case_rows[(row["case"], row["variant_id"])] = row
        layer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            layer_groups[str(row["encoder_layer"])].append(row)
        summary_rows.append({
            "prompt": prompt,
            "action_family": action_family,
            "strength": strength,
            "variant_count": len(cases),
            "case_count": len({row["case"] for row in group}),
            "encoder_relative_delta_mean": mean_or_zero(row["encoder_relative_delta"] for row in group),
            "encoder_relative_delta_std": std_or_zero(row["encoder_relative_delta"] for row in group),
            "prediction_probability_mse_mean": mean_or_zero(row["prediction_probability_mse"] for row in group),
            "prediction_probability_mae_mean": mean_or_zero(row["prediction_probability_mae"] for row in group),
            "mask_consistency_dice_mean": mean_or_zero(row["mask_consistency_dice"] for row in group),
            "mask_consistency_iou_mean": mean_or_zero(row["mask_consistency_iou"] for row in group),
            "source_case_dice_mean": mean_or_zero(unique_case_rows[key]["source_case_dice"] for key in unique_case_rows),
            "action_case_dice_mean": mean_or_zero(unique_case_rows[key]["action_case_dice"] for key in unique_case_rows),
            "case_dice_change_mean": mean_or_zero(unique_case_rows[key]["case_dice_change"] for key in unique_case_rows),
            "source_case_miou_mean": mean_or_zero(unique_case_rows[key]["source_case_miou"] for key in unique_case_rows),
            "action_case_miou_mean": mean_or_zero(unique_case_rows[key]["action_case_miou"] for key in unique_case_rows),
            "case_miou_change_mean": mean_or_zero(unique_case_rows[key]["case_miou_change"] for key in unique_case_rows),
            "case_dice_change_std": std_or_zero(unique_case_rows[key]["case_dice_change"] for key in unique_case_rows),
            "case_miou_change_std": std_or_zero(unique_case_rows[key]["case_miou_change"] for key in unique_case_rows),
            "encoder_layers": json.dumps({
                layer: {
                    "relative_delta_mean": mean_or_zero(item["encoder_relative_delta"] for item in items),
                    "relative_delta_std": std_or_zero(item["encoder_relative_delta"] for item in items),
                }
                for layer, items in sorted(layer_groups.items())
            }, sort_keys=True),
        })

    family_summary_rows = []
    family_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_grouped[(str(row["prompt"]), str(row["action_family"]))].append(row)
    for (prompt, action_family), group in sorted(family_grouped.items()):
        unique_case_variants = {}
        for row in group:
            unique_case_variants[(row["case"], row["variant_id"])] = row
        family_summary_rows.append({
            "prompt": prompt,
            "action_family": action_family,
            "strength": "all",
            "direction": "all",
            "variant_count": len(unique_case_variants),
            "case_count": len({row["case"] for row in group}),
            "encoder_relative_delta_mean": mean_or_zero(row["encoder_relative_delta"] for row in group),
            "prediction_probability_mse_mean": mean_or_zero(row["prediction_probability_mse"] for row in group),
            "prediction_probability_mae_mean": mean_or_zero(row["prediction_probability_mae"] for row in group),
            "mask_consistency_dice_mean": mean_or_zero(row["mask_consistency_dice"] for row in group),
            "mask_consistency_iou_mean": mean_or_zero(row["mask_consistency_iou"] for row in group),
            "source_case_dice_mean": mean_or_zero(unique_case_variants[key]["source_case_dice"] for key in unique_case_variants),
            "action_case_dice_mean": mean_or_zero(unique_case_variants[key]["action_case_dice"] for key in unique_case_variants),
            "case_dice_change_mean": mean_or_zero(unique_case_variants[key]["case_dice_change"] for key in unique_case_variants),
            "source_case_miou_mean": mean_or_zero(unique_case_variants[key]["source_case_miou"] for key in unique_case_variants),
            "action_case_miou_mean": mean_or_zero(unique_case_variants[key]["action_case_miou"] for key in unique_case_variants),
            "case_miou_change_mean": mean_or_zero(unique_case_variants[key]["case_miou_change"] for key in unique_case_variants),
            "descriptive_only": True,
        })

    ranking_grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ranking_grouped[
            (
                str(row["prompt"]),
                str(row["action_family"]),
                str(row["strength"]),
                str(row["direction"]),
            )
        ].append(row)
    ranking_rows = []
    for (prompt, action_family, strength, direction), group in sorted(ranking_grouped.items()):
        case_variant_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            case_variant_groups[(str(row["case"]), str(row["variant_id"]))].append(row)
        case_variant_rows = [
            {
                "encoder_relative_delta": mean_or_zero(item["encoder_relative_delta"] for item in items),
                "prediction_probability_mse": float(items[0]["prediction_probability_mse"]),
                "case_dice_change": float(items[0]["case_dice_change"]),
                "case_miou_change": float(items[0]["case_miou_change"]),
            }
            for items in case_variant_groups.values()
        ]
        transition = [float(row["encoder_relative_delta"]) for row in case_variant_rows]
        probability_mse = [float(row["prediction_probability_mse"]) for row in case_variant_rows]
        dice_changes = [float(row["case_dice_change"]) for row in case_variant_rows]
        miou_changes = [float(row["case_miou_change"]) for row in case_variant_rows]
        mean_transition = mean_or_zero(transition)
        mean_abs_change = mean_or_zero(abs(value) for value in dice_changes)
        heterogeneity = float(np.std(dice_changes) / (mean_abs_change + EPS)) if dice_changes else 0.0
        heterogeneity_score = min(1.0, heterogeneity)
        nonzero_rate = mean_or_zero(
            (transition_value > nonzero_delta_threshold) and (mse_value > EPS)
            for transition_value, mse_value in zip(transition, probability_mse)
        )
        transition_cv = float(np.std(transition) / (mean_transition + EPS)) if mean_transition > 0 else float("inf")
        stability_score = 1.0 / (1.0 + min(10.0, transition_cv if math.isfinite(transition_cv) else 10.0))
        severe_drop_rate = mean_or_zero(
            (dice_change <= -severe_performance_drop) or (miou_change <= -severe_performance_drop)
            for dice_change, miou_change in zip(dice_changes, miou_changes)
        )
        negative_dice_change_rate = mean_or_zero(
            change <= -meaningful_negative_drop for change in dice_changes
        )
        negative_miou_change_rate = mean_or_zero(
            change <= -meaningful_negative_drop for change in miou_changes
        )
        persistent_negative_action = bool(
            negative_dice_change_rate >= persistent_negative_rate
            or negative_miou_change_rate >= persistent_negative_rate
        )
        ranking_score = (
            0.30 * nonzero_rate
            + 0.20 * stability_score
            + 0.15 * heterogeneity_score
            + 0.15 * (1.0 - severe_drop_rate)
            + 0.20 * (1.0 - (negative_dice_change_rate + negative_miou_change_rate) / 2.0)
        )
        eligible = bool(nonzero_rate >= 0.50 and not persistent_negative_action)
        ranking_rows.append({
            "prompt": prompt,
            "action_family": action_family,
            "strength": strength,
            "direction": direction,
            "variant_id": group[0]["variant_id"],
            "parameter_name": group[0]["parameter_name"],
            "parameter_value": group[0]["parameter_value"],
            "ranking_score": ranking_score,
            "eligible_preferred_action": eligible,
            "nonzero_transition_rate": nonzero_rate,
            "mean_encoder_relative_delta": mean_transition,
            "encoder_relative_delta_cv": transition_cv,
            "stability_score": stability_score,
            "case_performance_heterogeneity_score": heterogeneity_score,
            "case_dice_change_std": std_or_zero(dice_changes),
            "case_miou_change_std": std_or_zero(miou_changes),
            "severe_performance_drop_rate": severe_drop_rate,
            "negative_dice_change_rate": negative_dice_change_rate,
            "negative_miou_change_rate": negative_miou_change_rate,
            "meaningful_negative_drop": meaningful_negative_drop,
            "persistent_negative_action": persistent_negative_action,
            "mean_case_dice_change": mean_or_zero(dice_changes),
            "mean_case_miou_change": mean_or_zero(miou_changes),
            "mean_prediction_probability_mse": mean_or_zero(probability_mse),
            "strengths_evaluated": ",".join(sorted({str(row["strength"]) for row in group})),
            "selection_rule": (
                "prefer stable non-zero transition, case heterogeneity, and low severe-drop rate; "
                "exclude persistent negative Dice/mIoU actions; "
                f"eligible requires nonzero_transition_rate>=0.50 and negative rates<{persistent_negative_rate:.2f}; "
                f"meaningful negative drop={meaningful_negative_drop:.3f}; "
                f"severe drop threshold={severe_performance_drop:.3f}"
            ),
        })
    ranking_rows.sort(key=lambda row: float(row["ranking_score"]), reverse=True)
    rank = 1
    for row in ranking_rows:
        if row["persistent_negative_action"]:
            row["rank"] = None
            row["ranking_status"] = "excluded_persistent_negative"
        elif not row["eligible_preferred_action"]:
            row["rank"] = None
            row["ranking_status"] = "excluded_insufficient_transition"
        else:
            row["rank"] = rank
            row["ranking_status"] = "ranked_candidate"
            rank += 1
    return summary_rows, family_summary_rows, ranking_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from vls.data import binary_gt_from_label, iter_cases, read_image_and_label
    from vls.voxtell_states import VoxTellStateInterface

    set_seed(args.seed)
    device = resolve_device(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir,
        device=device,
        voxtell_root=paths.voxtell_root,
    )
    interface.network.eval()
    effective_limit = 1 if args.smoke_only else args.limit_cases
    cases = iter_cases(paths, split=args.split, limit=effective_limit)
    variants = SMOKE_ACTION_VARIANTS if args.smoke_only else action_variants()
    effective_max_patches = 1 if args.smoke_only else args.max_patches_per_case
    if not cases:
        raise ValueError("No cases found for the selected split")

    rows: list[dict[str, Any]] = []
    predictor = interface.predictor
    for case in cases:
        image, label_map, _ = read_image_and_label(case)
        gt = binary_gt_from_label(label_map, args.label_value).astype(bool)
        source_probability_map = predictor.predict_single_image(
            image, args.prompts, output_type="probabilities"
        )
        source_padded, source_slicers, source_preprocessed_shape = preprocess_image(predictor, image)
        for variant in variants:
            action_image = apply_action(
                image,
                variant,
                case_name=case.case,
                seed=args.seed,
                bias_sigma_fraction=args.bias_sigma_fraction,
            )
            action_probability_map = predictor.predict_single_image(
                action_image, args.prompts, output_type="probabilities"
            )
            action_padded, action_slicers, action_preprocessed_shape = preprocess_image(predictor, action_image)
            geometry_checks = assert_action_geometry_unchanged(
                image,
                action_image,
                source_preprocessed_shape,
                action_preprocessed_shape,
                source_padded,
                action_padded,
                source_slicers,
                action_slicers,
            )
            layer_values = encoder_layer_deltas(
                interface,
                source_padded,
                action_padded,
                source_slicers,
                effective_max_patches,
            )
            for prompt_index, prompt in enumerate(args.prompts):
                source_probability = as_probability_array(source_probability_map, prompt_index)
                action_probability = as_probability_array(action_probability_map, prompt_index)
                if source_probability.shape != action_probability.shape or source_probability.shape != gt.shape:
                    raise ValueError(
                        "Prediction/GT geometry mismatch: "
                        f"source={source_probability.shape}, action={action_probability.shape}, gt={gt.shape}"
                    )
                rows.extend(variant_case_row(
                    case_name=case.case,
                    prompt=prompt,
                    variant=variant,
                    source_probability=source_probability,
                    action_probability=action_probability,
                    gt=gt,
                    layer_values=layer_values,
                    mask_threshold=args.mask_threshold,
                    geometry_checks=geometry_checks,
                    nonzero_delta_threshold=args.nonzero_delta_threshold,
                ))
            del action_image, action_probability_map, action_padded, action_preprocessed_shape, layer_values
        del image, label_map, source_probability_map, source_padded

    summary_rows, family_summary_rows, ranking_rows = aggregate_rows(
        rows,
        nonzero_delta_threshold=args.nonzero_delta_threshold,
        severe_performance_drop=args.severe_performance_drop,
        meaningful_negative_drop=args.meaningful_negative_drop,
        persistent_negative_rate=args.persistent_negative_rate,
    )
    per_sample_path = output_dir / "per_sample.csv"
    summary_path = output_dir / "action_strength_summary.csv"
    family_summary_path = output_dir / "action_family_summary.csv"
    ranking_path = output_dir / "action_ranking.csv"
    write_csv(per_sample_path, rows, list(rows[0].keys()))
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    write_csv(family_summary_path, family_summary_rows, list(family_summary_rows[0].keys()))
    write_csv(ranking_path, ranking_rows, list(ranking_rows[0].keys()))

    preferred = [row for row in ranking_rows if row["eligible_preferred_action"]]
    summary = {
        "experiment": "v0_mri_action_screening",
        "scope": {
            "frozen_voxtell": True,
            "world_predictor_trained": False,
            "sfda_run": False,
            "spatial_actions": [],
            "candidate_actions": sorted({variant.action_family for variant in variants}),
            "strengths": ["mild", "moderate"],
            "action_grid": [
                {
                    "action_family": variant.action_family,
                    "strength": variant.strength,
                    "direction": variant.direction,
                    "parameter_name": variant.parameter_name,
                    "parameter_value": variant.parameter_value,
                }
                for variant in variants
            ],
            "labels_unchanged": True,
            "source_vls_untouched": True,
            "geometry_checks": [
                "raw support and outside-support values identical",
                "preprocessed shape identical",
                "padded shape identical",
                "sliding-window topology identical",
            ],
        },
        "args": vars(args),
        "num_cases": len(cases),
        "cases": [case.case for case in cases],
        "num_per_sample_rows": len(rows),
        "num_action_strength_rows": len(summary_rows),
        "num_action_family_rows": len(family_summary_rows),
        "per_sample_csv": str(per_sample_path),
        "action_strength_summary_csv": str(summary_path),
        "action_family_summary_csv": str(family_summary_path),
        "action_ranking_csv": str(ranking_path),
        "smoke_only": args.smoke_only,
        "preferred_actions": preferred,
        "excluded_persistent_negative_actions": [
            row for row in ranking_rows if row["persistent_negative_action"]
        ],
        "excluded_insufficient_transition_actions": [
            row for row in ranking_rows if row["ranking_status"] == "excluded_insufficient_transition"
        ],
        "action_ranking": ranking_rows,
        "ranking_note": (
            "Ranking favors stable non-zero latent/input sensitivity, case-level performance heterogeneity, "
            "and avoidance of persistent meaningful negative performance changes. Negative rates use "
            f"meaningful_negative_drop={args.meaningful_negative_drop:.3f}. Inspect action_strength_summary.csv and "
            "per_sample.csv before selecting the next-stage action."
        ),
    }
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
