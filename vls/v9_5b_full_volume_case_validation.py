"""V9.5b full-volume, case-level validation for blur reliability.

This evaluation-only diagnostic uses the eight V9.3 test cases and validates
the fixed V9.3 test manifest without using its extracted patches as the final
statistical unit.  Source, real blur(sigma=1.5), and imagined blur from the
frozen V9.3 WP are reconstructed over the complete native sliding-window
volume.  Reliability is then evaluated on full-volume probability maps.

No model is trained, no VoxTell/V9.3 module is modified, and no SFDA path is
run.  GT is used only to diagnose source pseudo-label correctness and full-
volume reliability validity.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import resolve_device, visual_action
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, pad_label_like_image
from vls.v7_0d_protocol_sanity import set_seed
from vls.v8_3_real_aug_oracle_reliability import rank_metrics, tie_aware_percentile_rank
from vls.voxtell_states import VoxTellStateInterface
from vls.v9_3_hierarchical_residual_world_predictor import (
    LEVEL_NAMES,
    level_features,
    native_decoder_from_predicted_skips,
)
from vls.v9_4_hierarchical_wp_matched_diagnostic import (
    BLUR_SIGMA,
    build_text_delta,
    load_test_manifest,
    load_v93_world_predictor,
)


OUTPUT_DIR = Path("outputs/v9_5b_full_volume_case_validation")
METHODS = ("confidence", "real_blur", "imagined_blur")
RELIABILITY_SCOPES = ("overall", "foreground_union", "predicted_positive")
REGIONS = ("TP", "FP", "FN", "TN")

PER_CASE_FIELDS = (
    "case",
    "source_dice",
    "source_foreground_iou",
    "source_background_iou",
    "source_miou",
    "real_blur_dice",
    "real_blur_foreground_iou",
    "real_blur_background_iou",
    "real_blur_miou",
    "imagined_blur_dice",
    "imagined_blur_foreground_iou",
    "imagined_blur_background_iou",
    "imagined_blur_miou",
    "source_minus_real_blur_dice_drop",
    "source_minus_real_blur_miou_drop",
    "real_blur_probability_mse",
    "real_blur_probability_mae",
    "real_blur_mask_dice_vs_source",
    "real_blur_mask_iou_vs_source",
    "imagined_blur_probability_mse",
    "imagined_blur_probability_mae",
    "imagined_blur_mask_dice_vs_source",
    "imagined_blur_mask_iou_vs_source",
)

RELIABILITY_FIELDS = (
    "case",
    "method",
    "scope",
    "voxel_count",
    "pseudo_accuracy",
    "reliability_mean",
    "auroc",
    "auprc",
    "spearman",
)

REGION_FIELDS = (
    "case",
    "method",
    "scope",
    "region",
    "voxel_count",
    "reliability_mean",
)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V9.5b full-volume case-level blur validation")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/v9_3_hierarchical_residual_world_predictor/best_hierarchical_world_predictor.pt",
    )
    parser.add_argument(
        "--patch-manifest",
        default="outputs/v9_3_hierarchical_residual_world_predictor/test_patch_manifest.json",
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0, help="debug smoke-test limit; 0 uses all 8 test cases")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run one complete case without writing formal outputs",
    )
    return parser.parse_args()


def make_paths(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    tn = int(np.count_nonzero(~prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    fg_union = tp + fp + fn
    fg_iou = 1.0 if fg_union == 0 else tp / max(fg_union, 1)
    bg_union = tn + fp + fn
    bg_iou = 1.0 if bg_union == 0 else tn / max(bg_union, 1)
    dice = 1.0 if fg_union == 0 else 2.0 * tp / max(2 * tp + fp + fn, 1)
    return {
        "dice": float(dice),
        "foreground_iou": float(fg_iou),
        "background_iou": float(bg_iou),
        "miou": float((fg_iou + bg_iou) / 2.0),
    }


def prediction_disagreement(left: np.ndarray, right: np.ndarray, threshold: float) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    delta = left - right
    dice, iou = mask_agreement(left > threshold, right > threshold)
    return {
        "probability_mse": float(np.mean(delta.astype(np.float64) ** 2)),
        "probability_mae": float(np.mean(np.abs(delta), dtype=np.float64)),
        "mask_dice": dice,
        "mask_iou": iou,
    }


def mask_agreement(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    left_count = int(np.count_nonzero(left))
    right_count = int(np.count_nonzero(right))
    dice = 1.0 if left_count + right_count == 0 else 2.0 * intersection / max(left_count + right_count, 1)
    iou = 1.0 if union == 0 else intersection / max(union, 1)
    return float(dice), float(iou)


def _volume_setup(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    preprocessed: torch.Tensor,
) -> tuple[torch.Tensor, tuple, tuple, list[tuple], torch.Tensor]:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    predictor = interface.predictor
    padded, slicer_revert_padding = pad_nd_image(
        preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None,
    )
    slicers = predictor._internal_get_sliding_window_slicers(padded.shape[1:])
    gaussian = compute_gaussian(
        tuple(predictor.patch_size), sigma_scale=1.0 / 8,
        value_scaling_factor=10, device=torch.device("cpu"),
    )
    return padded, slicer_revert_padding, tuple(int(size) for size in image.shape[-3:]), slicers, gaussian


def restore_volume(
    results: torch.Tensor,
    slicer_revert_padding: tuple,
    original_shape: tuple[int, int, int],
    bbox: Any,
) -> np.ndarray:
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    cropped = results[(slice(None), *slicer_revert_padding[1:])].float().numpy()
    restored = np.zeros((1, *original_shape), dtype=np.float32)
    insert_crop_into_image(restored, cropped, bbox)
    return restored[0]


@torch.inference_mode()
def full_volume_source_and_imagined(
    interface: VoxTellStateInterface,
    world_predictor: torch.nn.Module,
    image: np.ndarray,
    prompt_embedding: torch.Tensor,
    text_delta: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictor = interface.predictor
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    padded, slicer_revert_padding, spatial_shape, slicers, gaussian = _volume_setup(
        interface, image, preprocessed,
    )
    source_results = torch.zeros((1, *padded.shape[1:]), dtype=torch.half, device="cpu")
    imagined_results = torch.zeros_like(source_results)
    counts = torch.zeros(padded.shape[1:], dtype=torch.half, device="cpu")
    for tile_slice in slicers:
        patch = torch.clone(padded[tile_slice][None], memory_format=torch.contiguous_format)
        source_context = interface.forward_with_audit_context(patch, prompt_embedding)
        source_logits = source_context["final_prediction"][:, :1][0].to("cpu")
        source_features = level_features(source_context["decoder_audit"]["skips"], LEVEL_NAMES)
        predicted_features = world_predictor(
            source_features,
            visual_action("blur", BLUR_SIGMA, device),
            text_delta,
        )
        imagined_logits = native_decoder_from_predicted_skips(
            interface, source_context, predicted_features, LEVEL_NAMES,
        )[0].to("cpu")
        source_results[tile_slice] += source_logits * gaussian
        imagined_results[tile_slice] += imagined_logits * gaussian
        counts[tile_slice[1:]] += gaussian
        del patch, source_context, source_logits, source_features, predicted_features, imagined_logits
    torch.div(source_results, counts, out=source_results)
    torch.div(imagined_results, counts, out=imagined_results)
    source_logits_volume = restore_volume(source_results, slicer_revert_padding, spatial_shape, bbox)
    imagined_logits_volume = restore_volume(imagined_results, slicer_revert_padding, spatial_shape, bbox)
    if tuple(source_logits_volume.shape) != tuple(imagined_logits_volume.shape):
        raise AssertionError("Source and imagined full-volume output shapes differ")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return source_logits_volume, imagined_logits_volume


@torch.inference_mode()
def full_volume_real_blur(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    prompt_embedding: torch.Tensor,
) -> np.ndarray:
    from vls.augmentations import gaussian_blur_augment

    predictor = interface.predictor
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    blurred = gaussian_blur_augment(preprocessed.numpy(), BLUR_SIGMA)
    padded, slicer_revert_padding, spatial_shape, slicers, gaussian = _volume_setup(
        interface, image, torch.from_numpy(blurred),
    )
    results = torch.zeros((prompt_embedding.shape[1], *padded.shape[1:]), dtype=torch.half, device="cpu")
    counts = torch.zeros(padded.shape[1:], dtype=torch.half, device="cpu")
    network_device = interface.device
    embedding = prompt_embedding.to(network_device)
    for tile_slice in slicers:
        patch = torch.clone(padded[tile_slice][None], memory_format=torch.contiguous_format).to(network_device)
        context = torch.autocast(network_device.type, enabled=True) if network_device.type == "cuda" else torch.no_grad()
        with context:
            prediction = interface.network(patch, embedding)[0].to("cpu")
        results[tile_slice] += prediction * gaussian
        counts[tile_slice[1:]] += gaussian
        del patch, prediction
    torch.div(results, counts, out=results)
    output = restore_volume(results, slicer_revert_padding, spatial_shape, bbox)
    if tuple(output.shape) != tuple(int(size) for size in original_shape):
        raise AssertionError(f"Real blur output shape differs from original shape: {output.shape} vs {original_shape}")
    if network_device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def reliability_maps(
    source_probability: np.ndarray,
    real_blur_probability: np.ndarray,
    imagined_blur_probability: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    source_probability = np.asarray(source_probability, dtype=np.float32)
    real_blur_probability = np.asarray(real_blur_probability, dtype=np.float32)
    imagined_blur_probability = np.asarray(imagined_blur_probability, dtype=np.float32)
    if source_probability.shape != real_blur_probability.shape or source_probability.shape != imagined_blur_probability.shape:
        raise AssertionError("Full-volume probability shapes differ")
    confidence = np.maximum(source_probability, 1.0 - source_probability)
    real_delta = np.abs(source_probability - real_blur_probability)
    imagined_delta = np.abs(source_probability - imagined_blur_probability)
    maps = {
        "confidence": tie_aware_percentile_rank(confidence).reshape(confidence.shape),
        "real_blur": 1.0 - tie_aware_percentile_rank(real_delta).reshape(real_delta.shape),
        "imagined_blur": 1.0 - tie_aware_percentile_rank(imagined_delta).reshape(imagined_delta.shape),
    }
    raw = {
        "confidence": {"probability_mse": None, "probability_mae": None},
        "real_blur": {
            "probability_mse": float(np.mean((source_probability - real_blur_probability).astype(np.float64) ** 2)),
            "probability_mae": float(np.mean(np.abs(source_probability - real_blur_probability), dtype=np.float64)),
        },
        "imagined_blur": {
            "probability_mse": float(np.mean((source_probability - imagined_blur_probability).astype(np.float64) ** 2)),
            "probability_mae": float(np.mean(np.abs(source_probability - imagined_blur_probability), dtype=np.float64)),
        },
    }
    return maps, raw


def correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) != len(y):
        raise AssertionError("Correlation inputs have different lengths")
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"n": len(x), "pearson": None, "spearman": None}
    result: dict[str, Any] = {"n": len(x), "pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": None}
    try:
        from scipy.stats import spearmanr

        value = spearmanr(x, y).statistic
        result["spearman"] = None if not np.isfinite(value) else float(value)
    except ImportError:
        pass
    return result


def region_weighted_mean(
    region_rows: list[dict[str, Any]],
    method: str,
    scope: str,
    regions: set[str],
) -> float | None:
    selected = [
        row for row in region_rows
        if row["method"] == method and row["scope"] == scope
        and row["region"] in regions and int(row["voxel_count"]) > 0
    ]
    total = sum(int(row["voxel_count"]) for row in selected)
    return None if total == 0 else float(
        sum(float(row["reliability_mean"]) * int(row["voxel_count"]) for row in selected) / total
    )


def aggregate_reliability(
    rows: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate_rows = []
    region_summary: dict[str, Any] = {}
    for method in METHODS:
        region_summary[method] = {}
        for scope in RELIABILITY_SCOPES:
            selected = [row for row in rows if row["method"] == method and row["scope"] == scope]
            aggregate_rows.append({
                "case": "__all__",
                "method": method,
                "scope": scope,
                "voxel_count": None,
                **{
                    field: None if not [row[field] for row in selected if row[field] is not None]
                    else float(np.mean([row[field] for row in selected if row[field] is not None], dtype=np.float64))
                    for field in ("pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman")
                },
            })
            region_summary[method][scope] = {
                region: region_weighted_mean(region_rows, method, scope, {region})
                for region in REGIONS
            }
    return aggregate_rows, region_summary


def bootstrap_mean(values: list[float], rng: np.random.Generator, replicates: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n_cases": 0, "replicates": replicates, "mean": None, "ci95_low": None, "ci95_high": None}
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[indices].mean(axis=1)
    return {
        "n_cases": int(array.size),
        "replicates": int(replicates),
        "mean": float(array.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def bootstrap_paired(
    left: list[float | None],
    right: list[float | None],
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    pairs = np.asarray(
        [(float(a), float(b)) for a, b in zip(left, right, strict=True) if a is not None and b is not None],
        dtype=np.float64,
    )
    if pairs.size == 0:
        return {"n_cases": 0, "replicates": replicates, "mean_difference_left_minus_right": None, "ci95_low": None, "ci95_high": None}
    differences = pairs[:, 0] - pairs[:, 1]
    result = bootstrap_mean(differences.tolist(), rng, replicates)
    return {
        "n_cases": result["n_cases"],
        "replicates": result["replicates"],
        "mean_difference_left_minus_right": result["mean"],
        "ci95_low": result["ci95_low"],
        "ci95_high": result["ci95_high"],
        "positive_case_count": int(np.count_nonzero(differences > 0)),
        "positive_case_fraction": float(np.mean(differences > 0)),
    }


def build_bootstrap(
    case_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap: dict[str, Any] = {
        "unit": "case",
        "case_count": len(case_rows),
        "replicates": replicates,
        "seed": seed,
        "segmentation_and_disagreement": {},
        "reliability_paired_differences": {},
    }
    for name in (
        "source_minus_real_blur_dice_drop",
        "source_minus_real_blur_miou_drop",
        "real_blur_probability_mae",
        "imagined_blur_probability_mae",
        "real_blur_probability_mse",
        "imagined_blur_probability_mse",
    ):
        bootstrap["segmentation_and_disagreement"][name] = bootstrap_mean(
            [float(row[name]) for row in case_rows], rng, replicates,
        )
    for scope in RELIABILITY_SCOPES:
        for metric in ("auroc", "auprc", "spearman"):
            for method, reference in (
                ("imagined_blur", "confidence"),
                ("imagined_blur", "real_blur"),
            ):
                left = [row[metric] for row in reliability_rows if row["scope"] == scope and row["method"] == method]
                right = [row[metric] for row in reliability_rows if row["scope"] == scope and row["method"] == reference]
                bootstrap["reliability_paired_differences"][f"{scope}:{method}_minus_{reference}:{metric}"] = bootstrap_paired(
                    left, right, rng, replicates,
                )
    return bootstrap


def build_summary(
    args: argparse.Namespace,
    cases: list[Any],
    checkpoint_metadata: dict[str, Any],
    case_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    region_summary: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    correlations = {}
    for scope_name, predicate in (
        ("overall", lambda row: True),
        ("foreground_union_cases", lambda row: True),
    ):
        selected = [row for row in case_rows if predicate(row)]
        correlations[scope_name] = {
            "real_blur_probability_mae_vs_dice_drop": correlation(
                [float(row["real_blur_probability_mae"]) for row in selected],
                [float(row["source_minus_real_blur_dice_drop"]) for row in selected],
            ),
            "imagined_blur_probability_mae_vs_dice_drop": correlation(
                [float(row["imagined_blur_probability_mae"]) for row in selected],
                [float(row["source_minus_real_blur_dice_drop"]) for row in selected],
            ),
            "real_blur_probability_mae_vs_miou_drop": correlation(
                [float(row["real_blur_probability_mae"]) for row in selected],
                [float(row["source_minus_real_blur_miou_drop"]) for row in selected],
            ),
            "imagined_blur_probability_mae_vs_miou_drop": correlation(
                [float(row["imagined_blur_probability_mae"]) for row in selected],
                [float(row["source_minus_real_blur_miou_drop"]) for row in selected],
            ),
        }

    reliability_comparison = {}
    direction = {}
    for scope in RELIABILITY_SCOPES:
        values = {method: next(row for row in aggregate_rows if row["method"] == method and row["scope"] == scope) for method in METHODS}
        reliability_comparison[scope] = {}
        direction[scope] = {}
        for method in METHODS:
            correct = region_weighted_mean(region_rows, method, scope, {"TP", "TN"})
            incorrect = region_weighted_mean(region_rows, method, scope, {"FP", "FN"})
            direction[scope][method] = {
                "correct_region_reliability_mean": correct,
                "incorrect_region_reliability_mean": incorrect,
                "correct_minus_incorrect": None if correct is None or incorrect is None else correct - incorrect,
                "auroc": values[method]["auroc"],
                "spearman": values[method]["spearman"],
                "direction_correct": (
                    correct is not None and incorrect is not None and correct > incorrect
                    and values[method]["auroc"] is not None and float(values[method]["auroc"]) >= 0.5
                    and values[method]["spearman"] is not None and float(values[method]["spearman"]) >= 0.0
                ),
            }
        for method, reference in (("imagined_blur", "confidence"), ("imagined_blur", "real_blur")):
            left = values[method]
            right = values[reference]
            reliability_comparison[scope][f"{method}_vs_{reference}"] = {
                "deltas": {metric: None if left[metric] is None or right[metric] is None else float(left[metric]) - float(right[metric]) for metric in ("auroc", "auprc", "spearman")},
                "better_on": [metric for metric in ("auroc", "auprc", "spearman") if left[metric] is not None and right[metric] is not None and float(left[metric]) > float(right[metric])],
            }
    return {
        "stage": "V9.5b full-volume case-level validation",
        "evaluation_only": True,
        "models_trained": False,
        "sfda_run": False,
        "voxtell_modified": False,
        "v93_wp_modified": False,
        "loss_modified": False,
        "segmentation_head_added": False,
        "test_case_count": len(cases),
        "test_cases": [case.case for case in cases],
        "final_statistical_unit": "complete restored native-space test case",
        "fixed_manifest": str(Path(args.patch_manifest)),
        "fixed_manifest_patch_count": 32,
        "action_protocol": {"blur_sigma": BLUR_SIGMA},
        "checkpoint": checkpoint_metadata,
        "prediction_paths": {
            "source": "native VoxTell sliding-window output from source image",
            "real_blur": "native VoxTell sliding-window output from preprocessing-space Gaussian blur sigma=1.5",
            "imagined_blur": "V9.3 correct blur action WP prediction per source tile, followed by unchanged native decoder and full-volume Gaussian aggregation",
        },
        "reliability": {
            "methods": list(METHODS),
            "scopes": {
                "overall": "all full-volume voxels",
                "foreground_union": "GT positive union source predicted positive",
                "predicted_positive": "source predicted positive only",
            },
            "higher_is_more_reliable": True,
            "gt_usage": "GT only labels source pseudo-label correctness and is not used to construct predictions or select windows.",
            "aggregation": "case macro for AUROC/AUPRC/Spearman; TP/FP/FN/TN means are voxel-count weighted.",
        },
        "case_level_correlations": correlations,
        "reliability_direction": direction,
        "imagined_blur_comparison": reliability_comparison,
        "region_reliability_means": region_summary,
        "bootstrap": bootstrap,
        "row_counts": {
            "per_case": len(case_rows),
            "reliability": len(reliability_rows),
            "reliability_regions": len(region_rows),
            "aggregate_reliability": len(aggregate_rows),
        },
        "outputs": {
            "per_case": str(OUTPUT_DIR / "per_case_metrics.csv"),
            "reliability": str(OUTPUT_DIR / "reliability_metrics.csv"),
            "reliability_regions": str(OUTPUT_DIR / "reliability_regions.csv"),
            "bootstrap": str(OUTPUT_DIR / "bootstrap.json"),
            "summary": str(OUTPUT_DIR / "summary.json"),
        },
    }


def run(args: argparse.Namespace) -> None:
    if BLUR_SIGMA != 1.5:
        raise AssertionError("V9.5b requires blur sigma=1.5")
    if args.case_limit < 0 or args.bootstrap_replicates <= 0:
        raise AssertionError("Invalid case limit or bootstrap replicate count")
    if args.prediction_threshold != 0.5:
        raise AssertionError("V9.3 prediction threshold must remain 0.5")
    set_seed(args.seed)
    paths = make_paths(args)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V9.5b requires CUDA, resolved {device}")
    cases = iter_cases(paths, split="test")
    if len(cases) != 8:
        raise AssertionError(f"V9.5b requires the original 8-case V9.3 test split, got {len(cases)}")
    if args.case_limit:
        cases = cases[: args.case_limit]
    # Validate the fixed V9.3 manifest, but deliberately do not use its patch
    # coordinates as the final statistical unit.
    load_test_manifest(Path(args.patch_manifest), cases, patch_limit=0)

    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    world_predictor, checkpoint_metadata = load_v93_world_predictor(Path(args.world_checkpoint), device)
    prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    text_delta = build_text_delta(interface, device)
    text_backbone = getattr(interface.predictor, "text_backbone", None)
    interface.predictor.text_backbone = None
    interface.predictor.tokenizer = None
    interface.predictor._text_embedding_cache.clear()
    del text_backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    case_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        print(f"[V9.5b] case {case_index}/{len(cases)} start {case.case}", flush=True)
        image, label, _ = read_image_and_label(case)
        source_logits, imagined_logits = full_volume_source_and_imagined(
            interface, world_predictor, image, prompt_embedding, text_delta, device,
        )
        real_blur_logits = full_volume_real_blur(interface, image, prompt_embedding)
        source_probability = 1.0 / (1.0 + np.exp(-source_logits))
        real_blur_probability = 1.0 / (1.0 + np.exp(-real_blur_logits))
        imagined_blur_probability = 1.0 / (1.0 + np.exp(-imagined_logits))
        gt = np.asarray(label == args.label_value, dtype=bool)
        if source_probability.shape != gt.shape or real_blur_probability.shape != gt.shape or imagined_blur_probability.shape != gt.shape:
            raise AssertionError(f"Full-volume shape mismatch for {case.case}: source={source_probability.shape}, gt={gt.shape}")
        source_mask = source_probability > args.prediction_threshold
        real_blur_mask = real_blur_probability > args.prediction_threshold
        imagined_blur_mask = imagined_blur_probability > args.prediction_threshold
        source_metrics = binary_metrics(source_mask, gt)
        real_metrics = binary_metrics(real_blur_mask, gt)
        imagined_metrics = binary_metrics(imagined_blur_mask, gt)
        real_disagreement = prediction_disagreement(source_probability, real_blur_probability, args.prediction_threshold)
        imagined_disagreement = prediction_disagreement(source_probability, imagined_blur_probability, args.prediction_threshold)
        case_row = {
            "case": case.case,
            **{f"source_{key}": value for key, value in source_metrics.items()},
            **{f"real_blur_{key}": value for key, value in real_metrics.items()},
            **{f"imagined_blur_{key}": value for key, value in imagined_metrics.items()},
            "source_minus_real_blur_dice_drop": source_metrics["dice"] - real_metrics["dice"],
            "source_minus_real_blur_miou_drop": source_metrics["miou"] - real_metrics["miou"],
            "real_blur_probability_mse": real_disagreement["probability_mse"],
            "real_blur_probability_mae": real_disagreement["probability_mae"],
            "real_blur_mask_dice_vs_source": real_disagreement["mask_dice"],
            "real_blur_mask_iou_vs_source": real_disagreement["mask_iou"],
            "imagined_blur_probability_mse": imagined_disagreement["probability_mse"],
            "imagined_blur_probability_mae": imagined_disagreement["probability_mae"],
            "imagined_blur_mask_dice_vs_source": imagined_disagreement["mask_dice"],
            "imagined_blur_mask_iou_vs_source": imagined_disagreement["mask_iou"],
        }
        case_rows.append(case_row)

        maps, _ = reliability_maps(source_probability, real_blur_probability, imagined_blur_probability)
        source_pseudo = source_mask
        correct = source_pseudo == gt
        flat_gt = gt.reshape(-1)
        flat_pseudo = source_pseudo.reshape(-1)
        flat_correct = correct.reshape(-1)
        scope_masks = {
            "overall": np.ones(flat_gt.size, dtype=bool),
            "foreground_union": flat_gt | flat_pseudo,
            "predicted_positive": flat_pseudo,
        }
        region_masks = {
            "TP": flat_pseudo & flat_gt,
            "FP": flat_pseudo & ~flat_gt,
            "FN": ~flat_pseudo & flat_gt,
            "TN": ~flat_pseudo & ~flat_gt,
        }
        for method in METHODS:
            flat_reliability = maps[method].reshape(-1)
            for scope, scope_mask in scope_masks.items():
                metrics = rank_metrics(flat_reliability, flat_correct, scope_mask)
                reliability_rows.append({
                    "case": case.case,
                    "method": method,
                    "scope": scope,
                    "voxel_count": int(scope_mask.sum()),
                    **metrics,
                })
                for region, region_mask in region_masks.items():
                    selected = flat_reliability[region_mask & scope_mask]
                    region_rows.append({
                        "case": case.case,
                        "method": method,
                        "scope": scope,
                        "region": region,
                        "voxel_count": int(selected.size),
                        "reliability_mean": None if selected.size == 0 else float(np.mean(selected, dtype=np.float64)),
                    })
        print(f"[V9.5b] case {case_index}/{len(cases)} complete", flush=True)
        del image, label, source_logits, real_blur_logits, imagined_logits
        del source_probability, real_blur_probability, imagined_blur_probability, gt, maps
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.smoke_only:
        aggregate_rows, region_summary = aggregate_reliability(reliability_rows, region_rows)
        print(json.dumps({
            "smoke": "passed",
            "test_cases_evaluated": len(cases),
            "full_volume_case_rows": len(case_rows),
            "reliability_rows": len(reliability_rows),
            "region_rows": len(region_rows),
            "first_case": case_rows[0],
            "overall_reliability": [row for row in aggregate_rows if row["scope"] == "overall"],
            "training_run": False,
            "sfda_run": False,
        }, indent=2), flush=True)
        return

    aggregate_rows, region_summary = aggregate_reliability(reliability_rows, region_rows)
    bootstrap = build_bootstrap(case_rows, reliability_rows, args.seed, args.bootstrap_replicates)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "per_case_metrics.csv", PER_CASE_FIELDS, case_rows)
    write_csv(OUTPUT_DIR / "reliability_metrics.csv", RELIABILITY_FIELDS, reliability_rows + aggregate_rows)
    write_csv(OUTPUT_DIR / "reliability_regions.csv", REGION_FIELDS, region_rows)
    (OUTPUT_DIR / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2))
    summary = build_summary(
        args, cases, checkpoint_metadata, case_rows, reliability_rows, region_rows,
        aggregate_rows, region_summary, bootstrap,
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"complete": True, "outputs": summary["outputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
