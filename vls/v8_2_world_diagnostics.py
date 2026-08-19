"""V8.2 streaming mechanism diagnosis for the frozen V8.0 World Predictor.

This module is diagnostic-only.  It performs no training, checkpoint
selection, or hyperparameter tuning.  It keeps the V8.0 action protocol and
the V7 reliability definitions unchanged while processing one patch/action
at a time and writing scalar results incrementally.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import (
    gamma_augment,
    gaussian_blur_augment,
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
from vls.v7_0d_protocol_sanity import (
    STRONG_ACTIONS,
    binary_metrics,
    reliability_from_source,
    set_seed,
)
from vls.voxtell_states import VoxTellStateInterface


ACTION_PROTOCOL = (("gamma", 0.30), ("blur", 1.5))
MATCHED_WRONG_ACTIONS = {
    "gamma": ("blur", 1.5),
    "blur": ("gamma", 0.30),
}
RELIABILITY_SOURCES = {
    "confidence_rank": "confidence_rank",
    "world_pairwise": "world_stability",
    "joint_product": "joint_product",
}
SUMMARY_NUMERIC_NAMES = (
    "true_shift_norm", "conditioned_shift_norm", "same_model_no_action_shift_norm", "matched_wrong_shift_norm",
    "conditioned_vs_same_model_no_action_distance", "gamma_vs_blur_conditioned_distance",
    "conditioned_state_mse", "same_model_no_action_state_mse", "matched_wrong_state_mse",
    "correct_vs_same_model_no_action_margin", "correct_vs_matched_wrong_margin",
    "conditioned_delta_cosine", "same_model_no_action_delta_cosine", "matched_wrong_delta_cosine",
    "magnitude_ratio_conditioned", "repeat_forward_distance",
    "source_dice", "actual_dice", "conditioned_dice", "same_model_no_action_dice", "matched_wrong_dice",
    "source_precision", "actual_precision", "conditioned_precision", "same_model_no_action_precision", "matched_wrong_precision",
    "source_recall", "actual_recall", "conditioned_recall", "same_model_no_action_recall", "matched_wrong_recall",
    "source_iou", "actual_iou", "conditioned_iou", "same_model_no_action_iou", "matched_wrong_iou",
    "conditioned_vs_actual_seg_logits_mse", "same_model_no_action_vs_actual_seg_logits_mse",
    "matched_wrong_vs_actual_seg_logits_mse", "seg_gain_vs_same_model_no_action", "seg_gain_vs_matched_wrong",
)
FEATURE_FIELDS = (
    "case", "patch_index", "patch_kind", "action", "action_strength", "wrong_type_action",
    "wrong_type_strength", "true_shift_norm", "conditioned_shift_norm", "same_model_no_action_shift_norm", "matched_wrong_shift_norm",
    "conditioned_vs_same_model_no_action_distance", "gamma_vs_blur_conditioned_distance",
    "conditioned_state_mse", "same_model_no_action_state_mse", "matched_wrong_state_mse",
    "correct_vs_same_model_no_action_margin", "correct_vs_matched_wrong_margin",
    "conditioned_delta_cosine", "same_model_no_action_delta_cosine", "matched_wrong_delta_cosine",
    "conditioned_delta_cosine_valid", "same_model_no_action_delta_cosine_valid", "matched_wrong_delta_cosine_valid",
    "magnitude_ratio_conditioned", "magnitude_ratio_conditioned_valid", "repeat_forward_distance",
)
SEGMENTATION_FIELDS = (
    "segmentation_level", "case", "patch_index", "patch_kind", "action", "action_strength",
    "source_dice", "actual_dice", "conditioned_dice", "same_model_no_action_dice", "matched_wrong_dice",
    "source_precision", "actual_precision", "conditioned_precision", "same_model_no_action_precision", "matched_wrong_precision",
    "source_recall", "actual_recall", "conditioned_recall", "same_model_no_action_recall", "matched_wrong_recall",
    "source_iou", "actual_iou", "conditioned_iou", "same_model_no_action_iou", "matched_wrong_iou",
    "conditioned_vs_actual_seg_logits_mse", "same_model_no_action_vs_actual_seg_logits_mse",
    "matched_wrong_vs_actual_seg_logits_mse", "seg_gain_vs_same_model_no_action", "seg_gain_vs_matched_wrong",
)
CASE_SUMMARY_FIELDS = (
    "summary_level", "case", "action", "row_count",
    *(f"mean_{name}" for name in SUMMARY_NUMERIC_NAMES),
)
RELIABILITY_METRIC_FIELDS = (
    "source", "action", "case", "scope", "patch_count", "valid_auroc_count",
    "pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman",
)
RELIABILITY_VOXEL_FIELDS = (
    "source", "action", "case", "region", "count", "mean", "median", "p10", "p50", "p90",
)
RELIABILITY_DECILE_FIELDS = (
    "source", "action", "case", "scope", "decile", "voxel_count",
    "reliability_mean", "pseudo_label_accuracy", "aggregation",
)
FULL_VOLUME_FIELDS = (
    "case", "action", "strength", "dice", "iou", "precision", "recall",
    "aggregation",
)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V8.2 streaming World Predictor mechanism diagnosis")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v8_0_full_world_predictor/best_world_predictor.pt")
    parser.add_argument("--output-dir", default="outputs/v8_2_fix_world_diagnostics")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def validate_protocol(args: argparse.Namespace) -> None:
    if tuple(STRONG_ACTIONS) != ACTION_PROTOCOL:
        raise AssertionError("V8.2 action protocol differs from the frozen V8 protocol")
    if MATCHED_WRONG_ACTIONS != {"gamma": ("blur", 1.5), "blur": ("gamma", 0.30)}:
        raise AssertionError("V8.2 matched-wrong action protocol must be gamma .30 <-> blur 1.5")
    if set(RELIABILITY_SOURCES.values()) != {"confidence_rank", "world_stability", "joint_product"}:
        raise AssertionError("V8.2 reliability definitions changed")
    if args.patches_per_case <= 0:
        raise AssertionError("patches_per_case must be positive")


class CsvSink:
    def __init__(self, path: Path, fields: tuple[str, ...]) -> None:
        self.handle = path.open("w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(fields), extrasaction="ignore")
        self.writer.writeheader()
        self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: Any) -> None:
        if value is None:
            return
        value = float(value)
        if np.isfinite(value):
            self.total += value
            self.count += 1

    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class MetricAccumulator:
    patch_count: int = 0
    metrics: dict[str, RunningMean] = field(default_factory=dict)

    def update(self, row: dict[str, Any]) -> None:
        self.patch_count += 1
        for name in ("pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman"):
            self.metrics.setdefault(name, RunningMean()).update(row.get(name))

    def row(self, source: str, action: str, case: str, scope: str) -> dict[str, Any]:
        return {
            "source": source,
            "action": action,
            "case": case,
            "scope": scope,
            "patch_count": self.patch_count,
            "valid_auroc_count": self.metrics.get("auroc", RunningMean()).count,
            **{name: self.metrics.get(name, RunningMean()).mean() for name in (
                "pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman",
            )},
        }


class Histogram:
    """Fixed 100-bin [0,1] histogram for streaming approximate quantiles."""

    def __init__(self) -> None:
        self.counts = np.zeros(100, dtype=np.int64)
        self.total = 0
        self.total_sum = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size:
            return
        clipped = np.clip(values, 0.0, 1.0)
        bins = np.minimum((clipped * 100.0).astype(np.int64), 99)
        self.counts += np.bincount(bins, minlength=100)
        self.total += int(values.size)
        self.total_sum += float(clipped.sum())

    def quantile(self, fraction: float) -> float | None:
        if self.total == 0:
            return None
        target = max(1, int(np.ceil(self.total * fraction)))
        index = int(np.searchsorted(np.cumsum(self.counts), target, side="left"))
        return (index + 0.5) / 100.0

    def row(self, source: str, action: str, case: str, region: str) -> dict[str, Any]:
        return {
            "source": source,
            "action": action,
            "case": case,
            "region": region,
            "count": self.total,
            "mean": self.total_sum / self.total if self.total else None,
            "median": self.quantile(0.5),
            "p10": self.quantile(0.1),
            "p50": self.quantile(0.5),
            "p90": self.quantile(0.9),
        }


class DecileAccumulator:
    def __init__(self) -> None:
        self.count = np.zeros(10, dtype=np.int64)
        self.correct = np.zeros(10, dtype=np.int64)
        self.reliability_sum = np.zeros(10, dtype=np.float64)

    def update(self, reliability: np.ndarray, correct: np.ndarray, mask: np.ndarray) -> None:
        reliability = np.asarray(reliability, dtype=np.float64).reshape(-1)
        correct = np.asarray(correct, dtype=bool).reshape(-1)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if not mask.any():
            return
        values = np.clip(reliability[mask], 0.0, 1.0)
        bins = np.minimum((values * 10.0).astype(np.int64), 9)
        self.count += np.bincount(bins, minlength=10)
        self.correct += np.bincount(bins, weights=correct[mask].astype(np.int64), minlength=10).astype(np.int64)
        self.reliability_sum += np.bincount(bins, weights=values, minlength=10)

    def rows(self, source: str, action: str, case: str, scope: str, aggregation: str) -> list[dict[str, Any]]:
        rows = []
        for index in range(10):
            if self.count[index] == 0:
                continue
            rows.append({
                "source": source,
                "action": action,
                "case": case,
                "scope": scope,
                "decile": index + 1,
                "voxel_count": int(self.count[index]),
                "reliability_mean": float(self.reliability_sum[index] / self.count[index]),
                "pseudo_label_accuracy": float(self.correct[index] / self.count[index]),
                "aggregation": aggregation,
            })
        return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if not n_positive or not n_negative:
        return None
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    positive_count = int((labels == 1).sum())
    if not positive_count:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64).reshape(-1), kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels == 1)
    precision = cumulative / np.arange(1, labels.size + 1)
    return float(precision[sorted_labels == 1].sum() / positive_count)


def spearman(scores: np.ndarray, labels: np.ndarray) -> float | None:
    score_rank = rankdata(scores)
    label_rank = rankdata(labels)
    score_rank -= score_rank.mean()
    label_rank -= label_rank.mean()
    denominator = float(np.linalg.norm(score_rank) * np.linalg.norm(label_rank))
    return None if denominator == 0.0 else float(np.dot(score_rank, label_rank) / denominator)


def rms_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt((left.float() - right.float()).pow(2).mean()).detach().cpu())


def state_mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).pow(2).mean().detach().cpu())


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> tuple[float | None, bool]:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    valid = float(left_norm.detach().cpu()) > 0.0 and float(right_norm.detach().cpu()) > 0.0
    if not valid:
        return None, False
    return float(torch.dot(left, right).div(left_norm * right_norm).detach().cpu()), True


def magnitude_ratio(delta: torch.Tensor, true_delta: torch.Tensor) -> tuple[float | None, bool]:
    denominator = float(torch.linalg.vector_norm(true_delta.float()).detach().cpu())
    if denominator <= 0.0:
        return None, False
    return float(torch.linalg.vector_norm(delta.float()).detach().cpu()) / denominator, True


def segmentation_metrics(logits: torch.Tensor, gt: torch.Tensor, threshold: float) -> dict[str, float]:
    if tuple(logits.shape[-3:]) != tuple(gt.shape[-3:]):
        gt = F.interpolate(gt.float(), size=logits.shape[-3:], mode="nearest") > 0.5
    prediction = (torch.sigmoid(logits) > threshold).detach().cpu().numpy().astype(bool).reshape(-1)
    target = gt.detach().cpu().numpy().astype(bool).reshape(-1)
    metrics = binary_metrics(prediction, target)
    tp, fp, fn = int(metrics["tp"]), int(metrics["fp"]), int(metrics["fn"])
    return {
        "dice": float(metrics["dice"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "iou": 1.0 if tp + fp + fn == 0 else tp / max(tp + fp + fn, 1),
    }


@torch.inference_mode()
def full_volume_action_metrics(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    label: np.ndarray,
    prompt_embedding: torch.Tensor,
    action: str,
    strength: float,
    label_value: int,
    threshold: float,
) -> dict[str, float]:
    """Evaluate only VoxTell's real final full-volume segmentation.

    This deliberately does not feed World Predictor states into the full
    decoder.  The V8.0 checkpoint predicts one selected decoder state, while
    full-volume final prediction requires the complete sliding-window encoder
    and decoder path.  Keeping this check independent avoids inventing an
    imagined full-volume result.
    """
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    predictor = interface.predictor
    if action == "source":
        transformed = image
        preprocessed, bbox, original_shape = predictor.preprocess(transformed)
    elif action == "gamma":
        transformed = gamma_augment(image, 1.0 + strength)
        preprocessed, bbox, original_shape = predictor.preprocess(transformed)
    elif action == "blur":
        transformed = None
        preprocessed, bbox, original_shape = predictor.preprocess(image)
        blurred = gaussian_blur_augment(preprocessed.numpy(), strength)
        preprocessed = torch.from_numpy(blurred)
    else:
        raise ValueError(f"Unsupported full-volume action: {action}")

    cropped_logits = predictor.predict_sliding_window_return_logits(
        preprocessed, prompt_embedding,
    ).detach().float().cpu().numpy()
    full_logits = np.zeros((cropped_logits.shape[0], *original_shape), dtype=np.float32)
    full_logits = insert_crop_into_image(full_logits, cropped_logits, bbox)
    prediction = (1.0 / (1.0 + np.exp(-full_logits[0])) > threshold).reshape(-1)
    target = (label == label_value).reshape(-1).astype(bool)
    metrics = binary_metrics(prediction, target)
    tp, fp, fn = int(metrics["tp"]), int(metrics["fp"]), int(metrics["fn"])
    result = {
        "dice": float(metrics["dice"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "iou": 1.0 if tp + fp + fn == 0 else tp / max(tp + fp + fn, 1),
    }
    del cropped_logits, full_logits, preprocessed
    if transformed is not image:
        del transformed
    gc.collect()
    if interface.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def patch_reliability_metrics(
    reliability: np.ndarray,
    correct: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    scores = np.asarray(reliability).reshape(-1)[mask.reshape(-1)]
    labels = np.asarray(correct).reshape(-1)[mask.reshape(-1)].astype(np.int8)
    return {
        "pseudo_accuracy": float(labels.mean()) if labels.size else None,
        "reliability_mean": float(scores.mean()) if scores.size else None,
        "auroc": auroc(scores, labels) if scores.size else None,
        "auprc": auprc(scores, labels) if scores.size else None,
        "spearman": spearman(scores, labels) if scores.size else None,
    }


def memory_status(device: torch.device) -> dict[str, Any]:
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    result: dict[str, Any] = {"cpu_rss_mb": round(float(rss_mb), 1)}
    if device.type == "cuda" and torch.cuda.is_available():
        result.update({
            "gpu_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
        })
    else:
        result.update({"gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0})
    return result


def write_progress(
    path: Path,
    completed_cases: list[str],
    rows: int,
    device: torch.device,
    stage: str,
) -> None:
    payload = {
        "stage": stage,
        "completed_cases": completed_cases,
        "completed_case_count": len(completed_cases),
        "completed_rows": rows,
        "memory": memory_status(device),
    }
    path.write_text(json.dumps(payload, indent=2))


def aggregate_row(rows: list[dict[str, Any]], level: str, case: str, action: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "summary_level": level,
        "case": case,
        "action": action,
        "row_count": len(rows),
    }
    for name in SUMMARY_NUMERIC_NAMES:
        values = [float(row[name]) for row in rows if row.get(name) is not None and np.isfinite(float(row[name]))]
        output[f"mean_{name}"] = float(np.mean(values)) if values else None
    return output


def update_summary_macro(accumulator: dict[str, dict[str, RunningMean]], key: str, row: dict[str, Any]) -> None:
    for name in SUMMARY_NUMERIC_NAMES:
        accumulator.setdefault(key, {}).setdefault(name, RunningMean()).update(row.get(f"mean_{name}"))


def summary_macro_row(accumulator: dict[str, dict[str, RunningMean]], key: str, action: str) -> dict[str, Any]:
    return {
        "summary_level": "overall_case_macro",
        "case": "__all__",
        "action": action,
        "row_count": accumulator.get(key, {}).get("mean_true_shift_norm", RunningMean()).count,
        **{f"mean_{name}": accumulator.get(key, {}).get(name, RunningMean()).mean() for name in SUMMARY_NUMERIC_NAMES},
    }


@torch.inference_mode()
def evaluate_action(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    source_state: torch.Tensor,
    true_patch: torch.Tensor,
    action: str,
    strength: float,
    matched_wrong_action: str,
    matched_wrong_strength: float,
    source_segmentation: dict[str, float],
    gt_patch: torch.Tensor,
    selected_stage: str,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    true_result = interface.forward_with_states(true_patch, interface._v8_prompt_embedding)
    true_state = true_result["decoder_states"][selected_stage][:, 0].detach().float().to(device)
    action_tensor = visual_action(action, strength, device)
    conditioned_state = world_model(source_state, action=action_tensor)
    repeated_state = world_model(source_state, action=action_tensor)
    repeat_distance = rms_distance(conditioned_state, repeated_state)
    same_model_no_action_state = world_model(source_state, action=None)
    matched_wrong_state = world_model(
        source_state,
        action=visual_action(matched_wrong_action, matched_wrong_strength, device),
    )

    true_delta = true_state - source_state
    conditioned_delta = conditioned_state - source_state
    same_model_no_action_delta = same_model_no_action_state - source_state
    matched_wrong_delta = matched_wrong_state - source_state
    conditioned_cosine, conditioned_cosine_valid = cosine_similarity(true_delta, conditioned_delta)
    same_model_no_action_cosine, same_model_no_action_cosine_valid = cosine_similarity(
        true_delta, same_model_no_action_delta,
    )
    matched_wrong_cosine, matched_wrong_cosine_valid = cosine_similarity(true_delta, matched_wrong_delta)
    ratio, ratio_valid = magnitude_ratio(conditioned_delta, true_delta)
    conditioned_mse = state_mse(conditioned_state, true_state)
    same_model_no_action_mse = state_mse(same_model_no_action_state, true_state)
    matched_wrong_mse = state_mse(matched_wrong_state, true_state)
    feature = {
        "action": action,
        "action_strength": strength,
        "wrong_type_action": matched_wrong_action,
        "wrong_type_strength": matched_wrong_strength,
        "true_shift_norm": float(torch.linalg.vector_norm(true_delta).detach().cpu()),
        "conditioned_shift_norm": float(torch.linalg.vector_norm(conditioned_delta).detach().cpu()),
        "same_model_no_action_shift_norm": float(torch.linalg.vector_norm(same_model_no_action_delta).detach().cpu()),
        "matched_wrong_shift_norm": float(torch.linalg.vector_norm(matched_wrong_delta).detach().cpu()),
        "conditioned_vs_same_model_no_action_distance": rms_distance(conditioned_state, same_model_no_action_state),
        "conditioned_state_mse": conditioned_mse,
        "same_model_no_action_state_mse": same_model_no_action_mse,
        "matched_wrong_state_mse": matched_wrong_mse,
        "correct_vs_same_model_no_action_margin": same_model_no_action_mse - conditioned_mse,
        "correct_vs_matched_wrong_margin": matched_wrong_mse - conditioned_mse,
        "conditioned_delta_cosine": conditioned_cosine,
        "same_model_no_action_delta_cosine": same_model_no_action_cosine,
        "matched_wrong_delta_cosine": matched_wrong_cosine,
        "conditioned_delta_cosine_valid": conditioned_cosine_valid,
        "same_model_no_action_delta_cosine_valid": same_model_no_action_cosine_valid,
        "matched_wrong_delta_cosine_valid": matched_wrong_cosine_valid,
        "magnitude_ratio_conditioned": ratio,
        "magnitude_ratio_conditioned_valid": ratio_valid,
        "repeat_forward_distance": repeat_distance,
    }

    actual_logits = state_to_intermediate_prediction(interface, selected_stage, true_state)
    conditioned_logits = state_to_intermediate_prediction(interface, selected_stage, conditioned_state)
    actual_segmentation = segmentation_metrics(actual_logits, gt_patch, threshold)
    conditioned_segmentation = segmentation_metrics(conditioned_logits, gt_patch, threshold)
    conditioned_vs_actual_logits_mse = state_mse(conditioned_logits, actual_logits)
    del conditioned_logits

    same_model_no_action_logits = state_to_intermediate_prediction(
        interface, selected_stage, same_model_no_action_state,
    )
    same_model_no_action_segmentation = segmentation_metrics(
        same_model_no_action_logits, gt_patch, threshold,
    )
    same_model_no_action_vs_actual_logits_mse = state_mse(same_model_no_action_logits, actual_logits)
    del same_model_no_action_logits

    matched_wrong_logits = state_to_intermediate_prediction(
        interface, selected_stage, matched_wrong_state,
    )
    matched_wrong_segmentation = segmentation_metrics(matched_wrong_logits, gt_patch, threshold)
    matched_wrong_vs_actual_logits_mse = state_mse(matched_wrong_logits, actual_logits)
    del actual_logits, matched_wrong_logits

    segmentation = {
        "segmentation_level": "intermediate",
        "source_dice": source_segmentation["dice"],
        "source_precision": source_segmentation["precision"],
        "source_recall": source_segmentation["recall"],
        "source_iou": source_segmentation["iou"],
        "actual_dice": actual_segmentation["dice"],
        "actual_precision": actual_segmentation["precision"],
        "actual_recall": actual_segmentation["recall"],
        "actual_iou": actual_segmentation["iou"],
        "conditioned_dice": conditioned_segmentation["dice"],
        "conditioned_precision": conditioned_segmentation["precision"],
        "conditioned_recall": conditioned_segmentation["recall"],
        "conditioned_iou": conditioned_segmentation["iou"],
        "same_model_no_action_dice": same_model_no_action_segmentation["dice"],
        "same_model_no_action_precision": same_model_no_action_segmentation["precision"],
        "same_model_no_action_recall": same_model_no_action_segmentation["recall"],
        "same_model_no_action_iou": same_model_no_action_segmentation["iou"],
        "matched_wrong_dice": matched_wrong_segmentation["dice"],
        "matched_wrong_precision": matched_wrong_segmentation["precision"],
        "matched_wrong_recall": matched_wrong_segmentation["recall"],
        "matched_wrong_iou": matched_wrong_segmentation["iou"],
        "conditioned_vs_actual_seg_logits_mse": conditioned_vs_actual_logits_mse,
        "same_model_no_action_vs_actual_seg_logits_mse": same_model_no_action_vs_actual_logits_mse,
        "matched_wrong_vs_actual_seg_logits_mse": matched_wrong_vs_actual_logits_mse,
        "seg_gain_vs_same_model_no_action": conditioned_segmentation["dice"] - same_model_no_action_segmentation["dice"],
        "seg_gain_vs_matched_wrong": conditioned_segmentation["dice"] - matched_wrong_segmentation["dice"],
    }
    result = {**feature, **segmentation}
    del true_result, true_state, true_delta, conditioned_delta
    del same_model_no_action_delta, matched_wrong_delta
    del same_model_no_action_state, matched_wrong_state, repeated_state, action_tensor
    return result, conditioned_state


def update_reliability(
    source: str,
    action: str,
    case: str,
    maps: dict[str, np.ndarray],
    correct: np.ndarray,
    gt: np.ndarray,
    pseudo: np.ndarray,
    metric_acc: dict[tuple[str, str, str], MetricAccumulator],
    region_acc: dict[tuple[str, str, str], Histogram],
    decile_acc: dict[tuple[str, str, str], DecileAccumulator],
) -> None:
    reliability = np.asarray(maps[RELIABILITY_SOURCES[source]], dtype=np.float32)
    flat_gt = gt.reshape(-1).astype(bool)
    flat_correct = correct.reshape(-1).astype(bool)
    flat_pseudo = pseudo.reshape(-1).astype(bool)
    flat_reliability = reliability.reshape(-1)
    masks = {
        "overall": np.ones(flat_gt.size, dtype=bool),
        "foreground": flat_gt,
        "background": ~flat_gt,
    }
    for scope, mask in masks.items():
        metrics = patch_reliability_metrics(flat_reliability, flat_correct, flat_gt, mask)
        metric_acc.setdefault((source, action, scope), MetricAccumulator()).update(metrics)
        decile_acc.setdefault((source, action, scope), DecileAccumulator()).update(
            flat_reliability, flat_correct, mask,
        )
    regions = {
        "TP": flat_pseudo & flat_gt,
        "FP": flat_pseudo & ~flat_gt,
        "TN": ~flat_pseudo & ~flat_gt,
        "FN": ~flat_pseudo & flat_gt,
    }
    for region, mask in regions.items():
        region_acc.setdefault((source, action, region), Histogram()).update(flat_reliability[mask])


def reliability_case_rows(
    case: str,
    metric_acc: dict[tuple[str, str, str], MetricAccumulator],
    region_acc: dict[tuple[str, str, str], Histogram],
    decile_acc: dict[tuple[str, str, str], DecileAccumulator],
    metric_sink: CsvSink,
    region_sink: CsvSink,
    decile_sink: CsvSink,
    global_metric: dict[tuple[str, str, str], MetricAccumulator],
    global_region: dict[tuple[str, str, str], dict[str, RunningMean]],
    global_decile: dict[tuple[str, str, str], list[RunningMean]],
) -> None:
    def add_global_metric(row: dict[str, Any]) -> None:
        key = (row["source"], row["action"], row["scope"])
        target = global_metric.setdefault(key, MetricAccumulator())
        target.patch_count += 1
        for name in ("pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman"):
            target.metrics.setdefault(name, RunningMean()).update(row.get(name))

    def combined_metric_row(source: str, case_action: str, scope: str) -> dict[str, Any] | None:
        rows = []
        for action_name in ("gamma", "blur"):
            accumulator = metric_acc.get((source, action_name, scope))
            if accumulator is not None:
                rows.append(accumulator.row(source, action_name, case, scope))
        if not rows:
            return None
        return {
            "source": source,
            "action": case_action,
            "case": case,
            "scope": scope,
            "patch_count": sum(int(row["patch_count"]) for row in rows),
            "valid_auroc_count": sum(int(row["valid_auroc_count"]) for row in rows),
            **{
                name: float(np.mean([row[name] for row in rows if row.get(name) is not None]))
                if any(row.get(name) is not None for row in rows) else None
                for name in ("pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman")
            },
        }

    def combined_histogram(source: str, region: str) -> Histogram | None:
        histograms = [
            region_acc[(source, action_name, region)]
            for action_name in ("gamma", "blur")
            if (source, action_name, region) in region_acc
        ]
        if not histograms:
            return None
        combined = Histogram()
        for histogram in histograms:
            combined.counts += histogram.counts
            combined.total += histogram.total
            combined.total_sum += histogram.total_sum
        return combined

    def add_global_region(row: dict[str, Any]) -> None:
        key = (row["source"], row["action"], row["region"])
        target = global_region.setdefault(
            key, {name: RunningMean() for name in ("mean", "median", "p10", "p50", "p90")},
        )
        for name in ("mean", "median", "p10", "p50", "p90"):
            target[name].update(row.get(name))

    for (source, action, scope), accumulator in sorted(metric_acc.items()):
        row = accumulator.row(source, action, case, scope)
        metric_sink.write(row)
        add_global_metric(row)

    for source in RELIABILITY_SOURCES:
        for scope in ("overall", "foreground", "background"):
            row = combined_metric_row(source, "all", scope)
            if row is not None:
                metric_sink.write(row)
                add_global_metric(row)

    for (source, action, region), histogram in sorted(region_acc.items()):
        row = histogram.row(source, action, case, region)
        region_sink.write(row)
        add_global_region(row)

    for source in RELIABILITY_SOURCES:
        for region in ("TP", "FP", "TN", "FN"):
            histogram = combined_histogram(source, region)
            if histogram is not None:
                row = histogram.row(source, "all", case, region)
                region_sink.write(row)
                add_global_region(row)

    for (source, action, scope), accumulator in sorted(decile_acc.items()):
        rows = accumulator.rows(source, action, case, scope, "case_voxel_bins")
        for row in rows:
            decile_sink.write(row)
            key = (source, action, scope)
            macro = global_decile.setdefault(key, [RunningMean() for _ in range(10)])
            macro[int(row["decile"]) - 1].update(row["pseudo_label_accuracy"])

    for source in RELIABILITY_SOURCES:
        for scope in ("overall", "foreground", "background"):
            accumulators = [
                decile_acc[(source, action_name, scope)]
                for action_name in ("gamma", "blur")
                if (source, action_name, scope) in decile_acc
            ]
            if not accumulators:
                continue
            combined = DecileAccumulator()
            for accumulator in accumulators:
                combined.count += accumulator.count
                combined.correct += accumulator.correct
                combined.reliability_sum += accumulator.reliability_sum
            for row in combined.rows(source, "all", case, scope, "case_voxel_bins"):
                decile_sink.write(row)
                key = (source, "all", scope)
                macro = global_decile.setdefault(key, [RunningMean() for _ in range(10)])
                macro[int(row["decile"]) - 1].update(row["pseudo_label_accuracy"])


def memory_summary(device: torch.device) -> dict[str, float]:
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if device.type == "cuda" and torch.cuda.is_available():
        return {
            "cpu_rss_mb": float(rss),
            "gpu_allocated_mb": float(torch.cuda.memory_allocated(device) / 1024**2),
            "gpu_reserved_mb": float(torch.cuda.memory_reserved(device) / 1024**2),
        }
    return {"cpu_rss_mb": float(rss), "gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0}


def run(args: argparse.Namespace) -> None:
    validate_protocol(args)
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V8.2 requires CUDA, resolved {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    test_cases = iter_cases(paths, split="test")
    if len(test_cases) != 8:
        raise AssertionError(f"V8.2 requires exactly 8 test cases, got {len(test_cases)}")
    checkpoint_path = Path(args.world_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing V8.0 checkpoint: {checkpoint_path}")
    feature_sink = CsvSink(output_dir / "feature_dynamics.csv", FEATURE_FIELDS)
    segmentation_sink = CsvSink(output_dir / "segmentation_relevance.csv", SEGMENTATION_FIELDS)
    full_volume_sink = CsvSink(output_dir / "full_volume_action_summary.csv", FULL_VOLUME_FIELDS)
    case_summary_sink = CsvSink(output_dir / "case_action_summary.csv", CASE_SUMMARY_FIELDS)
    reliability_metric_sink = CsvSink(output_dir / "reliability_validity_summary.csv", RELIABILITY_METRIC_FIELDS)
    reliability_region_sink = CsvSink(output_dir / "reliability_voxel_summary.csv", RELIABILITY_VOXEL_FIELDS)
    decile_sink = CsvSink(output_dir / "reliability_deciles.csv", RELIABILITY_DECILE_FIELDS)
    progress_path = output_dir / "progress.json"
    completed_cases: list[str] = []
    completed_rows = 0
    write_progress(progress_path, completed_cases, completed_rows, device, "initializing")

    try:
        print(f"[V8.2] loading World Predictor checkpoint {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "V8.0 full-train visual World Predictor":
            raise AssertionError("V8.2 requires the V8.0 World Predictor checkpoint")

        print("[V8.2] loading frozen VoxTell/base model", flush=True)
        interface = VoxTellStateInterface.from_model_dir(
            paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
        )
        prepare_functional_seg_head(interface, args.selected_stage)
        interface._v8_prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
        # The diagnostic only uses the cached prompt embedding and the image
        # network from this point onward.  Release the large text encoder before
        # case processing; this does not alter the prompt or any experiment rule.
        text_backbone = getattr(interface.predictor, "text_backbone", None)
        interface.predictor.text_backbone = None
        interface.predictor.tokenizer = None
        interface.predictor._text_embedding_cache.clear()
        del text_backbone
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("[V8.2] loading frozen World Predictor weights", flush=True)
        world_model = load_world_model(
            checkpoint_path,
            int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
            device,
            args.hidden_channels,
        )
        world_model.eval()
        del checkpoint
        gc.collect()
        write_progress(progress_path, completed_cases, completed_rows, device, "running")

        global_metric: dict[tuple[str, str, str], MetricAccumulator] = {}
        global_region: dict[tuple[str, str, str], dict[str, RunningMean]] = {}
        global_decile: dict[tuple[str, str, str], list[RunningMean]] = {}
        global_summary: dict[str, dict[str, RunningMean]] = {}
        full_volume_global: dict[str, dict[str, RunningMean]] = {}
        invalid_counts = defaultdict(int)

        for case_index, case in enumerate(test_cases, start=1):
            print(f"[V8.2] case {case_index}/8 start {case.case}", flush=True)
            image, label, _ = read_image_and_label(case)
            label_padded = pad_label_like_image(interface, label)
            for full_action, full_strength in (("source", 0.0), *ACTION_PROTOCOL):
                print(
                    f"[V8.2] case {case_index}/8 full-volume action={full_action} start",
                    flush=True,
                )
                full_metrics = full_volume_action_metrics(
                    interface,
                    image,
                    label,
                    interface._v8_prompt_embedding,
                    full_action,
                    full_strength,
                    args.label_value,
                    args.prediction_threshold,
                )
                full_volume_sink.write({
                    "case": case.case,
                    "action": full_action,
                    "strength": full_strength,
                    **full_metrics,
                    "aggregation": "per_case",
                })
                full_volume_global.setdefault(full_action, {})
                for metric_name in ("dice", "iou", "precision", "recall"):
                    full_volume_global[full_action].setdefault(metric_name, RunningMean()).update(
                        full_metrics[metric_name],
                    )
                print(
                    f"[V8.2] case {case_index}/8 full-volume action={full_action} done",
                    flush=True,
                )
            original_padded, slicers, patch_kinds = select_patch_slicers(
                interface,
                image,
                interface._v8_prompt_embedding,
                args.patches_per_case,
                args.foreground_patches_per_case,
                args.foreground_candidate_patches,
                args.foreground_threshold,
            )
            if len(slicers) != args.patches_per_case:
                raise AssertionError(f"Patch selector returned too few patches for {case.case}")

            case_rows: list[dict[str, Any]] = []
            case_metric: dict[tuple[str, str, str], MetricAccumulator] = {}
            case_region: dict[tuple[str, str, str], Histogram] = {}
            case_decile: dict[tuple[str, str, str], DecileAccumulator] = {}
            for patch_index, slicer in enumerate(slicers, start=1):
                print(f"[V8.2] case {case_index}/8 patch {patch_index}/{len(slicers)} source", flush=True)
                source_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
                gt_patch = label_padded[slicer][None] == args.label_value
                with torch.inference_mode():
                    source_result = interface.forward_with_states(source_patch, interface._v8_prompt_embedding)
                    source_state = source_result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
                    source_probability = torch.sigmoid(source_result["final_prediction"][:, 0:1].detach().float().to(device))
                    source_pseudo = source_probability > args.prediction_threshold
                    source_logits = state_to_intermediate_prediction(interface, args.selected_stage, source_state)
                    source_segmentation = segmentation_metrics(source_logits, gt_patch, args.prediction_threshold)
                del source_result, source_logits, source_patch

                final_shape = tuple(int(size) for size in source_probability.shape[-3:])
                with torch.inference_mode():
                    reliability_maps = reliability_from_source(
                        interface, world_model, source_state, source_probability,
                        args.selected_stage, final_shape, device,
                    )
                gt_np = gt_patch.detach().cpu().numpy().astype(bool)
                pseudo_np = source_pseudo.detach().cpu().numpy().astype(bool)
                correct_np = pseudo_np == gt_np
                gamma_result: dict[str, Any] | None = None
                gamma_state: torch.Tensor | None = None

                for action, strength in ACTION_PROTOCOL:
                    print(
                        f"[V8.2] case {case_index}/8 patch {patch_index}/{len(slicers)} action={action} feature start",
                        flush=True,
                    )
                    transformed_padded, _ = padded_visual_action_and_slicers(
                        interface.predictor, image, action, strength,
                    )
                    if tuple(transformed_padded.shape) != tuple(original_padded.shape):
                        raise AssertionError(f"Action padded shape mismatch for {case.case} {action}")
                    actual_patch = torch.clone(transformed_padded[slicer][None], memory_format=torch.contiguous_format)
                    wrong_action, wrong_strength = MATCHED_WRONG_ACTIONS[action]
                    result, conditioned_state = evaluate_action(
                        interface,
                        world_model,
                        source_state,
                        actual_patch,
                        action,
                        strength,
                        wrong_action,
                        wrong_strength,
                        source_segmentation,
                        gt_patch,
                        args.selected_stage,
                        args.prediction_threshold,
                        device,
                    )
                    result.update({
                        "case": case.case,
                        "patch_index": patch_index - 1,
                        "patch_kind": patch_kinds[patch_index - 1],
                    })
                    print(
                        f"[V8.2] case {case_index}/8 patch {patch_index}/{len(slicers)} action={action} feature done",
                        flush=True,
                    )
                    del transformed_padded, actual_patch

                    if action == "gamma":
                        gamma_result = result
                        gamma_state = conditioned_state
                        print(
                            f"[V8.2] case {case_index}/8 patch {patch_index}/{len(slicers)} action=gamma segmentation done",
                            flush=True,
                        )
                        continue

                    if gamma_result is None or gamma_state is None:
                        raise AssertionError("Gamma must be processed before blur")
                    gamma_blur_distance = rms_distance(gamma_state, conditioned_state)
                    result["gamma_vs_blur_conditioned_distance"] = gamma_blur_distance
                    gamma_result["gamma_vs_blur_conditioned_distance"] = gamma_blur_distance
                    for row in (gamma_result, result):
                        feature_sink.write({name: row.get(name) for name in FEATURE_FIELDS})
                        segmentation_sink.write({name: row.get(name) for name in SEGMENTATION_FIELDS})
                    case_rows.extend((gamma_result, result))
                    print(
                        f"[V8.2] case {case_index}/8 patch {patch_index}/{len(slicers)} action={action} segmentation done",
                        flush=True,
                    )

                    for output_name in RELIABILITY_SOURCES:
                        for action_name in ("gamma", "blur"):
                            update_reliability(
                                output_name,
                                action_name,
                                case.case,
                                reliability_maps,
                                correct_np,
                                gt_np,
                                pseudo_np,
                                case_metric,
                                case_region,
                                case_decile,
                            )
                    print(
                        f"[V8.2] case {case_index}/8 patch {patch_index}/{len(slicers)} action=gamma reliability done",
                        flush=True,
                    )
                    print(
                        f"[V8.2] case {case_index}/8 patch {patch_index}/{len(slicers)} action=blur reliability done",
                        flush=True,
                    )

                    for row in (gamma_result, result):
                        for name in SUMMARY_NUMERIC_NAMES:
                            if row.get(name) is None:
                                if "cosine" in name:
                                    invalid_counts[name] += 1
                        case_rows[-1 if row is result else -2] = row
                    del result, conditioned_state, gamma_result, gamma_state
                    gamma_result = None
                    gamma_state = None

                del reliability_maps, source_state, source_probability, source_pseudo
                del gt_np, pseudo_np, correct_np, gt_patch
                gc.collect()

            for action in ("gamma", "blur"):
                action_rows = [row for row in case_rows if row["action"] == action]
                row = aggregate_row(action_rows, "per_case", case.case, action)
                case_summary_sink.write(row)
                update_summary_macro(global_summary, action, row)
            case_all = aggregate_row(case_rows, "per_case", case.case, "all")
            case_summary_sink.write(case_all)
            update_summary_macro(global_summary, "all", case_all)
            reliability_case_rows(
                case.case,
                case_metric,
                case_region,
                case_decile,
                reliability_metric_sink,
                reliability_region_sink,
                decile_sink,
                global_metric,
                global_region,
                global_decile,
            )
            completed_cases.append(case.case)
            completed_rows += len(case_rows)
            progress_path.write_text(json.dumps({
                "stage": "running",
                "completed_cases": completed_cases,
                "completed_case_count": len(completed_cases),
                "completed_rows": completed_rows,
                "memory": memory_summary(device),
            }, indent=2))
            print(
                f"[V8.2] case {case_index}/8 complete rows={len(case_rows)} memory={memory_summary(device)}",
                flush=True,
            )
            del case_rows, case_metric, case_region, case_decile
            del label_padded, original_padded, image, label
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for action in ("gamma", "blur", "all"):
            case_summary_sink.write(summary_macro_row(global_summary, action, action))
        for (source, action, scope), accumulator in sorted(global_metric.items()):
            row = accumulator.row(source, action, "__all__", scope)
            row["aggregation"] = "case_level_macro"
            reliability_metric_sink.write(row)
        for (source, action, region), stats in sorted(global_region.items()):
            reliability_region_sink.write({
                "source": source,
                "action": action,
                "case": "__all__",
                "region": region,
                "count": None,
                **{name: stats[name].mean() for name in ("mean", "median", "p10", "p50", "p90")},
            })
        for (source, action, scope), decile_stats in sorted(global_decile.items()):
            for index, stat in enumerate(decile_stats, start=1):
                if stat.count:
                    decile_sink.write({
                        "source": source,
                        "action": action,
                        "case": "__all__",
                        "scope": scope,
                        "decile": index,
                        "voxel_count": None,
                        "reliability_mean": None,
                        "pseudo_label_accuracy": stat.mean(),
                        "aggregation": "case_level_macro",
                    })

        for action, metrics in sorted(full_volume_global.items()):
            full_volume_sink.write({
                "case": "__all__",
                "action": action,
                "strength": 0.0 if action == "source" else dict(ACTION_PROTOCOL)[action],
                **{name: value.mean() for name, value in metrics.items()},
                "aggregation": "case_level_macro",
            })

        overall = summary_macro_row(global_summary, "all", "all")
        gamma = summary_macro_row(global_summary, "gamma", "gamma")
        blur = summary_macro_row(global_summary, "blur", "blur")
        overall_metric = {
            source: next(
                (row for row in global_metric_rows(global_metric, source, "all", "overall")),
                None,
            )
            for source in RELIABILITY_SOURCES
        }
        region_lookup = {
            (source, action, region): stats
            for (source, action, region), stats in global_region.items()
        }
        cond_mse = overall["mean_conditioned_state_mse"]
        same_model_no_action_mse = overall["mean_same_model_no_action_state_mse"]
        matched_wrong_mse = overall["mean_matched_wrong_state_mse"]
        cond_cos = overall["mean_conditioned_delta_cosine"]
        same_model_no_action_cos = overall["mean_same_model_no_action_delta_cosine"]
        cond_dice = overall["mean_conditioned_dice"]
        same_model_no_action_dice = overall["mean_same_model_no_action_dice"]
        matched_wrong_dice = overall["mean_matched_wrong_dice"]
        def strictly_less(left: Any, right: Any) -> bool:
            return left is not None and right is not None and float(left) < float(right)

        feature_better = strictly_less(cond_mse, same_model_no_action_mse)
        wrong_better = strictly_less(cond_mse, matched_wrong_mse)
        cosine_better = cond_cos is not None and same_model_no_action_cos is not None and cond_cos > same_model_no_action_cos
        seg_better_same_model = (
            cond_dice is not None and same_model_no_action_dice is not None
            and cond_dice > same_model_no_action_dice
        )
        seg_better_wrong = cond_dice is not None and matched_wrong_dice is not None and cond_dice > matched_wrong_dice
        available = [source for source, row in overall_metric.items() if row and row.get("auroc") is not None]
        best_reliability = max(available, key=lambda source: float(overall_metric[source]["auroc"])) if available else None
        foreground_metric = {
            source: next(
                (row for row in global_metric_rows(global_metric, source, "all", "foreground")),
                None,
            )
            for source in RELIABILITY_SOURCES
        }
        foreground_available = [
            source for source, row in foreground_metric.items()
            if row and row.get("auroc") is not None
        ]
        best_foreground_reliability = (
            max(foreground_available, key=lambda source: float(foreground_metric[source]["auroc"]))
            if foreground_available else None
        )

        def region_mean(source: str, region: str) -> float | None:
            stats = region_lookup.get((source, "all", region), {})
            statistic = stats.get("mean")
            return statistic.mean() if statistic is not None else None

        world_region_means = {region: region_mean("world_pairwise", region) for region in ("TP", "FP", "FN", "TN")}
        world_fn_overweight = (
            world_region_means["FN"] is not None
            and world_region_means["TP"] is not None
            and world_region_means["FN"] > world_region_means["TP"]
        )

        def status_supported(condition: bool, available_values: tuple[Any, ...]) -> str:
            return "unavailable" if any(value is None for value in available_values) else (
                "supported" if condition else "not_supported"
            )

        world_action_status = status_supported(
            feature_better and wrong_better and cosine_better,
            (cond_mse, same_model_no_action_mse, matched_wrong_mse, cond_cos, same_model_no_action_cos),
        )
        segmentation_status = status_supported(
            seg_better_same_model and seg_better_wrong,
            (cond_dice, same_model_no_action_dice, matched_wrong_dice),
        )
        reliability_values = tuple(
            row.get("auroc") if row is not None else None for row in foreground_metric.values()
        )
        reliability_status = status_supported(
            best_reliability == "world_pairwise" and not world_fn_overweight,
            reliability_values,
        )
        summary = {
            "stage": "V8.2 World Predictor mechanism diagnosis",
            "checkpoint": str(checkpoint_path),
            "test_cases": [case.case for case in test_cases],
            "test_case_count": len(test_cases),
            "completed_case_count": len(completed_cases),
            "completed_rows": completed_rows,
            "expected_rows": len(test_cases) * args.patches_per_case * len(ACTION_PROTOCOL),
            "gt_used_only_for_diagnosis": True,
            "gradient_updates": 0,
            "world_predictor_retrained": False,
            "lora_trained": False,
            "streaming_memory_mode": True,
            "reliability_formula_modified": False,
            "patch_protocol": {
                "patches_per_case": args.patches_per_case,
                "foreground_patches_per_case": args.foreground_patches_per_case,
            },
            "action_protocol": {"gamma": 0.30, "blur": 1.5},
            "matched_wrong_action_protocol": {
                "gamma": {"family": "blur", "strength": 1.5},
                "blur": {"family": "gamma", "strength": 0.30},
            },
            "same_model_no_action_definition": {
                "available": True,
                "independent_same_model_no_action_checkpoint_available": False,
                "note": "same_model_no_action is world_model(source_state, action=None), not an independently trained model; an independent no-action checkpoint is unavailable and no checkpoint was trained in V8.2-fix",
            },
            "invalid_zero_norm_counts": dict(invalid_counts),
            "answers": {
                "world_action_dynamics_status": world_action_status,
                "segmentation_relevance_status": segmentation_status,
                "reliability_mapping_status": reliability_status,
                "conditioned_feature_mse_better_than_same_model_no_action": {
                    "supported": feature_better,
                    "conditioned_mean_mse": cond_mse,
                    "same_model_no_action_mean_mse": same_model_no_action_mse,
                    "margin_same_model_no_action_minus_conditioned": None if cond_mse is None or same_model_no_action_mse is None else same_model_no_action_mse - cond_mse,
                },
                "correct_action_better_than_matched_wrong_action": {
                    "supported": wrong_better,
                    "conditioned_mean_mse": cond_mse,
                    "matched_wrong_mean_mse": matched_wrong_mse,
                },
                "conditioned_delta_cosine_better_than_same_model_no_action": {
                    "supported": cosine_better,
                    "conditioned_mean_cosine": cond_cos,
                    "same_model_no_action_mean_cosine": same_model_no_action_cos,
                },
                "gamma_conditioned_feature_prediction": {
                    "conditioned_better_than_same_model_no_action": strictly_less(gamma["mean_conditioned_state_mse"], gamma["mean_same_model_no_action_state_mse"]),
                    "conditioned_better_than_matched_wrong": strictly_less(gamma["mean_conditioned_state_mse"], gamma["mean_matched_wrong_state_mse"]),
                },
                "blur_conditioned_feature_prediction": {
                    "conditioned_better_than_same_model_no_action": strictly_less(blur["mean_conditioned_state_mse"], blur["mean_same_model_no_action_state_mse"]),
                    "conditioned_better_than_matched_wrong": strictly_less(blur["mean_conditioned_state_mse"], blur["mean_matched_wrong_state_mse"]),
                },
                "conditioned_intermediate_segmentation_better_than_same_model_no_action": {
                    "supported": seg_better_same_model,
                    "conditioned_mean_dice": cond_dice,
                    "same_model_no_action_mean_dice": same_model_no_action_dice,
                },
                "conditioned_intermediate_segmentation_better_than_matched_wrong": {
                    "supported": seg_better_wrong,
                    "conditioned_mean_dice": cond_dice,
                    "matched_wrong_mean_dice": matched_wrong_dice,
                },
                "real_full_volume_action_segmentation": {
                    "source": {name: value.mean() for name, value in full_volume_global.get("source", {}).items()},
                    "gamma": {name: value.mean() for name, value in full_volume_global.get("gamma", {}).items()},
                    "blur": {name: value.mean() for name, value in full_volume_global.get("blur", {}).items()},
                    "imagined_world_state_full_volume_available": False,
                    "imagined_world_state_full_volume_note": "V8.0 World Predictor state is a selected intermediate decoder state; it is not reconnected to the complete sliding-window final decoder in this diagnostic",
                },
                "foreground_reliability_predictors": {
                    "source": best_foreground_reliability,
                    "auroc": {source: None if foreground_metric[source] is None else foreground_metric[source]["auroc"] for source in RELIABILITY_SOURCES},
                    "spearman": {source: None if foreground_metric[source] is None else foreground_metric[source]["spearman"] for source in RELIABILITY_SOURCES},
                },
                "reliability_predictors_overall": {
                    "source": best_reliability,
                    "auroc": {source: None if overall_metric[source] is None else overall_metric[source]["auroc"] for source in RELIABILITY_SOURCES},
                    "auprc": {source: None if overall_metric[source] is None else overall_metric[source]["auprc"] for source in RELIABILITY_SOURCES},
                    "spearman": {source: None if overall_metric[source] is None else overall_metric[source]["spearman"] for source in RELIABILITY_SOURCES},
                },
                "world_reliability_fn_check": {
                    "supported": world_fn_overweight,
                    "world_pairwise_region_mean": world_region_means,
                    "note": "FN is flagged when its world_pairwise mean reliability directly exceeds TP; TN is not used as the comparison threshold. All four region means are retained in reliability_voxel_summary.csv",
                },
            },
            "outputs": {
                "feature_dynamics": str(output_dir / "feature_dynamics.csv"),
                "segmentation_relevance": str(output_dir / "segmentation_relevance.csv"),
                "full_volume_action_summary": str(output_dir / "full_volume_action_summary.csv"),
                "reliability_voxel_summary": str(output_dir / "reliability_voxel_summary.csv"),
                "reliability_validity_summary": str(output_dir / "reliability_validity_summary.csv"),
                "reliability_deciles": str(output_dir / "reliability_deciles.csv"),
                "case_action_summary": str(output_dir / "case_action_summary.csv"),
                "progress": str(progress_path),
                "summary": str(output_dir / "summary.json"),
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_progress(progress_path, completed_cases, completed_rows, device, "complete")
    finally:
        for sink in (
            feature_sink, segmentation_sink, full_volume_sink, case_summary_sink,
            reliability_metric_sink, reliability_region_sink, decile_sink,
        ):
            sink.close()


def global_metric_rows(
    accumulator: dict[tuple[str, str, str], MetricAccumulator],
    source: str,
    action: str,
    scope: str,
) -> list[dict[str, Any]]:
    accumulator_item = accumulator.get((source, action, scope))
    if accumulator_item is None:
        return []
    row = accumulator_item.row(source, action, "__all__", scope)
    return [row]


if __name__ == "__main__":
    run(parse_args())
