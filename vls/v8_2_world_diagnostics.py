"""V8.2 mechanism diagnosis for the frozen V8.0 World Predictor.

This script performs diagnosis only.  It does not train, select, or modify a
World Predictor or LoRA model, and it keeps the V8.0/V8.1 action and
reliability definitions unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vls.config import DEFAULT_LABEL_VALUE, DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases, read_image, read_image_and_label
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
from vls.v7_0d_protocol_sanity import (
    STRONG_ACTIONS,
    binary_metrics,
    reliability_from_source,
    set_seed,
)
from vls.v7_1b_protocol_consolidation import write_csv
from vls.voxtell_states import VoxTellStateInterface


OUTPUT_DIR = Path("outputs/v8_2_world_diagnostics")
ACTION_PROTOCOL = (("gamma", 0.30), ("blur", 1.5))
RELIABILITY_SOURCES = {
    "confidence_rank": "confidence_rank",
    "world_pairwise": "world_stability",
    "joint_product": "joint_product",
}


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V8.2 World Predictor mechanism diagnosis")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/v8_0_full_world_predictor/best_world_predictor.pt",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def validate_protocol(args: argparse.Namespace) -> None:
    if tuple(STRONG_ACTIONS) != ACTION_PROTOCOL:
        raise AssertionError("V8.2 action protocol differs from the frozen V8 protocol")
    if set(RELIABILITY_SOURCES.values()) != {
        "confidence_rank", "world_stability", "joint_product",
    }:
        raise AssertionError("V8.2 reliability source definitions changed")
    if args.patches_per_case <= 0 or args.batch_size <= 0:
        raise AssertionError("V8.2 patches_per_case and batch_size must be positive")


def slicer_from_parts(starts: list[int], stops: list[int]) -> tuple[slice, ...]:
    return (slice(None), *(slice(int(start), int(stop), None) for start, stop in zip(starts, stops, strict=True)))


def rms_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt((left.float() - right.float()).pow(2).mean()).detach().cpu())


def state_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction.float() - target.float()).pow(2).mean().detach().cpu())


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> tuple[float | None, bool]:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    valid = bool(float(left_norm.detach().cpu()) > 0.0 and float(right_norm.detach().cpu()) > 0.0)
    if not valid:
        return None, False
    return float(torch.dot(left_flat, right_flat).div(left_norm * right_norm).detach().cpu()), True


def norm_ratio(delta: torch.Tensor, true_delta: torch.Tensor) -> tuple[float | None, bool]:
    denominator = float(torch.linalg.vector_norm(true_delta.float()).detach().cpu())
    if denominator <= 0.0:
        return None, False
    numerator = float(torch.linalg.vector_norm(delta.float()).detach().cpu())
    return numerator / denominator, True


def iou_from_metrics(metrics: dict[str, float | int]) -> float:
    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    return 1.0 if tp + fp + fn == 0 else tp / max(tp + fp + fn, 1)


def segmentation_metrics(
    logits: torch.Tensor,
    gt: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    if tuple(logits.shape[-3:]) != tuple(gt.shape[-3:]):
        gt = F.interpolate(gt.float(), size=logits.shape[-3:], mode="nearest") > 0.5
    prediction = (torch.sigmoid(logits) > threshold).detach().cpu().numpy().astype(bool).reshape(-1)
    target = gt.detach().cpu().numpy().astype(bool).reshape(-1)
    metrics = binary_metrics(prediction, target)
    return {
        "dice": float(metrics["dice"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "iou": iou_from_metrics(metrics),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    positives = labels == 1
    negatives = labels == 0
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = rankdata(scores)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def auprc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    positive_count = int((labels == 1).sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels == 1)
    positions = np.arange(1, labels.size + 1, dtype=np.float64)
    precision = cumulative / positions
    return float(precision[sorted_labels == 1].sum() / positive_count)


def spearman(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    score_rank = rankdata(scores)
    label_rank = rankdata(labels)
    score_centered = score_rank - score_rank.mean()
    label_centered = label_rank - label_rank.mean()
    denominator = float(np.linalg.norm(score_centered) * np.linalg.norm(label_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(score_centered, label_centered) / denominator)


def quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def decile_index(values: np.ndarray) -> np.ndarray:
    return np.minimum((np.asarray(values, dtype=np.float64) * 10.0).astype(np.int64) + 1, 10)


def add_decile_stats(
    accumulator: dict[tuple[str, str, str, str, int], list[float]],
    source: str,
    action: str,
    case: str,
    reliability: np.ndarray,
    correct: np.ndarray,
    gt: np.ndarray,
) -> None:
    flat_reliability = reliability.reshape(-1).astype(np.float64)
    flat_correct = correct.reshape(-1).astype(bool)
    flat_gt = gt.reshape(-1).astype(bool)
    scopes = {
        "overall": np.ones(flat_gt.size, dtype=bool),
        "foreground": flat_gt,
        "background": ~flat_gt,
        f"case:{case}": np.ones(flat_gt.size, dtype=bool),
    }
    deciles = decile_index(flat_reliability)
    for scope, scope_mask in scopes.items():
        for decile in range(1, 11):
            mask = scope_mask & (deciles == decile)
            if not bool(mask.any()):
                continue
            for group_action in (action, "all"):
                for group_case in ("__all__", case):
                    key = (source, group_action, group_case, scope, decile)
                    item = accumulator.setdefault(key, [0.0, 0.0, 0.0])
                    item[0] += float(mask.sum())
                    item[1] += float(flat_correct[mask].sum())
                    item[2] += float(flat_reliability[mask].sum())


def add_metric_group(
    accumulator: dict[tuple[str, str, str, str], list[tuple[np.ndarray, np.ndarray]]],
    source: str,
    action: str,
    case: str,
    reliability: np.ndarray,
    correct: np.ndarray,
    gt: np.ndarray,
) -> None:
    flat_reliability = reliability.reshape(-1).astype(np.float64)
    flat_correct = correct.reshape(-1).astype(np.int8)
    flat_gt = gt.reshape(-1).astype(bool)
    for scope, mask in (
        ("overall", np.ones(flat_gt.size, dtype=bool)),
        ("foreground", flat_gt),
        ("background", ~flat_gt),
    ):
        for group_case in ("__all__", case):
            key = (source, action, group_case, scope)
            accumulator.setdefault(key, []).append((flat_reliability[mask], flat_correct[mask]))


def add_region_values(
    accumulator: dict[tuple[str, str, str, str], list[np.ndarray]],
    source: str,
    action: str,
    case: str,
    reliability: np.ndarray,
    gt: np.ndarray,
    pseudo: np.ndarray,
) -> None:
    flat_reliability = reliability.reshape(-1).astype(np.float64)
    flat_gt = gt.reshape(-1).astype(bool)
    flat_pseudo = pseudo.reshape(-1).astype(bool)
    regions = {
        "TP": flat_pseudo & flat_gt,
        "FP": flat_pseudo & ~flat_gt,
        "TN": ~flat_pseudo & ~flat_gt,
        "FN": ~flat_pseudo & flat_gt,
    }
    for region, mask in regions.items():
        for group_action in (action, "all"):
            for group_case in ("__all__", case):
                key = (source, group_action, group_case, region)
                accumulator.setdefault(key, []).append(flat_reliability[mask])


def mean_numeric(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None and np.isfinite(float(row[name]))]
    return float(np.mean(values)) if values else None


def combined_summary_rows(combined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_names = [
        "true_shift_norm", "conditioned_shift_norm", "agnostic_shift_norm", "wrong_shift_norm",
        "conditioned_vs_agnostic_distance", "gamma_vs_blur_conditioned_distance",
        "conditioned_state_mse", "agnostic_state_mse", "wrong_state_mse",
        "correct_vs_agnostic_margin", "correct_vs_wrong_margin",
        "conditioned_delta_cosine", "agnostic_delta_cosine", "wrong_delta_cosine",
        "magnitude_ratio_conditioned", "source_dice", "actual_dice", "conditioned_dice",
        "agnostic_dice", "wrong_dice", "source_precision", "actual_precision",
        "conditioned_precision", "agnostic_precision", "wrong_precision", "source_recall",
        "actual_recall", "conditioned_recall", "agnostic_recall", "wrong_recall",
        "source_iou", "actual_iou", "conditioned_iou", "agnostic_iou", "wrong_iou",
        "conditioned_vs_actual_seg_logits_mse", "agnostic_vs_actual_seg_logits_mse",
        "wrong_vs_actual_seg_logits_mse", "seg_gain_vs_agnostic", "seg_gain_vs_wrong",
        "repeat_forward_distance",
    ]
    groups: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for action in ("gamma", "blur"):
        groups.append(("action", "__all__", action, [row for row in combined_rows if row["action"] == action]))
    for case in sorted({row["case"] for row in combined_rows}):
        groups.append(("case", case, "all", [row for row in combined_rows if row["case"] == case]))
        for action in ("gamma", "blur"):
            groups.append(("case_action", case, action, [
                row for row in combined_rows if row["case"] == case and row["action"] == action
            ]))
    groups.append(("overall", "__all__", "all", combined_rows))
    output = []
    for level, case, action, rows in groups:
        item: dict[str, Any] = {
            "summary_level": level,
            "case": case,
            "action": action,
            "row_count": len(rows),
        }
        for name in numeric_names:
            item[f"mean_{name}"] = mean_numeric(rows, name)
        output.append(item)
    return output


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
        raise AssertionError(f"V8.2 diagnostic set must contain exactly 8 test cases, got {len(test_cases)}")
    test_names = [case.case for case in test_cases]

    checkpoint_path = Path(args.world_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing V8.0 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "V8.0 full-train visual World Predictor":
        raise AssertionError("V8.2 requires the V8.0 full-train World Predictor checkpoint")
    if not checkpoint.get("test_used_for_checkpoint_selection"):
        raise AssertionError("V8.0 checkpoint metadata is missing test checkpoint-selection provenance")

    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    prepare_functional_seg_head(interface, args.selected_stage)
    prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    world_model = load_world_model(
        checkpoint_path,
        int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
        device,
        args.hidden_channels,
    )
    world_model.eval()

    feature_rows: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    metric_groups: dict[tuple[str, str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    region_groups: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    decile_groups: dict[tuple[str, str, str, str, int], list[float]] = {}
    case_patch_counts: dict[str, int] = defaultdict(int)
    total_rows = 0
    invalid_cosine_counts: dict[str, int] = defaultdict(int)
    invalid_ratio_count = 0

    for case in test_cases:
        image, label, _ = read_image_and_label(case)
        label_padded = pad_label_like_image(interface, label)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface,
            image,
            prompt_embedding,
            args.patches_per_case,
            args.foreground_patches_per_case,
            args.foreground_candidate_patches,
            args.foreground_threshold,
        )
        if len(slicers) != args.patches_per_case:
            raise AssertionError(f"V8.2 patch selector returned too few patches for {case.case}")
        for patch_index, slicer in enumerate(slicers):
            case_patch_counts[case.case] += 1
            source_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            gt_patch = (label_padded[slicer][None] == args.label_value)
            source_result = interface.forward_with_states(source_patch, prompt_embedding)
            source_state = source_result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
            source_probability = torch.sigmoid(source_result["final_prediction"][:, 0:1].detach().float().to(device))
            source_pseudo = source_probability > args.prediction_threshold
            final_shape = tuple(int(size) for size in source_probability.shape[-3:])
            reliability_maps = reliability_from_source(
                interface,
                world_model,
                source_state,
                source_probability,
                args.selected_stage,
                final_shape,
                device,
            )
            gt_np = gt_patch.detach().cpu().numpy().astype(bool)
            pseudo_np = source_pseudo.detach().cpu().numpy().astype(bool)
            correct_np = (pseudo_np == gt_np)

            action_cache: list[dict[str, Any]] = []
            transformed_by_action = {
                action: padded_visual_action_and_slicers(
                    interface.predictor, image, action, strength,
                )[0]
                for action, strength in ACTION_PROTOCOL
            }
            for action, strength in ACTION_PROTOCOL:
                transformed_padded = transformed_by_action[action]
                if tuple(transformed_padded.shape) != tuple(original_padded.shape):
                    raise AssertionError(f"Action padded shape mismatch for {case.case} {action}")
                actual_patch = torch.clone(transformed_padded[slicer][None], memory_format=torch.contiguous_format)
                true_result = interface.forward_with_states(actual_patch, prompt_embedding)
                true_state = true_result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
                action_tensor = visual_action(action, strength, device)
                wrong_action = "blur" if action == "gamma" else "gamma"
                conditioned_state = world_model(source_state, action=action_tensor)
                repeated_state = world_model(source_state, action=action_tensor)
                repeat_distance = rms_distance(conditioned_state, repeated_state)
                agnostic_state = world_model(source_state, action=None)
                wrong_state = world_model(
                    source_state,
                    action=visual_action(wrong_action, strength, device),
                )
                true_delta = true_state - source_state
                conditioned_delta = conditioned_state - source_state
                agnostic_delta = agnostic_state - source_state
                wrong_delta = wrong_state - source_state
                cond_cos, cond_cos_valid = cosine_similarity(true_delta, conditioned_delta)
                agn_cos, agn_cos_valid = cosine_similarity(true_delta, agnostic_delta)
                wrong_cos, wrong_cos_valid = cosine_similarity(true_delta, wrong_delta)
                ratio, ratio_valid = norm_ratio(conditioned_delta, true_delta)
                invalid_cosine_counts["conditioned"] += int(not cond_cos_valid)
                invalid_cosine_counts["agnostic"] += int(not agn_cos_valid)
                invalid_cosine_counts["wrong"] += int(not wrong_cos_valid)
                invalid_ratio_count += int(not ratio_valid)
                feature = {
                    "case": case.case,
                    "patch_index": patch_index,
                    "patch_kind": patch_kinds[patch_index],
                    "action": action,
                    "action_strength": strength,
                    "wrong_type_action": wrong_action,
                    "true_shift_norm": float(torch.linalg.vector_norm(true_delta).detach().cpu()),
                    "conditioned_shift_norm": float(torch.linalg.vector_norm(conditioned_delta).detach().cpu()),
                    "agnostic_shift_norm": float(torch.linalg.vector_norm(agnostic_delta).detach().cpu()),
                    "wrong_shift_norm": float(torch.linalg.vector_norm(wrong_delta).detach().cpu()),
                    "conditioned_vs_agnostic_distance": rms_distance(conditioned_state, agnostic_state),
                    "conditioned_state_mse": state_mse(conditioned_state, true_state),
                    "agnostic_state_mse": state_mse(agnostic_state, true_state),
                    "wrong_state_mse": state_mse(wrong_state, true_state),
                    "correct_vs_agnostic_margin": state_mse(agnostic_state, true_state) - state_mse(conditioned_state, true_state),
                    "correct_vs_wrong_margin": state_mse(wrong_state, true_state) - state_mse(conditioned_state, true_state),
                    "conditioned_delta_cosine": cond_cos,
                    "agnostic_delta_cosine": agn_cos,
                    "wrong_delta_cosine": wrong_cos,
                    "conditioned_delta_cosine_valid": cond_cos_valid,
                    "agnostic_delta_cosine_valid": agn_cos_valid,
                    "wrong_delta_cosine_valid": wrong_cos_valid,
                    "magnitude_ratio_conditioned": ratio,
                    "magnitude_ratio_conditioned_valid": ratio_valid,
                    "repeat_forward_distance": repeat_distance,
                }
                cond_logits = state_to_intermediate_prediction(interface, args.selected_stage, conditioned_state)
                true_logits = state_to_intermediate_prediction(interface, args.selected_stage, true_state)
                agn_logits = state_to_intermediate_prediction(interface, args.selected_stage, agnostic_state)
                wrong_logits = state_to_intermediate_prediction(interface, args.selected_stage, wrong_state)
                source_logits = state_to_intermediate_prediction(interface, args.selected_stage, source_state)
                seg_metrics = {}
                for name, logits in (
                    ("source", source_logits),
                    ("actual", true_logits),
                    ("conditioned", cond_logits),
                    ("agnostic", agn_logits),
                    ("wrong", wrong_logits),
                ):
                    metrics = segmentation_metrics(logits, gt_patch, args.prediction_threshold)
                    for metric_name in ("dice", "precision", "recall", "iou"):
                        seg_metrics[f"{name}_{metric_name}"] = metrics[metric_name]
                segmentation = {
                    "case": case.case,
                    "patch_index": patch_index,
                    "patch_kind": patch_kinds[patch_index],
                    "action": action,
                    "action_strength": strength,
                    **seg_metrics,
                    "conditioned_vs_actual_seg_logits_mse": state_mse(cond_logits, true_logits),
                    "agnostic_vs_actual_seg_logits_mse": state_mse(agn_logits, true_logits),
                    "wrong_vs_actual_seg_logits_mse": state_mse(wrong_logits, true_logits),
                    "seg_gain_vs_agnostic": float(seg_metrics["conditioned_dice"] - seg_metrics["agnostic_dice"]),
                    "seg_gain_vs_wrong": float(seg_metrics["conditioned_dice"] - seg_metrics["wrong_dice"]),
                }
                action_cache.append({
                    "action": action,
                    "feature": feature,
                    "segmentation": segmentation,
                    "conditioned_state": conditioned_state,
                    "reliability_maps": reliability_maps,
                })
                del transformed_padded, actual_patch, true_result, true_state

            if len(action_cache) != 2:
                raise AssertionError("V8.2 requires exactly gamma and blur per patch")
            gamma_state = next(item["conditioned_state"] for item in action_cache if item["action"] == "gamma")
            blur_state = next(item["conditioned_state"] for item in action_cache if item["action"] == "blur")
            gamma_blur_distance = rms_distance(gamma_state, blur_state)
            del gamma_state, blur_state
            for item in action_cache:
                feature = item["feature"]
                feature["gamma_vs_blur_conditioned_distance"] = gamma_blur_distance
                segmentation = item["segmentation"]
                combined = {**feature, **segmentation}
                feature_rows.append(feature)
                segmentation_rows.append(segmentation)
                combined_rows.append(combined)
                total_rows += 1
                for output_name, source_name in RELIABILITY_SOURCES.items():
                    reliability = np.asarray(item["reliability_maps"][source_name], dtype=np.float32)
                    add_metric_group(metric_groups, output_name, item["action"], case.case, reliability, correct_np, gt_np)
                    add_region_values(region_groups, output_name, item["action"], case.case, reliability, gt_np, pseudo_np)
                    add_decile_stats(decile_groups, output_name, item["action"], case.case, reliability, correct_np, gt_np)
                del item["conditioned_state"]
            del action_cache, transformed_by_action, source_result, source_state, source_probability
            del source_pseudo, reliability_maps, gt_np, pseudo_np, correct_np, source_patch, gt_patch

    if total_rows != len(test_cases) * args.patches_per_case * len(ACTION_PROTOCOL):
        raise AssertionError("V8.2 did not complete every case x patch x action row")
    if set(case_patch_counts) != set(test_names) or any(
        count != args.patches_per_case for count in case_patch_counts.values()
    ):
        raise AssertionError("V8.2 patch coverage is incomplete")

    reliability_summary_rows: list[dict[str, Any]] = []
    for (source, action, case, scope), groups in sorted(metric_groups.items()):
        scores = np.concatenate([pair[0] for pair in groups])
        labels = np.concatenate([pair[1] for pair in groups])
        reliability_summary_rows.append({
            "source": source,
            "action": action,
            "case": case,
            "scope": scope,
            "voxel_count": int(scores.size),
            "pseudo_accuracy": float(labels.mean()) if labels.size else None,
            "auroc": auroc(scores, labels),
            "auprc": auprc(scores, labels),
            "spearman": spearman(scores, labels),
            "reliability_mean": float(scores.mean()) if scores.size else None,
        })
    decile_rows: list[dict[str, Any]] = []
    for (source, action, case, scope, decile), values in sorted(decile_groups.items()):
        count, correct_count, reliability_sum = values
        decile_rows.append({
            "source": source,
            "action": action,
            "case": case,
            "scope": scope,
            "decile": decile,
            "voxel_count": int(count),
            "reliability_mean": reliability_sum / count,
            "pseudo_label_accuracy": correct_count / count,
        })
    voxel_rows: list[dict[str, Any]] = []
    for (source, action, case, region), groups in sorted(region_groups.items()):
        values = np.concatenate([values for values in groups]) if groups else np.asarray([], dtype=np.float64)
        voxel_rows.append({
            "source": source,
            "action": action,
            "case": case,
            "region": region,
            **quantiles(values),
        })

    case_action_rows = combined_summary_rows(combined_rows)
    overall_feature = next(row for row in case_action_rows if row["summary_level"] == "overall")
    action_feature = {
        action: next(row for row in case_action_rows if row["summary_level"] == "action" and row["action"] == action)
        for action in ("gamma", "blur")
    }
    # The metric table is keyed by action; make an all-action aggregation for
    # the automatic diagnosis without changing any voxel reliability formula.
    for source in RELIABILITY_SOURCES:
        for scope in ("overall", "foreground", "background"):
            groups = [
                pair
                for (src, _action, case, group_scope), pairs in metric_groups.items()
                if src == source and case == "__all__" and group_scope == scope
                for pair in pairs
            ]
            if not groups:
                continue
            scores = np.concatenate([pair[0] for pair in groups])
            labels = np.concatenate([pair[1] for pair in groups])
            reliability_summary_rows.append({
                "source": source,
                "action": "all",
                "case": "__all__",
                "scope": scope,
                "voxel_count": int(scores.size),
                "pseudo_accuracy": float(labels.mean()),
                "auroc": auroc(scores, labels),
                "auprc": auprc(scores, labels),
                "spearman": spearman(scores, labels),
                "reliability_mean": float(scores.mean()),
            })
    overall_reliability = {
        source: next((row for row in reliability_summary_rows if row["source"] == source and row["action"] == "all" and row["case"] == "__all__" and row["scope"] == "overall"), None)
        for source in RELIABILITY_SOURCES
    }
    region_lookup = {
        (row["source"], row["action"], row["case"], row["region"]): row
        for row in voxel_rows
    }
    def region_mean(source: str, region: str) -> float | None:
        return region_lookup.get((source, "all", "__all__", region), {}).get("mean")

    cond_mse = float(overall_feature["mean_conditioned_state_mse"])
    agn_mse = float(overall_feature["mean_agnostic_state_mse"])
    wrong_mse = float(overall_feature["mean_wrong_state_mse"])
    cond_cos = overall_feature["mean_conditioned_delta_cosine"]
    agn_cos = overall_feature["mean_agnostic_delta_cosine"]
    cond_dice = float(overall_feature["mean_conditioned_dice"])
    agn_dice = float(overall_feature["mean_agnostic_dice"])
    wrong_dice = float(overall_feature["mean_wrong_dice"])
    gamma_cond = action_feature["gamma"]
    blur_cond = action_feature["blur"]
    available_reliability = [
        source for source in RELIABILITY_SOURCES
        if overall_reliability[source] is not None and overall_reliability[source]["auroc"] is not None
    ]
    best_reliability = (
        max(available_reliability, key=lambda source: float(overall_reliability[source]["auroc"]))
        if available_reliability else None
    )
    world_fp_mean = region_mean("world_pairwise", "FP")
    world_tp_mean = region_mean("world_pairwise", "TP")
    world_fp_high = bool(
        world_fp_mean is not None and world_tp_mean is not None and world_fp_mean > world_tp_mean
    )
    feature_better = cond_mse < agn_mse
    wrong_better = cond_mse < wrong_mse
    cosine_better = cond_cos is not None and agn_cos is not None and cond_cos > agn_cos
    segmentation_better_agnostic = cond_dice > agn_dice
    segmentation_better_wrong = cond_dice > wrong_dice
    feature_to_segmentation = feature_better and segmentation_better_agnostic and segmentation_better_wrong
    if not feature_better:
        diagnosis = "World Predictor mechanism remains the leading issue"
    elif feature_better and not segmentation_better_agnostic:
        diagnosis = "feature-to-segmentation task relevance is the leading issue"
    elif world_fp_high and best_reliability == "world_pairwise":
        diagnosis = "reliability mapping is the leading issue"
    else:
        diagnosis = "evidence is temporarily unable to distinguish the mechanisms"

    summary = {
        "stage": "V8.2 World Predictor mechanism diagnosis",
        "checkpoint": str(checkpoint_path),
        "test_cases": test_names,
        "test_case_count": len(test_cases),
        "completed_case_count": len(case_patch_counts),
        "completed_rows": total_rows,
        "expected_rows": len(test_cases) * args.patches_per_case * len(ACTION_PROTOCOL),
        "gt_used_only_for_diagnosis": True,
        "gradient_updates": 0,
        "world_predictor_retrained": False,
        "lora_trained": False,
        "hyperparameters_tuned": False,
        "reliability_formula_modified": False,
        "action_protocol": {"gamma": 0.30, "blur": 1.5},
        "invalid_zero_norm_counts": {
            "conditioned_delta_cosine": invalid_cosine_counts["conditioned"],
            "agnostic_delta_cosine": invalid_cosine_counts["agnostic"],
            "wrong_delta_cosine": invalid_cosine_counts["wrong"],
            "magnitude_ratio_conditioned": invalid_ratio_count,
        },
        "answers": {
            "conditioned_feature_mse_better_than_agnostic": {
                "supported": feature_better,
                "conditioned_mean_mse": cond_mse,
                "agnostic_mean_mse": agn_mse,
                "margin_agnostic_minus_conditioned": agn_mse - cond_mse,
            },
            "correct_action_better_than_wrong_action": {
                "supported": wrong_better,
                "conditioned_mean_mse": cond_mse,
                "wrong_type_mean_mse": wrong_mse,
                "margin_wrong_minus_conditioned": wrong_mse - cond_mse,
            },
            "conditioned_delta_cosine_better_than_agnostic": {
                "supported": cosine_better,
                "conditioned_mean_cosine": cond_cos,
                "agnostic_mean_cosine": agn_cos,
            },
            "gamma_conditioned_feature_prediction": {
                "conditioned_better_than_agnostic": gamma_cond["mean_conditioned_state_mse"] < gamma_cond["mean_agnostic_state_mse"],
                "conditioned_better_than_wrong": gamma_cond["mean_conditioned_state_mse"] < gamma_cond["mean_wrong_state_mse"],
                "conditioned_mean_mse": gamma_cond["mean_conditioned_state_mse"],
                "agnostic_mean_mse": gamma_cond["mean_agnostic_state_mse"],
                "wrong_mean_mse": gamma_cond["mean_wrong_state_mse"],
            },
            "blur_conditioned_feature_prediction": {
                "conditioned_better_than_agnostic": blur_cond["mean_conditioned_state_mse"] < blur_cond["mean_agnostic_state_mse"],
                "conditioned_better_than_wrong": blur_cond["mean_conditioned_state_mse"] < blur_cond["mean_wrong_state_mse"],
                "conditioned_mean_mse": blur_cond["mean_conditioned_state_mse"],
                "agnostic_mean_mse": blur_cond["mean_agnostic_state_mse"],
                "wrong_mean_mse": blur_cond["mean_wrong_state_mse"],
            },
            "conditioned_segmentation_better_than_agnostic": {
                "supported": segmentation_better_agnostic,
                "conditioned_mean_dice": cond_dice,
                "agnostic_mean_dice": agn_dice,
                "gain": cond_dice - agn_dice,
            },
            "conditioned_segmentation_better_than_wrong": {
                "supported": segmentation_better_wrong,
                "conditioned_mean_dice": cond_dice,
                "wrong_mean_dice": wrong_dice,
                "gain": cond_dice - wrong_dice,
            },
            "feature_prediction_improvement_translates_to_segmentation": {
                "supported": feature_to_segmentation,
                "feature_better": feature_better,
                "segmentation_better_than_agnostic": segmentation_better_agnostic,
                "segmentation_better_than_wrong": segmentation_better_wrong,
            },
            "best_reliability_predictor_of_pseudo_label_correctness": {
                "source": best_reliability,
                "overall_auroc": {
                    source: None if overall_reliability[source] is None else overall_reliability[source]["auroc"]
                    for source in RELIABILITY_SOURCES
                },
                "overall_auprc": {
                    source: None if overall_reliability[source] is None else overall_reliability[source]["auprc"]
                    for source in RELIABILITY_SOURCES
                },
                "overall_spearman": {
                    source: None if overall_reliability[source] is None else overall_reliability[source]["spearman"]
                    for source in RELIABILITY_SOURCES
                },
            },
            "world_reliability_overweights_fp": {
                "supported": world_fp_high,
                "world_pairwise_fp_mean": world_fp_mean,
                "world_pairwise_tp_mean": world_tp_mean,
                "fp_minus_tp": None if world_fp_mean is None or world_tp_mean is None else world_fp_mean - world_tp_mean,
            },
            "mechanism_diagnosis": diagnosis,
        },
        "outputs": {
            "feature_dynamics": str(output_dir / "feature_dynamics.csv"),
            "segmentation_relevance": str(output_dir / "segmentation_relevance.csv"),
            "reliability_voxel_summary": str(output_dir / "reliability_voxel_summary.csv"),
            "reliability_deciles": str(output_dir / "reliability_deciles.csv"),
            "case_action_summary": str(output_dir / "case_action_summary.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_csv(output_dir / "feature_dynamics.csv", feature_rows)
    write_csv(output_dir / "segmentation_relevance.csv", segmentation_rows)
    write_csv(output_dir / "reliability_voxel_summary.csv", voxel_rows)
    write_csv(output_dir / "reliability_deciles.csv", decile_rows)
    write_csv(output_dir / "case_action_summary.csv", case_action_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["answers"], indent=2))


if __name__ == "__main__":
    run(parse_args())
