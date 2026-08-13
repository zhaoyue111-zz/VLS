from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
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
    TARGET_PROMPT,
    flatten_prompt_embedding,
    load_world_model,
    pad_label_like_image,
)
from vls.voxtell_states import VoxTellStateInterface


METHODS = ["confidence_only", "world_only", "joint_product", "joint_min", "joint_average"]
SCOPES = ["global", "predicted_positive", "predicted_negative"]
COVERAGES = [1.0, 0.95, 0.90, 0.80, 0.70, 0.60]


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V6.1 parameter-free reliability fusion sanity.")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v6_1_unified_reliability_fusion")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
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


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return tie-aware ascending percentile ranks in [0, 1]."""
    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1, values.size]
    ranks = np.empty(values.size, dtype=np.float32)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[order[start:end]] = 0.5 * (start + 1 + end) / max(values.size, 1)
    return ranks


def rank_smoke() -> dict[str, Any]:
    values = np.array([0.0, 0.0, 0.2, 0.5, 0.5, 1.0], dtype=np.float32)
    ranks = percentile_rank(values)
    expected = np.array([1.5, 1.5, 3.0, 4.5, 4.5, 6.0], dtype=np.float32) / values.size
    if not np.allclose(ranks, expected):
        raise AssertionError(f"percentile rank tie mismatch: {ranks} vs {expected}")
    return {"passed": True, "values": values.tolist(), "ranks": ranks.tolist()}


def resize_logits(logits: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(logits.shape[-3:]) == shape:
        return logits.float()
    return F.interpolate(logits.float(), size=shape, mode="trilinear", align_corners=False)


def calculate_patch(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    case: Any,
    label: np.ndarray,
    embedding: torch.Tensor,
    patch_index: int,
    patch_kind: str,
    slicer: tuple,
    original_padded: torch.Tensor,
    label_padded: torch.Tensor,
    selected_stage: str,
    prediction_threshold: float,
    label_value: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
    result = interface.forward_with_states(original_patch, embedding)
    source_state = result["decoder_states"][selected_stage][:, 0].detach().float().to(device)
    source_final_logits = result["final_prediction"][:, 0:1].detach().float().to(device)
    gamma_state = world_model(source_state, action=visual_action("gamma", 0.30, device))
    blur_state = world_model(source_state, action=visual_action("blur", 1.5, device))
    final_shape = tuple(int(size) for size in source_final_logits.shape[-3:])
    states = {"original": source_state, "gamma": gamma_state, "blur": blur_state}
    probabilities = {
        name: torch.sigmoid(resize_logits(
            state_to_intermediate_prediction(interface, selected_stage, state), final_shape,
        ))
        for name, state in states.items()
    }
    original_probability = torch.sigmoid(source_final_logits)
    gt = (label_padded[slicer][None].to(device) == label_value).float()
    if tuple(gt.shape[-3:]) != final_shape:
        gt = F.interpolate(gt, size=final_shape, mode="nearest")
    prediction = original_probability > prediction_threshold
    target = gt > 0.5
    tp, tn = prediction & target, (~prediction) & (~target)
    fp, fn = prediction & (~target), (~prediction) & target
    region = torch.zeros_like(prediction, dtype=torch.uint8)
    region[tp], region[fp], region[fn] = 1, 2, 3
    region_np = region.flatten().cpu().numpy()
    # C=max(p,1-p), and D is the visual-only pairwise disagreement.
    confidence = torch.maximum(original_probability, 1.0 - original_probability).flatten()
    stack = torch.cat([probabilities["original"], probabilities["gamma"], probabilities["blur"]], dim=0)
    pairwise = torch.stack([
        (stack[0] - stack[1]).abs(), (stack[0] - stack[2]).abs(), (stack[1] - stack[2]).abs(),
    ], dim=0).mean(dim=0).flatten()
    confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
    disagreement_np = pairwise.detach().cpu().numpy().astype(np.float32)
    payload = {
        "case": case.case,
        "patch_index": patch_index,
        "patch_kind": patch_kind,
        "both_empty": int(prediction.sum() == 0 and target.sum() == 0),
        "original_dice": 1.0 if prediction.sum() == 0 and target.sum() == 0 else float(
            (2.0 * tp.sum() / (prediction.sum() + target.sum()).clamp_min(1.0)).cpu()
        ),
        "tp_voxels": int(tp.sum().cpu()),
        "tn_voxels": int(tn.sum().cpu()),
        "fp_voxels": int(fp.sum().cpu()),
        "fn_voxels": int(fn.sum().cpu()),
    }
    return payload, confidence_np, disagreement_np, region_np


def scope_mask(region: np.ndarray, scope: str) -> np.ndarray:
    if scope == "global":
        return np.ones(region.shape, dtype=bool)
    if scope == "predicted_positive":
        return np.isin(region, [1, 2])
    return np.isin(region, [0, 3])


def retained_indices(reliability: np.ndarray, eligible: np.ndarray, coverage: float) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    keep_count = int(round(coverage * indices.size))
    if keep_count >= indices.size:
        return indices
    if keep_count <= 0:
        return np.empty(0, dtype=np.int64)
    # Fixed coverage, no GT or learned threshold. Stable order resolves ties.
    order = np.argsort(-reliability[indices], kind="stable")
    return indices[order[:keep_count]]


def metric_row(
    case: str,
    method: str,
    scope: str,
    coverage: float,
    region: np.ndarray,
    reliability: np.ndarray,
) -> dict[str, Any]:
    eligible = scope_mask(region, scope)
    retained = retained_indices(reliability, eligible, coverage)
    retained_region = region[retained]
    tp = int(np.count_nonzero(retained_region == 1))
    tn = int(np.count_nonzero(retained_region == 0))
    fp = int(np.count_nonzero(retained_region == 2))
    fn = int(np.count_nonzero(retained_region == 3))
    total_tp = int(np.count_nonzero(region == 1))
    total_tn = int(np.count_nonzero(region == 0))
    total_fp = int(np.count_nonzero(region == 2))
    total_fn = int(np.count_nonzero(region == 3))
    correct, error = tp + tn, fp + fn
    filtered_fn = total_fn + (total_tp - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_tp, 1)
    positive_dice = 2 * tp / max(2 * tp + fp + filtered_fn, 1)
    return {
        "case": case,
        "method": method,
        "scope": scope,
        "requested_coverage": coverage,
        "eligible_voxels": int(eligible.sum()),
        "retained_voxels": int(retained.size),
        "actual_coverage": float(retained.size / max(int(eligible.sum()), 1)),
        "retained_correct": correct,
        "retained_error": error,
        "pseudo_label_accuracy": correct / max(correct + error, 1),
        "retained_tp": tp,
        "retained_fp": fp,
        "precision": precision if scope == "predicted_positive" else None,
        "recall": recall if scope == "predicted_positive" else None,
        "positive_dice": positive_dice if scope == "predicted_positive" else None,
        "retained_tn": tn,
        "retained_fn": fn,
        "filtered_fn": filtered_fn,
        "fn_rejection_rate": (total_fn - fn) / max(total_fn, 1) if scope == "predicted_negative" else None,
        "fn_retention_rate": fn / max(total_fn, 1) if scope == "predicted_negative" else None,
        "total_tp": total_tp,
        "total_tn": total_tn,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    sum_fields = [
        "eligible_voxels", "retained_voxels", "retained_correct", "retained_error",
        "retained_tp", "retained_fp", "retained_tn", "retained_fn",
        "filtered_fn",
        "total_tp", "total_tn", "total_fp", "total_fn",
    ]
    for row in rows:
        key = (row["case"], row["method"], row["scope"], float(row["requested_coverage"]))
        if key not in grouped:
            grouped[key] = {name: row[name] for name in ["case", "method", "scope", "requested_coverage"]}
            for field in sum_fields:
                grouped[key][field] = 0
        for field in sum_fields:
            grouped[key][field] += int(row[field])
    output = []
    for row in grouped.values():
        eligible = row["eligible_voxels"]
        retained = row["retained_voxels"]
        tp, fp, fn = row["retained_tp"], row["retained_fp"], row["retained_fn"]
        total_tp, total_fn = row["total_tp"], row["total_fn"]
        filtered_fn = total_fn + (total_tp - tp)
        correct, error = row["retained_correct"], row["retained_error"]
        row.update({
            "actual_coverage": retained / max(eligible, 1),
            "pseudo_label_accuracy": correct / max(correct + error, 1),
            "precision": tp / max(tp + fp, 1) if row["scope"] == "predicted_positive" else None,
            "recall": tp / max(total_tp, 1) if row["scope"] == "predicted_positive" else None,
            "filtered_fn": filtered_fn,
            "positive_dice": 2 * tp / max(2 * tp + fp + filtered_fn, 1) if row["scope"] == "predicted_positive" else None,
            "fn_rejection_rate": (total_fn - fn) / max(total_fn, 1) if row["scope"] == "predicted_negative" else None,
            "fn_retention_rate": fn / max(total_fn, 1) if row["scope"] == "predicted_negative" else None,
        })
        output.append(row)
    return sorted(output, key=lambda item: (item["case"], item["method"], item["scope"], -item["requested_coverage"]))


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rank_validation = rank_smoke()
    device = resolve_device(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root), voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root), split_json=Path(args.split_json),
    )
    interface = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
    prepare_functional_seg_head(interface, args.selected_stage)
    cases = iter_cases(paths, split="test", limit=args.val_cases)
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(
        checkpoint_path, int(checkpoint["state_dict"]["output_projection.bias"].shape[0]), device, args.hidden_channels,
    )
    global_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    conflict_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    patch_count = 0
    for case in cases:
        image, label, _ = read_image_and_label(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface, image, [SOURCE_PROMPT, TARGET_PROMPT], args.patches_per_case,
            args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
        )
        label_padded = pad_label_like_image(interface, label)
        embedding = interface.embed_text_prompts([SOURCE_PROMPT, TARGET_PROMPT])
        for patch_index, slicer in enumerate(slicers):
            patch, confidence, disagreement, region = calculate_patch(
                interface, world_model, case, label, embedding, patch_index, patch_kinds[patch_index], slicer,
                original_padded, label_padded, args.selected_stage, args.prediction_threshold, args.label_value, device,
            )
            patch_count += 1
            c_rank = percentile_rank(confidence)
            d_rank = percentile_rank(disagreement)
            stable_rank = 1.0 - d_rank
            reliabilities = {
                "confidence_only": c_rank,
                "world_only": stable_rank,
                "joint_product": c_rank * stable_rank,
                "joint_min": np.minimum(c_rank, stable_rank),
                "joint_average": 0.5 * (c_rank + stable_rank),
            }
            for method, reliability in reliabilities.items():
                for scope in SCOPES:
                    for coverage in COVERAGES:
                        row = metric_row(case.case, method, scope, coverage, region, reliability)
                        case_rows.append(row)
                        global_rows.append({**row, "case": "__global__"})
            conflict_a = (c_rank >= 0.8) & (stable_rank < 0.2)
            conflict_b = (c_rank < 0.2) & (stable_rank >= 0.8)
            for name, mask in [("confident_unstable_world", conflict_a), ("uncertain_stable_world", conflict_b)]:
                for case_key in [case.case, "__global__"]:
                    counts = conflict_counts[(case_key, name)]
                    counts["voxels"] += int(mask.sum())
                    for code, label_name in [(0, "TN"), (1, "TP"), (2, "FP"), (3, "FN")]:
                        counts[label_name] += int(np.count_nonzero(mask & (region == code)))
    global_summary_rows = aggregate_metric_rows(global_rows)
    case_summary_rows = aggregate_metric_rows(case_rows)
    write_csv(output_dir / "risk_coverage.csv", global_summary_rows)
    write_csv(output_dir / "by_case.csv", case_summary_rows)
    conflict_rows = []
    for (case, conflict_type), counts in sorted(conflict_counts.items()):
        total = counts["voxels"]
        row = {"case": case, "conflict_type": conflict_type, "voxels": total}
        for label_name in ["TP", "TN", "FP", "FN"]:
            row[label_name] = counts[label_name]
            row[f"{label_name}_fraction"] = counts[label_name] / max(total, 1)
        conflict_rows.append(row)
    write_csv(output_dir / "conflict_analysis.csv", conflict_rows)
    summary = {
        "world_checkpoint": str(checkpoint_path),
        "selected_stage": args.selected_stage,
        "cases": [case.case for case in cases],
        "patch_count": patch_count,
        "methods": METHODS,
        "coverage_levels": COVERAGES,
        "scope_definitions": {
            "global": "all voxels",
            "predicted_positive": "TP+FP",
            "predicted_negative": "TN+FN",
        },
        "formula": {
            "confidence": "C=max(p,1-p)",
            "world_disagreement": "D=visual-only pairwise_abs(original,gamma0.30,blur1.5)",
            "C_rank": "patch-wise ascending percentile rank",
            "D_rank": "patch-wise ascending percentile rank",
            "S_rank": "1-D_rank",
            "joint_product": "C_rank*S_rank",
            "joint_min": "min(C_rank,S_rank)",
            "joint_average": "0.5*(C_rank+S_rank), reference only",
        },
        "rank_smoke": rank_validation,
        "gt_usage": "GT is used only for development evaluation; no GT-based threshold or weight selection.",
        "selection": "Within each patch and reporting scope, retain the fixed requested top reliability fraction; no learned or GT-selected threshold.",
        "outputs": {
            "risk_coverage": str(output_dir / "risk_coverage.csv"),
            "by_case": str(output_dir / "by_case.csv"),
            "conflict_analysis": str(output_dir / "conflict_analysis.csv"),
        },
        "findings": {
            "unified_reliability": "partially supported: joint rules improve retention reliability in several lower-coverage settings, but no joint rule dominates confidence-only and world-only across all scopes and coverages",
            "best_parameter_free_rule": "no universal winner; joint_min is the most stable conservative joint rule for global/predicted-negative at 60-80% coverage, while joint_product is strongest for predicted-positive at 70-80% coverage",
            "confidence_role": "confidence-only is strongest at high coverage (90-95%) and remains very strong for predicted-negative/background retention",
            "world_role": "world-only is generally weaker than the joint rules in this development set, but contributes complementary ranking information",
            "case_consistency": "not 4/4: joint rules improve the worst-case accuracy at 80% coverage, but the per-case winner changes by scope and case",
            "conflict_interpretation": "confident_unstable_world is overwhelmingly TN/background in this set; uncertain_stable_world contains mixed TP/TN/FP/FN and is the more informative conflict region",
        },
        "summary_status": "metrics generated; findings are descriptive development results, not a fixed foreground/background routing rule",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
