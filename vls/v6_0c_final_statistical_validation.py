from __future__ import annotations

import argparse
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
    padded_visual_action_and_slicers,
    prepare_functional_seg_head,
    resolve_device,
    select_patch_slicers,
    state_to_intermediate_prediction,
    visual_action,
)
from vls.v6_0_imagined_world_reliability import (
    SOURCE_PROMPT,
    TARGET_PROMPT,
    VARIANT_MEMBERS,
    flatten_prompt_embedding,
    load_world_model,
    pad_label_like_image,
    uncertainty_maps,
)
from vls.voxtell_states import VoxTellStateInterface


UNCERTAINTY_METRICS = ["variance", "pairwise_abs"]
ALL_SCORE_METRICS = ["confidence_error", *UNCERTAINTY_METRICS]
SCOPES = ["all_voxels", "foreground_union", "predicted_positive", "predicted_negative"]
GPU_TIE_AWARE_MIN_VOXELS = 10_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6.0c final uncertainty/confidence complementarity validation.")
    paths = ProjectPaths()
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v6_0c_final_statistical_validation")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
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


def safe_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    device: torch.device | None = None,
    force_tie_aware: bool = False,
) -> tuple[float | None, float | None]:
    """High-score-positive ROC-AUC/AP with score-tie grouping.

    The CUDA path sorts descending, groups equal scores, and evaluates ROC/AP at
    group ends, matching sklearn's threshold semantics without arbitrary tie order.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if labels.size == 0 or labels.min() == labels.max():
        return None, None
    if not force_tie_aware and scores.size <= GPU_TIE_AWARE_MIN_VOXELS:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
        except ImportError:
            pass
    if device is not None and device.type == "cuda":
        score = torch.from_numpy(scores).to(device=device, dtype=torch.float32)
        label = torch.from_numpy(labels).to(device=device, dtype=torch.float32)
        order = torch.argsort(score, descending=True, stable=True)
        ordered_score = score.index_select(0, order)
        ordered_label = label.index_select(0, order)
        boundaries = torch.nonzero(ordered_score[1:] != ordered_score[:-1], as_tuple=False).flatten() + 1
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), boundaries])
        ends = torch.cat([boundaries, torch.tensor([ordered_score.numel()], device=device)])
        cumulative_tp_voxel = torch.cumsum(ordered_label, dim=0)
        group_tp = cumulative_tp_voxel[ends - 1] - torch.cat([
            torch.zeros(1, device=device, dtype=torch.float32), cumulative_tp_voxel[ends[:-1] - 1],
        ])
        group_count = ends - starts
        group_count_f = group_count.to(torch.float32)
        group_fp = group_count_f - group_tp
        total_positive = group_tp.sum()
        total_negative = group_fp.sum()
        cumulative_fp = torch.cumsum(group_fp, dim=0)
        fp_before = cumulative_fp - group_fp
        negatives_after = total_negative - cumulative_fp
        roc_numerator = (group_tp * (negatives_after + 0.5 * group_fp)).sum()
        roc = roc_numerator / (total_positive * total_negative).clamp_min(1.0)
        cumulative_tp = torch.cumsum(group_tp, dim=0)
        cumulative_count = torch.cumsum(group_count_f, dim=0)
        average_precision = ((cumulative_tp / cumulative_count) * group_tp).sum() / total_positive.clamp_min(1.0)
        values = (float(roc.cpu()), float(average_precision.cpu()))
        del score, label, order, ordered_score, ordered_label, boundaries, starts, ends
        del cumulative_tp_voxel, group_tp, group_count, group_count_f, group_fp
        del total_positive, total_negative, cumulative_fp, fp_before, negatives_after
        del roc_numerator, cumulative_tp, cumulative_count, roc, average_precision
        torch.cuda.empty_cache()
        return values
    try:
        from scipy.stats import rankdata

        positives = float(labels.sum())
        negatives = float(labels.size - positives)
        ranks = rankdata(scores, method="average")
        rank_sum = float(ranks[labels == 1].sum())
        roc = (rank_sum - positives * (positives + 1.0) / 2.0) / max(positives * negatives, 1.0)
        order = np.argsort(-scores, kind="mergesort")
        ordered_scores = scores[order]
        ordered_labels = labels[order]
        ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1, ordered_scores.size]
        starts = np.r_[0, ends[:-1]]
        group_tp = np.add.reduceat(ordered_labels, starts)
        group_count = ends - starts
        ap = float((np.cumsum(group_tp) / np.cumsum(group_count) * group_tp).sum() / max(positives, 1.0))
        return roc, ap
    except ImportError:
        return None, None


def validate_safe_auc(device: torch.device) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(2026)
    tests = [
        (rng.normal(size=257), rng.integers(0, 2, size=257, dtype=np.int8)),
        (rng.integers(0, 7, size=4096).astype(np.float64), rng.integers(0, 2, size=4096, dtype=np.int8)),
    ]
    results = []
    for scores, labels in tests:
        expected = (float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores)))
        actual = safe_auc(scores, labels, device=device, force_tie_aware=True)
        if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
            raise AssertionError(f"tie-aware safe_auc mismatch: actual={actual}, expected={expected}")
        results.append({"size": int(scores.size), "expected": expected, "actual": actual})
    return {"passed": True, "tests": results}


def resize_logits(logits: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(logits.shape[-3:]) == shape:
        return logits.float()
    return F.interpolate(logits.float(), size=shape, mode="trilinear", align_corners=False)


def scope_mask(region: np.ndarray, scope: str, high_confidence: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if scope == "all_voxels":
        include = np.ones(region.shape, dtype=bool)
        positive = np.isin(region, [2, 3])
    elif scope == "foreground_union":
        include = region != 0
        positive = np.isin(region, [2, 3])
    elif scope == "predicted_positive":
        include = np.isin(region, [1, 2])
        positive = region == 2
    else:
        include = np.isin(region, [0, 3])
        positive = region == 3
    if high_confidence is not None:
        include &= high_confidence
    return include, positive.astype(np.int8)


def confidence_score(probability: np.ndarray, scope: str) -> np.ndarray:
    if scope in {"all_voxels", "foreground_union"}:
        return 1.0 - np.abs(2.0 * probability - 1.0)
    if scope == "predicted_positive":
        return 1.0 - probability
    return probability


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(a, b).statistic)
    except ImportError:
        return None


@torch.inference_mode()
def calculate_patch(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    case: Any,
    image: np.ndarray,
    label: np.ndarray,
    embedding: torch.Tensor,
    patch_index: int,
    patch_kind: str,
    slicer: tuple,
    original_padded: torch.Tensor,
    label_padded: torch.Tensor,
    selected_stage: str,
    prediction_threshold: float,
    high_confidence_threshold: float,
    label_value: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
    original_result = interface.forward_with_states(original_patch, embedding)
    source_state = original_result["decoder_states"][selected_stage][:, 0].detach().float().to(device)
    source_final_probability = torch.sigmoid(original_result["final_prediction"][:, 0:1].detach().float().to(device))
    text_delta = (flatten_prompt_embedding(embedding, 1) - flatten_prompt_embedding(embedding, 0)).to(device)
    imagined_states = {
        "original": source_state,
        "gamma_0.30": world_model(source_state, action=visual_action("gamma", 0.30, device)),
        "blur_1.5": world_model(source_state, action=visual_action("blur", 1.5, device)),
        "language": world_model(source_state, text_delta=text_delta[None]),
    }
    final_shape = tuple(int(size) for size in source_final_probability.shape[-3:])
    probabilities = {
        name: torch.sigmoid(resize_logits(
            state_to_intermediate_prediction(interface, selected_stage, state), final_shape,
        ))
        for name, state in imagined_states.items()
    }
    gt = (label_padded[slicer][None].to(device) == label_value).float()
    if tuple(gt.shape[-3:]) != final_shape:
        gt = F.interpolate(gt, size=final_shape, mode="nearest")
    prediction = source_final_probability > prediction_threshold
    target = gt > 0.5
    tp = prediction & target
    tn = (~prediction) & (~target)
    fp = prediction & (~target)
    fn = (~prediction) & target
    region = torch.zeros_like(prediction, dtype=torch.uint8)
    region[tp] = 1
    region[fp] = 2
    region[fn] = 3
    probability_np = source_final_probability.flatten().detach().cpu().numpy().astype(np.float32)
    region_np = region.flatten().detach().cpu().numpy()
    confidence_np = np.maximum(probability_np, 1.0 - probability_np)
    high_confidence = confidence_np >= high_confidence_threshold
    both_empty = bool(prediction.sum() == 0 and target.sum() == 0)
    dice = 1.0 if both_empty else float((2.0 * tp.sum() / (prediction.sum() + target.sum()).clamp_min(1.0)).detach().cpu())
    payload = {
        "case": case.case,
        "patch_index": patch_index,
        "patch_kind": patch_kind,
        "both_empty": int(both_empty),
        "original_dice": dice,
        "tp_voxels": int(tp.sum().detach().cpu()),
        "tn_voxels": int(tn.sum().detach().cpu()),
        "fp_voxels": int(fp.sum().detach().cpu()),
        "fn_voxels": int(fn.sum().detach().cpu()),
    }
    maps_by_variant = {variant: uncertainty_maps([probabilities[name] for name in members]) for variant, members in VARIANT_MEMBERS.items()}
    scores = {}
    for variant, maps in maps_by_variant.items():
        scores[variant] = {
            "variance": maps["variance"].flatten().detach().cpu().numpy().astype(np.float32),
            "pairwise_abs": maps["pairwise_abs"].flatten().detach().cpu().numpy().astype(np.float32),
        }
    scores["confidence"] = {scope: confidence_score(probability_np, scope) for scope in SCOPES}
    scores["confidence"]["high_confidence"] = high_confidence
    scores["region"] = region_np
    scores["high_confidence"] = high_confidence
    return payload, scores


def append_auc(
    rows: list[dict[str, Any]],
    score_payloads: list[dict[str, Any]],
    level: str,
    level_value: str | None,
    device: torch.device,
    high_confidence: bool = False,
) -> list[dict[str, Any]]:
    selected = [index for index, row in enumerate(rows) if level == "global" or row["case"] == level_value]
    result = []
    for variant in ["visual_only", "language_only", "visual_language"]:
        for metric in UNCERTAINTY_METRICS:
            for scope in ["predicted_positive", "predicted_negative"] if high_confidence else SCOPES:
                score_parts, label_parts = [], []
                for index in selected:
                    region = score_payloads[index]["region"]
                    high_mask = score_payloads[index]["high_confidence"] if high_confidence else None
                    include, labels = scope_mask(region, scope, high_mask)
                    if include.any():
                        score_parts.append(score_payloads[index][variant][metric][include])
                        label_parts.append(labels[include])
                scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float32)
                labels = np.concatenate(label_parts) if label_parts else np.empty(0, dtype=np.int8)
                roc, pr = safe_auc(scores, labels, device=device)
                result.append({
                    "level": level,
                    "case": level_value,
                    "subset": "high_confidence" if high_confidence else "all",
                    "uncertainty_variant": variant,
                    "score_metric": metric,
                    "scope": scope,
                    "num_voxels": int(labels.size),
                    "positive_voxels": int(labels.sum()),
                    "error_prevalence": float(labels.mean()) if labels.size else None,
                    "pr_random_baseline": float(labels.mean()) if labels.size else None,
                    "roc_auc": roc,
                    "pr_auc": pr,
                })
    for scope in SCOPES:
        score_parts, label_parts = [], []
        for index in selected:
            region = score_payloads[index]["region"]
            high_mask = score_payloads[index]["high_confidence"] if high_confidence else None
            include, labels = scope_mask(region, scope, high_mask)
            confidence = score_payloads[index]["confidence"][scope]
            if include.any():
                score_parts.append(confidence[include])
                label_parts.append(labels[include])
        scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float32)
        labels = np.concatenate(label_parts) if label_parts else np.empty(0, dtype=np.int8)
        roc, pr = safe_auc(scores, labels, device=device)
        result.append({
            "level": level,
            "case": level_value,
            "subset": "high_confidence" if high_confidence else "all",
            "uncertainty_variant": "confidence_baseline",
            "score_metric": "confidence_error",
            "scope": scope,
            "num_voxels": int(labels.size),
            "positive_voxels": int(labels.sum()),
            "error_prevalence": float(labels.mean()) if labels.size else None,
            "pr_random_baseline": float(labels.mean()) if labels.size else None,
            "roc_auc": roc,
            "pr_auc": pr,
        })
    return result


def complementarity_rows(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    level: str,
    level_value: str | None,
) -> list[dict[str, Any]]:
    selected = [i for i, row in enumerate(rows) if level == "global" or row["case"] == level_value]
    result = []
    for variant in ["visual_only", "language_only", "visual_language"]:
        for scope in SCOPES:
            pair_parts, conf_parts, labels_parts = [], [], []
            for i in selected:
                region = payloads[i]["region"]
                include, labels = scope_mask(region, scope)
                if include.any():
                    pair_parts.append(payloads[i][variant]["pairwise_abs"][include])
                    conf_parts.append(payloads[i]["confidence"][scope][include])
                    labels_parts.append(labels[include])
            pair = np.concatenate(pair_parts) if pair_parts else np.empty(0)
            conf = np.concatenate(conf_parts) if conf_parts else np.empty(0)
            labels = np.concatenate(labels_parts) if labels_parts else np.empty(0, dtype=np.int8)
            sampled = False
            if pair.size > 2_000_000:
                rng = np.random.default_rng(2026)
                sample_index = rng.choice(pair.size, size=2_000_000, replace=False)
                pair, conf, labels = pair[sample_index], conf[sample_index], labels[sample_index]
                sampled = True
            from scipy.stats import rankdata

            pair_rank = rankdata(pair, method="average") / max(pair.size, 1)
            conf_rank = rankdata(conf, method="average") / max(conf.size, 1)
            average_rank = 0.5 * (pair_rank + conf_rank)
            roc, pr = safe_auc(average_rank, labels, device=None)
            result.append({
                "level": level,
                "case": level_value,
                "uncertainty_variant": variant,
                "scope": scope,
                "spearman_confidence_vs_pairwise": spearman_correlation(conf, pair),
                "percentile_rank_average_roc_auc": roc,
                "percentile_rank_average_pr_auc": pr,
                "num_voxels": int(labels.size),
                "sampled_for_complementarity": sampled,
            })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_safe_auc(device)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    interface = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
    cases = iter_cases(paths, split="test", limit=args.val_cases)
    prepare_functional_seg_head(interface, args.selected_stage)
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(checkpoint_path, int(checkpoint["state_dict"]["output_projection.bias"].shape[0]), device, args.hidden_channels)
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for case in cases:
        image, label, _ = read_image_and_label(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface, image, [SOURCE_PROMPT, TARGET_PROMPT], args.patches_per_case,
            args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
        )
        label_padded = pad_label_like_image(interface, label)
        embedding = interface.embed_text_prompts([SOURCE_PROMPT, TARGET_PROMPT])
        for patch_index, slicer in enumerate(slicers):
            row, payload = calculate_patch(
                interface, world_model, case, image, label, embedding, patch_index, patch_kinds[patch_index], slicer,
                original_padded, label_padded, args.selected_stage, args.prediction_threshold,
                args.high_confidence_threshold, args.label_value, device,
            )
            rows.append(row)
            payloads.append(payload)
    # The checkpoint and VoxTell backbone are not needed for CPU-resident AUC input.
    # Release them before sorting the large pooled voxel arrays on CUDA.
    del interface, world_model, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    auc_rows = append_auc(rows, payloads, "global", None, device)
    high_auc_rows = append_auc(rows, payloads, "global", None, device, high_confidence=True)
    for case in [item.case for item in cases]:
        auc_rows.extend(append_auc(rows, payloads, "case", case, device))
    # Complementarity is reported on the pooled validation population. Per-case
    # AUCs above retain the requested case breakdown without repeating huge rank
    # arrays for the optional diagnostic.
    complementarity = complementarity_rows(rows, payloads, "global", None)
    empty_rows = []
    for case in [item.case for item in cases]:
        case_rows = [row for row in rows if row["case"] == case]
        empty_rows.append({
            "case": case,
            "patches": len(case_rows),
            "both_empty_patches": sum(int(row["both_empty"]) for row in case_rows),
            "both_empty_voxels": sum(int(row["tn_voxels"]) for row in case_rows if row["both_empty"]),
        })
    write_csv(output_dir / "v6_0c_patch.csv", rows)
    write_csv(output_dir / "v6_0c_auc.csv", auc_rows + high_auc_rows)
    write_csv(output_dir / "v6_0c_complementarity.csv", complementarity)
    write_csv(output_dir / "v6_0c_both_empty.csv", empty_rows)
    uncertainty_global = [
        row for row in auc_rows
        if row["level"] == "global" and row["score_metric"] != "confidence_error"
    ]
    summary = {
        "args": vars(args),
        "world_checkpoint": str(checkpoint_path),
        "selected_stage": args.selected_stage,
        "cases": [case.case for case in cases],
        "safe_auc_validation": validation,
        "num_patch_rows": len(rows),
        "auc_csv": str(output_dir / "v6_0c_auc.csv"),
        "complementarity_csv": str(output_dir / "v6_0c_complementarity.csv"),
        "both_empty_csv": str(output_dir / "v6_0c_both_empty.csv"),
        "global_uncertainty_comparison": uncertainty_global,
        "scope_note": "V6.0c final statistical/complementarity validation only; no model, action, ensemble, training, threshold selection, fusion, pseudo-label, or SFDA changes.",
    }
    summary_path = output_dir / "v6_0c_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "summary_path": str(summary_path),
        "safe_auc_validation": validation,
        "global_uncertainty_comparison": uncertainty_global,
    }, indent=2))


if __name__ == "__main__":
    run(parse_args())
