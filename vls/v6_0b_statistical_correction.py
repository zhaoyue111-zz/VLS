from __future__ import annotations

import argparse
import csv
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
    resize_logits,
    uncertainty_maps,
)
from vls.voxtell_states import VoxTellStateInterface


UNCERTAINTY_METRICS = ["variance", "std", "pairwise_abs"]
SCORE_METRICS = ["mean_probability", *UNCERTAINTY_METRICS]
SCOPES = {
    "all_voxels": {"regions": ["TP", "TN", "FP", "FN"], "exclude_both_empty": False},
    "foreground_union": {"regions": ["TP", "FP", "FN"], "exclude_both_empty": True},
    "predicted_positive": {"regions": ["TP", "FP"], "exclude_both_empty": True},
    "predicted_negative": {"regions": ["TN", "FN"], "exclude_both_empty": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6.0b corrected imagined-world uncertainty statistics.")
    paths = ProjectPaths()
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v6_0b_statistical_correction")
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


def safe_auc(scores: np.ndarray, labels: np.ndarray, device: torch.device | None = None) -> tuple[float | None, float | None]:
    """Return high-score-positive ROC-AUC and average precision."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if labels.size == 0 or labels.min() == labels.max():
        return None, None
    # sklearn is preferred for manageable scopes so ties are handled exactly.
    if scores.size <= 20_000_000:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
        except ImportError:
            pass
    # For very large pooled all-voxel scopes, use a descending high-score-positive
    # implementation on CUDA to avoid multi-minute CPU sorting.
    if device is not None and device.type == "cuda":
        score_tensor = torch.from_numpy(scores).to(device=device, dtype=torch.float32)
        label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.float64)
        order = torch.argsort(score_tensor, descending=True)
        sorted_labels = label_tensor.index_select(0, order)
        positives = sorted_labels.sum()
        negatives = sorted_labels.numel() - positives
        ranks = torch.arange(1, sorted_labels.numel() + 1, device=device, dtype=torch.float64)
        rank_sum = (ranks * sorted_labels).sum()
        ascending_rank_sum = positives * (positives + 1.0) / 2.0
        roc = (positives * negatives - (rank_sum - ascending_rank_sum)) / (positives * negatives).clamp_min(1.0)
        cumulative = torch.cumsum(sorted_labels, dim=0)
        precision = cumulative / ranks
        pr = (precision * sorted_labels).sum() / positives.clamp_min(1.0)
        return float(roc.cpu()), float(pr.cpu())
    try:
        from scipy.stats import rankdata

        positives = float(labels.sum())
        negatives = float(labels.size - positives)
        ascending_ranks = rankdata(scores, method="average")
        rank_sum = float(ascending_ranks[labels == 1].sum())
        roc = (rank_sum - positives * (positives + 1.0) / 2.0) / max(positives * negatives, 1.0)
        order = np.argsort(-scores, kind="mergesort")
        ordered_scores = scores[order]
        ordered_labels = labels[order]
        group_end = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1, ordered_scores.size]
        group_start = np.r_[0, group_end[:-1]]
        group_tp = np.add.reduceat(ordered_labels, group_start)
        cumulative_tp = np.cumsum(group_tp)
        cumulative_count = group_end
        pr = float((cumulative_tp / cumulative_count * group_tp).sum() / max(positives, 1.0))
        return roc, pr
    except ImportError:
        pass
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
    except ImportError:
        # Descending score order is required because high uncertainty is the positive class.
        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        positives = float(sorted_labels.sum())
        negatives = float(sorted_labels.size - positives)
        ranks = np.arange(1, sorted_labels.size + 1, dtype=np.float64)
        rank_sum = float(ranks[sorted_labels == 1].sum())
        ascending_rank_sum = positives * (positives + 1.0) / 2.0
        roc = (positives * negatives - (rank_sum - ascending_rank_sum)) / max(positives * negatives, 1.0)
        cumulative = np.cumsum(sorted_labels)
        precision = cumulative / ranks
        pr = float((precision * sorted_labels).sum() / max(positives, 1.0))
        return roc, pr


def resize_stage_logits(logits: torch.Tensor, final_shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(logits.shape[-3:]) == final_shape:
        return logits.float()
    return F.interpolate(logits.float(), size=final_shape, mode="trilinear", align_corners=False)


def evaluate_case(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    case: Any,
    selected_stage: str,
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
    prediction_threshold: float,
    label_value: int,
) -> list[dict[str, Any]]:
    image, label, _ = read_image_and_label(case)
    original_padded, slicers, patch_kinds = select_patch_slicers(
        interface, image, [SOURCE_PROMPT, TARGET_PROMPT], patches_per_case,
        foreground_patches_per_case, foreground_candidate_patches, foreground_threshold,
    )
    gamma_padded, _ = padded_visual_action_and_slicers(interface.predictor, image, "gamma", 0.30)
    blur_padded, _ = padded_visual_action_and_slicers(interface.predictor, image, "blur", 1.5)
    label_padded = pad_label_like_image(interface, label)
    if tuple(label_padded.shape[1:]) != tuple(original_padded.shape[1:]):
        raise ValueError(f"GT/image preprocessing shape mismatch: {label_padded.shape} vs {original_padded.shape}")
    embedding = interface.embed_text_prompts([SOURCE_PROMPT, TARGET_PROMPT])
    device = next(world_model.parameters()).device
    text_delta = (flatten_prompt_embedding(embedding, 1) - flatten_prompt_embedding(embedding, 0)).to(device)
    gamma_action = visual_action("gamma", 0.30, device)
    blur_action = visual_action("blur", 1.5, device)
    rows: list[dict[str, Any]] = []

    for patch_index, slicer in enumerate(slicers):
        original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
        original_result = interface.forward_with_states(original_patch, embedding)
        source_state = original_result["decoder_states"][selected_stage][:, 0].detach().float().to(device)
        source_final_logits = original_result["final_prediction"][:, 0:1].detach().float().to(device)
        gamma_state = world_model(source_state, action=gamma_action)
        blur_state = world_model(source_state, action=blur_action)
        language_state = world_model(source_state, text_delta=text_delta[None])
        final_shape = tuple(int(size) for size in source_final_logits.shape[-3:])
        stage_logits = {
            "original": resize_stage_logits(state_to_intermediate_prediction(interface, selected_stage, source_state), final_shape),
            "gamma_0.30": resize_stage_logits(state_to_intermediate_prediction(interface, selected_stage, gamma_state), final_shape),
            "blur_1.5": resize_stage_logits(state_to_intermediate_prediction(interface, selected_stage, blur_state), final_shape),
            "language": resize_stage_logits(state_to_intermediate_prediction(interface, selected_stage, language_state), final_shape),
        }
        probabilities = {name: torch.sigmoid(logits) for name, logits in stage_logits.items()}
        uncertainty_by_variant = {
            variant: uncertainty_maps([probabilities[name] for name in members])
            for variant, members in VARIANT_MEMBERS.items()
        }
        scores_by_variant = {
            variant: {metric: uncertainty[metric] for metric in UNCERTAINTY_METRICS}
            | {"mean_probability": uncertainty["mean"]}
            for variant, uncertainty in uncertainty_by_variant.items()
        }
        original_probability = torch.sigmoid(source_final_logits)
        gt = (label_padded[slicer][None].to(device) == label_value).float()
        if tuple(gt.shape[-3:]) != final_shape:
            gt = F.interpolate(gt, size=final_shape, mode="nearest")
        prediction = original_probability > prediction_threshold
        target = gt > 0.5
        tp = prediction & target
        tn = (~prediction) & (~target)
        fp = prediction & (~target)
        fn = (~prediction) & target
        region_codes = torch.zeros_like(prediction, dtype=torch.uint8)
        region_codes[tp] = 1
        region_codes[fp] = 2
        region_codes[fn] = 3
        region_flat = region_codes.flatten().detach().cpu().numpy()
        both_empty = bool(prediction.sum() == 0 and target.sum() == 0)
        for variant, metric_maps in scores_by_variant.items():
            row: dict[str, Any] = {
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "uncertainty_variant": variant,
                "ensemble_members": "+".join(VARIANT_MEMBERS[variant]),
                "both_empty": int(both_empty),
                "tp_voxels": int(tp.sum().detach().cpu()),
                "tn_voxels": int(tn.sum().detach().cpu()),
                "fp_voxels": int(fp.sum().detach().cpu()),
                "fn_voxels": int(fn.sum().detach().cpu()),
                "original_dice": float((2.0 * tp.sum() / (prediction.sum() + target.sum()).clamp_min(1.0)).detach().cpu()),
            }
            for metric, values in metric_maps.items():
                flat = values.flatten().detach().cpu().numpy().astype(np.float32, copy=False)
                row[f"{metric}_global_mean"] = float(flat.mean())
                for code, name in [(1, "TP"), (0, "TN"), (2, "FP"), (3, "FN")]:
                    selected = flat[region_flat == code]
                    row[f"{metric}_{name}_sum"] = float(selected.sum())
                    row[f"{metric}_{name}_count"] = int(selected.size)
            rows.append(row)
    return rows


def pooled_region_stats(rows: list[dict[str, Any]], level_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[key_name]) for key_name in level_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group_rows in sorted(groups.items()):
        for variant in sorted(VARIANT_MEMBERS):
            variant_rows = [row for row in group_rows if row["uncertainty_variant"] == variant]
            for metric in SCORE_METRICS:
                for region in ["TP", "TN", "FP", "FN"]:
                    total_sum = sum(float(row[f"{metric}_{region}_sum"]) for row in variant_rows)
                    total_count = sum(int(row[f"{metric}_{region}_count"]) for row in variant_rows)
                    item: dict[str, Any] = {
                        "uncertainty_variant": variant,
                        "score_metric": metric,
                        "region": region,
                        "voxel_count": total_count,
                        "voxel_weighted_mean": total_sum / total_count if total_count else None,
                    }
                    for key_name, value in zip(level_keys, key, strict=True):
                        item[key_name] = value
                    output.append(item)
    return output


def selected_scope(region_codes: np.ndarray, patch_both_empty: bool, scope: str) -> tuple[np.ndarray, np.ndarray]:
    config = SCOPES[scope]
    if config["exclude_both_empty"] and patch_both_empty:
        return np.zeros(region_codes.shape, dtype=bool), np.zeros(region_codes.shape, dtype=np.int8)
    included = np.isin(region_codes, [0, 1, 2, 3])
    if scope == "foreground_union":
        included &= region_codes != 0
        labels = np.isin(region_codes, [2, 3]).astype(np.int8)
    elif scope == "predicted_positive":
        included &= np.isin(region_codes, [1, 2])
        labels = (region_codes == 2).astype(np.int8)
    elif scope == "predicted_negative":
        included &= np.isin(region_codes, [0, 3])
        labels = (region_codes == 3).astype(np.int8)
    else:
        labels = np.isin(region_codes, [2, 3]).astype(np.int8)
    return included, labels


def build_auc_rows(
    rows: list[dict[str, Any]],
    score_arrays: dict[tuple[str, int], dict[str, np.ndarray]],
    level_name: str,
    level_value: str,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    result = []
    selected_rows = [row for row in rows if row[level_name] == level_value] if level_name != "global" else rows
    selected_indices = {id(row) for row in selected_rows}
    for variant in sorted(VARIANT_MEMBERS):
        for metric in SCORE_METRICS:
            for scope, config in SCOPES.items():
                scores_parts: list[np.ndarray] = []
                labels_parts: list[np.ndarray] = []
                for index, row in enumerate(rows):
                    if id(row) not in selected_indices or row["uncertainty_variant"] != variant:
                        continue
                    region_codes = score_arrays[(variant, index)]["region_codes"]
                    included, labels = selected_scope(region_codes, bool(row["both_empty"]), scope)
                    if included.any():
                        scores_parts.append(score_arrays[(variant, index)][metric][included])
                        labels_parts.append(labels[included])
                scores = np.concatenate(scores_parts) if scores_parts else np.empty(0, dtype=np.float32)
                labels = np.concatenate(labels_parts) if labels_parts else np.empty(0, dtype=np.int8)
                roc_auc, pr_auc = safe_auc(scores, labels, device=device)
                positives = int(labels.sum())
                total = int(labels.size)
                result.append({
                    "level": "global" if level_name == "global" else "case",
                    "case": None if level_name == "global" else level_value,
                    "uncertainty_variant": variant,
                    "score_metric": metric,
                    "scope": scope,
                    "num_voxels": total,
                    "positive_voxels": positives,
                    "error_prevalence": positives / total if total else None,
                    "pr_random_baseline": positives / total if total else None,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "excluded_both_empty": config["exclude_both_empty"],
                })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
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
    interface = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
    cases = iter_cases(paths, split="test", limit=args.val_cases)
    prepare_functional_seg_head(interface, args.selected_stage)
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(
        checkpoint_path,
        int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
        device,
        args.hidden_channels,
    )
    rows: list[dict[str, Any]] = []
    score_arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for case in cases:
        image, label, _ = read_image_and_label(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface, image, [SOURCE_PROMPT, TARGET_PROMPT], args.patches_per_case,
            args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
        )
        gamma_padded, _ = padded_visual_action_and_slicers(interface.predictor, image, "gamma", 0.30)
        blur_padded, _ = padded_visual_action_and_slicers(interface.predictor, image, "blur", 1.5)
        label_padded = pad_label_like_image(interface, label)
        embedding = interface.embed_text_prompts([SOURCE_PROMPT, TARGET_PROMPT])
        text_delta = (flatten_prompt_embedding(embedding, 1) - flatten_prompt_embedding(embedding, 0)).to(device)
        gamma_action = visual_action("gamma", 0.30, device)
        blur_action = visual_action("blur", 1.5, device)
        for patch_index, slicer in enumerate(slicers):
            original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            result = interface.forward_with_states(original_patch, embedding)
            source_state = result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
            source_final_logits = result["final_prediction"][:, 0:1].detach().float().to(device)
            states = {
                "original": source_state,
                "gamma_0.30": world_model(source_state, action=gamma_action),
                "blur_1.5": world_model(source_state, action=blur_action),
                "language": world_model(source_state, text_delta=text_delta[None]),
            }
            final_shape = tuple(int(size) for size in source_final_logits.shape[-3:])
            probabilities = {
                name: torch.sigmoid(resize_stage_logits(state_to_intermediate_prediction(interface, args.selected_stage, state), final_shape))
                for name, state in states.items()
            }
            gt = (label_padded[slicer][None].to(device) == args.label_value).float()
            if tuple(gt.shape[-3:]) != final_shape:
                gt = F.interpolate(gt, size=final_shape, mode="nearest")
            prediction = torch.sigmoid(source_final_logits) > args.prediction_threshold
            target = gt > 0.5
            tp = prediction & target
            tn = (~prediction) & (~target)
            fp = prediction & (~target)
            fn = (~prediction) & target
            region_codes = torch.zeros_like(prediction, dtype=torch.uint8)
            region_codes[tp] = 1
            region_codes[fp] = 2
            region_codes[fn] = 3
            region_np = region_codes.flatten().detach().cpu().numpy()
            patch_both_empty = bool(prediction.sum() == 0 and target.sum() == 0)
            for variant, members in VARIANT_MEMBERS.items():
                maps = uncertainty_maps([probabilities[name] for name in members])
                metric_maps = {"mean_probability": maps["mean"], **{metric: maps[metric] for metric in UNCERTAINTY_METRICS}}
                payload = {metric: values.flatten().detach().cpu().numpy().astype(np.float32, copy=False) for metric, values in metric_maps.items()}
                payload["region_codes"] = region_np
                score_arrays[(variant, len(rows))] = payload
                tp_count, tn_count, fp_count, fn_count = [int(mask.sum().detach().cpu()) for mask in [tp, tn, fp, fn]]
                row = {
                    "case": case.case,
                    "global": "all",
                    "patch_index": patch_index,
                    "patch_kind": patch_kinds[patch_index],
                    "uncertainty_variant": variant,
                    "ensemble_members": "+".join(members),
                    "both_empty": int(patch_both_empty),
                    "tp_voxels": tp_count,
                    "tn_voxels": tn_count,
                    "fp_voxels": fp_count,
                    "fn_voxels": fn_count,
                    "original_dice": float((2.0 * tp.sum() / (prediction.sum() + target.sum()).clamp_min(1.0)).detach().cpu()),
                }
                for metric, values in metric_maps.items():
                    flat = payload[metric]
                    for code, name in [(1, "TP"), (0, "TN"), (2, "FP"), (3, "FN")]:
                        selected = flat[region_np == code]
                        row[f"{metric}_{name}_sum"] = float(selected.sum())
                        row[f"{metric}_{name}_count"] = int(selected.size)
                rows.append(row)
    # score_arrays keys are indexed by row order; normalize them to the index expected by build_auc_rows.
    normalized_scores = {}
    for row_index, row in enumerate(rows):
        normalized_scores[(row["uncertainty_variant"], row_index)] = score_arrays[(row["uncertainty_variant"], row_index)]
    score_arrays = normalized_scores
    region_stats = pooled_region_stats(rows, ["global"])
    region_stats.extend(pooled_region_stats(rows, ["case"]))
    auc_rows = build_auc_rows(rows, score_arrays, "global", "all", device=device)
    for case in [item.case for item in cases]:
        auc_rows.extend(build_auc_rows(rows, score_arrays, "case", case, device=device))
    both_empty_rows = []
    for case in [item.case for item in cases]:
        case_rows = [row for row in rows if row["case"] == case]
        for variant in sorted(VARIANT_MEMBERS):
            selected = [row for row in case_rows if row["uncertainty_variant"] == variant]
            both_empty_rows.append({
                "case": case,
                "uncertainty_variant": variant,
                "patches": len(selected),
                "both_empty_patches": sum(int(row["both_empty"]) for row in selected),
                "both_empty_voxels": sum(int(row["tn_voxels"]) for row in selected if row["both_empty"]),
            })
    detail_path = output_dir / "v6_0b_patch.csv"
    auc_path = output_dir / "v6_0b_auc.csv"
    region_path = output_dir / "v6_0b_region_stats.csv"
    empty_path = output_dir / "v6_0b_both_empty.csv"
    write_csv(detail_path, rows)
    write_csv(auc_path, auc_rows)
    write_csv(region_path, region_stats)
    write_csv(empty_path, both_empty_rows)
    comparison = [
        row for row in auc_rows
        if row["score_metric"] in UNCERTAINTY_METRICS and row["level"] == "global"
    ]
    summary = {
        "args": vars(args),
        "world_checkpoint": str(checkpoint_path),
        "selected_stage": args.selected_stage,
        "cases": [case.case for case in cases],
        "num_patch_rows": len(rows),
        "variants": VARIANT_MEMBERS,
        "scopes": SCOPES,
        "detail_csv": str(detail_path),
        "auc_csv": str(auc_path),
        "region_stats_csv": str(region_path),
        "both_empty_csv": str(empty_path),
        "comparison_uncertainty_only": comparison,
        "scope_note": "V6.0b correction: uncertainty AUC uses variance/std/pairwise disagreement; mean_probability is a separate baseline. Foreground-relevant scopes exclude both-empty patches; region means are voxel-weighted.",
    }
    summary_path = output_dir / "v6_0b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "summary_path": str(summary_path),
        "auc_csv": str(auc_path),
        "comparison_uncertainty_only": comparison,
    }, indent=2))


if __name__ == "__main__":
    run(parse_args())
