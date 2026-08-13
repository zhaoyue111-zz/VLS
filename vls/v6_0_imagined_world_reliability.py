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
from vls.data import binary_gt_from_label, iter_cases, read_image_and_label
from vls.v2_experiment import (
    padded_image_and_slicers,
    padded_visual_action_and_slicers,
    prepare_functional_seg_head,
    resolve_device,
    select_patch_slicers,
    state_to_intermediate_prediction,
    visual_action,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D


SOURCE_PROMPT = "liver"
TARGET_PROMPT = "the liver"
VARIANT_MEMBERS = {
    "visual_only": ["original", "gamma_0.30", "blur_1.5"],
    "language_only": ["original", "language"],
    "visual_language": ["original", "gamma_0.30", "blur_1.5", "language"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6.0 imagined-world reliability sanity.")
    paths = ProjectPaths()
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt",
    )
    parser.add_argument("--output-dir", default="outputs/v6_0_imagined_world_reliability")
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


def flatten_prompt_embedding(text_embedding: torch.Tensor, prompt_index: int) -> torch.Tensor:
    if text_embedding.ndim not in (3, 4):
        raise ValueError(f"Unexpected text embedding shape: {tuple(text_embedding.shape)}")
    return text_embedding[:, prompt_index].detach().float().flatten()


def load_world_model(
    checkpoint_path: Path,
    in_channels: int,
    device: torch.device,
    hidden_channels: int,
) -> VisualWorldPredictor3D:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["hidden_channels"]) != hidden_channels:
        raise ValueError("World checkpoint hidden_channels does not match the requested model")
    model = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        action_dim=3,
        num_blocks=2,
        use_action=True,
        text_delta_dim=int(checkpoint["text_delta_dim"]),
        use_language=True,
        allow_unconditioned=True,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def resize_logits(logits: torch.Tensor, final_shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(logits.shape[-3:]) == final_shape:
        return logits.float()
    return F.interpolate(logits.float(), size=final_shape, mode="trilinear", align_corners=False)


def pad_label_like_image(
    interface: VoxTellStateInterface,
    label: np.ndarray,
) -> torch.Tensor:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image

    padded, _ = pad_nd_image(
        torch.from_numpy(label[None].astype(np.float32, copy=False)),
        interface.predictor.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    return padded


def binary_dice(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = prediction.bool()
    target = target.bool()
    intersection = (pred & target).sum().float()
    denominator = pred.sum().float() + target.sum().float()
    return float(((2.0 * intersection + eps) / (denominator + eps)).detach().cpu())


def safe_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    if labels.size == 0 or labels.min() == labels.max():
        return None, None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
    except ImportError:
        order = np.argsort(scores)
        sorted_labels = labels[order]
        positives = float(labels.sum())
        negatives = float(labels.size - labels.sum())
        ranks = np.arange(1, labels.size + 1, dtype=np.float64)
        rank_sum = float(ranks[sorted_labels == 1].sum())
        roc = (rank_sum - positives * (positives + 1.0) / 2.0) / max(positives * negatives, 1.0)
        cumulative = np.cumsum(sorted_labels)
        precision = cumulative / np.arange(1, labels.size + 1)
        pr = float((precision * sorted_labels).sum() / max(positives, 1.0))
        return roc, pr


def uncertainty_maps(probabilities: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    stack = torch.cat(probabilities, dim=0)
    mean = stack.mean(dim=0)
    variance = stack.var(dim=0, unbiased=False)
    std = variance.sqrt()
    pairwise = []
    for index in range(stack.shape[0]):
        for other in range(index + 1, stack.shape[0]):
            pairwise.append((stack[index] - stack[other]).abs())
    pairwise_abs = torch.stack(pairwise, dim=0).mean(dim=0) if pairwise else torch.zeros_like(mean)
    return {"mean": mean, "variance": variance, "std": std, "pairwise_abs": pairwise_abs}


def region_stats(
    uncertainty: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    tp = prediction & target
    tn = (~prediction) & (~target)
    fp = prediction & (~target)
    fn = (~prediction) & target
    regions = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    result: dict[str, Any] = {
        "tp_voxels": int(tp.sum().detach().cpu()),
        "tn_voxels": int(tn.sum().detach().cpu()),
        "fp_voxels": int(fp.sum().detach().cpu()),
        "fn_voxels": int(fn.sum().detach().cpu()),
    }
    for metric_name, values in uncertainty.items():
        flat = values.flatten()
        result[f"{metric_name}_global_mean"] = float(flat.mean().detach().cpu())
        for region_name, mask in regions.items():
            region_values = flat[mask.flatten()]
            result[f"{metric_name}_{region_name}_mean"] = (
                float(region_values.mean().detach().cpu()) if region_values.numel() else None
            )
        error_values = flat[(fp | fn).flatten()]
        correct_values = flat[(tp | tn).flatten()]
        result[f"{metric_name}_error_mean"] = float(error_values.mean().detach().cpu()) if error_values.numel() else None
        result[f"{metric_name}_correct_mean"] = float(correct_values.mean().detach().cpu()) if correct_values.numel() else None
        result[f"{metric_name}_error_minus_correct"] = (
            result[f"{metric_name}_error_mean"] - result[f"{metric_name}_correct_mean"]
            if result[f"{metric_name}_error_mean"] is not None and result[f"{metric_name}_correct_mean"] is not None
            else None
        )
    return result


@torch.inference_mode()
def evaluate_case(
    interface: VoxTellStateInterface,
    world_model: VisualWorldPredictor3D,
    case: Any,
    selected_stage: str,
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
    prediction_threshold: float,
    label_value: int,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
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
    text_delta = (flatten_prompt_embedding(embedding, 1) - flatten_prompt_embedding(embedding, 0)).to(next(world_model.parameters()).device)
    device = next(world_model.parameters()).device
    rows: list[dict[str, Any]] = []
    score_store: dict[str, list[np.ndarray]] = {name: [] for name in VARIANT_MEMBERS}
    label_store: dict[str, list[np.ndarray]] = {name: [] for name in VARIANT_MEMBERS}
    gamma_action = visual_action("gamma", 0.30, device)
    blur_action = visual_action("blur", 1.5, device)

    for patch_index, slicer in enumerate(slicers):
        original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
        gamma_patch = torch.clone(gamma_padded[slicer][None], memory_format=torch.contiguous_format)
        blur_patch = torch.clone(blur_padded[slicer][None], memory_format=torch.contiguous_format)
        gt_patch = label_padded[slicer][None].to(device)
        original_result = interface.forward_with_states(original_patch, embedding)
        source_state = original_result["decoder_states"][selected_stage][:, 0].detach().float().to(device)
        source_final_logits = original_result["final_prediction"][:, 0:1].detach().float().to(device)
        gamma_state = world_model(source_state, action=gamma_action)
        blur_state = world_model(source_state, action=blur_action)
        language_state = world_model(source_state, text_delta=text_delta[None])
        final_shape = tuple(int(size) for size in source_final_logits.shape[-3:])
        source_stage_logits = state_to_intermediate_prediction(interface, selected_stage, source_state)
        gamma_logits = state_to_intermediate_prediction(interface, selected_stage, gamma_state)
        blur_logits = state_to_intermediate_prediction(interface, selected_stage, blur_state)
        language_logits = state_to_intermediate_prediction(interface, selected_stage, language_state)
        logits = {
            "original": resize_logits(source_stage_logits, final_shape),
            "gamma_0.30": resize_logits(gamma_logits, final_shape),
            "blur_1.5": resize_logits(blur_logits, final_shape),
            "language": resize_logits(language_logits, final_shape),
        }
        probabilities = {name: torch.sigmoid(value) for name, value in logits.items()}
        original_final_probability = torch.sigmoid(source_final_logits)
        gt = (gt_patch == label_value).float()
        if tuple(gt.shape[-3:]) != final_shape:
            gt = F.interpolate(gt, size=final_shape, mode="nearest")
        gt_bool = gt > 0.5
        original_prediction = original_final_probability > prediction_threshold
        dice = binary_dice(original_prediction, gt_bool)
        for variant, members in VARIANT_MEMBERS.items():
            maps = uncertainty_maps([probabilities[name] for name in members])
            stats = region_stats(maps, original_prediction, gt_bool)
            row: dict[str, Any] = {
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "uncertainty_variant": variant,
                "ensemble_members": "+".join(members),
                "original_dice": dice,
                "original_pred_foreground_voxels": int(original_prediction.sum().detach().cpu()),
                "gt_foreground_voxels": int(gt_bool.sum().detach().cpu()),
                "original_probability_mean": float(original_final_probability.mean().detach().cpu()),
                "both_empty": int(original_prediction.sum() == 0 and gt_bool.sum() == 0),
            }
            row.update(stats)
            scores = maps["mean"].flatten().detach().cpu().numpy().astype(np.float64)
            labels = (original_prediction ^ gt_bool).flatten().detach().cpu().numpy().astype(np.int8)
            score_store[variant].append(scores)
            label_store[variant].append(labels)
            rows.append(row)
    return rows, score_store, label_store


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[key_name]) for key_name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group_rows in sorted(groups.items()):
        item: dict[str, Any] = {"num_patches": len(group_rows)}
        for key_name, value in zip(group_keys, key, strict=True):
            item[key_name] = value
        for metric in ["original_dice", "mean", "variance", "std", "pairwise_abs"]:
            item[f"mean_{metric}"] = float(np.mean([
                float(row[f"{metric}_global_mean"]) if metric != "original_dice" else float(row[metric])
                for row in group_rows
                if (metric == "original_dice" or row[f"{metric}_global_mean"] is not None)
            ])) if group_rows else None
        for metric in ["mean", "variance", "std", "pairwise_abs"]:
            for field in ["error_mean", "correct_mean", "error_minus_correct"]:
                values = [row[f"{metric}_{field}"] for row in group_rows if row[f"{metric}_{field}"] is not None]
                item[f"{metric}_{field}"] = float(np.mean(values)) if values else None
        for field in ["tp_voxels", "tn_voxels", "fp_voxels", "fn_voxels"]:
            item[field] = int(sum(int(row[field]) for row in group_rows))
        output.append(item)
    return output


def auc_summary(score_store: dict[str, list[np.ndarray]], label_store: dict[str, list[np.ndarray]]) -> list[dict[str, Any]]:
    output = []
    for variant in VARIANT_MEMBERS:
        scores = np.concatenate(score_store[variant]) if score_store[variant] else np.empty(0)
        labels = np.concatenate(label_store[variant]) if label_store[variant] else np.empty(0, dtype=np.int8)
        roc_auc, pr_auc = safe_auc(scores, labels)
        output.append({
            "uncertainty_variant": variant,
            "num_voxels": int(labels.size),
            "error_voxels": int(labels.sum()),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })
    return output


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
    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    cases = iter_cases(paths, split="test", limit=args.val_cases)
    prepare_functional_seg_head(interface, args.selected_stage)
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_world_model(
        checkpoint_path,
        int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
        device,
        args.hidden_channels,
    )
    rows: list[dict[str, Any]] = []
    score_store: dict[str, list[np.ndarray]] = {name: [] for name in VARIANT_MEMBERS}
    label_store: dict[str, list[np.ndarray]] = {name: [] for name in VARIANT_MEMBERS}
    for case in cases:
        case_rows, case_scores, case_labels = evaluate_case(
            interface, model, case, args.selected_stage, args.patches_per_case,
            args.foreground_patches_per_case, args.foreground_candidate_patches,
            args.foreground_threshold, args.prediction_threshold, args.label_value,
        )
        rows.extend(case_rows)
        for variant in VARIANT_MEMBERS:
            score_store[variant].extend(case_scores[variant])
            label_store[variant].extend(case_labels[variant])
    by_case = aggregate_rows(rows, ["case", "uncertainty_variant"])
    overall = aggregate_rows(rows, ["uncertainty_variant"])
    aucs = auc_summary(score_store, label_store)
    detail_path = output_dir / "uncertainty_by_patch.csv"
    case_path = output_dir / "uncertainty_by_case.csv"
    overall_path = output_dir / "uncertainty_overall.csv"
    auc_path = output_dir / "uncertainty_auc.csv"
    write_csv(detail_path, rows)
    write_csv(case_path, by_case)
    write_csv(overall_path, overall)
    write_csv(auc_path, aucs)
    summary = {
        "args": vars(args),
        "world_checkpoint": str(checkpoint_path),
        "selected_stage": args.selected_stage,
        "cases": [case.case for case in cases],
        "num_patch_rows": len(rows),
        "uncertainty_variants": VARIANT_MEMBERS,
        "detail_csv": str(detail_path),
        "case_csv": str(case_path),
        "overall_csv": str(overall_path),
        "auc_csv": str(auc_path),
        "overall": overall,
        "by_case": by_case,
        "uncertainty_error_auc": aucs,
        "scope_note": "V6.0 evaluation only: three independent one-step imagined states from original liver state; no training, confidence fusion, rollout, new loss, or new module.",
    }
    summary_path = output_dir / "v6_0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "summary_path": str(summary_path),
        "num_patch_rows": len(rows),
        "overall": overall,
        "uncertainty_error_auc": aucs,
    }, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
