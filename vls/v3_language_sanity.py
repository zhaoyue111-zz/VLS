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

from vls.config import ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import (
    padded_image_and_slicers,
    resolve_device,
    select_patch_slicers,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import normalized_mse


DEFAULT_PROMPT_PAIRS = [
    ("liver", "the liver"),
    ("the liver", "liver"),
    ("liver", "liver organ"),
    ("liver", "the liver organ"),
    ("liver", "segment the liver"),
]


def parse_prompt_pair(value: str) -> tuple[str, str]:
    if "::" not in value:
        raise argparse.ArgumentTypeError("Prompt pairs must use source::target format")
    source, target = value.split("::", 1)
    source = source.strip()
    target = target.strip()
    if not source or not target:
        raise argparse.ArgumentTypeError("Prompt pair source and target must be non-empty")
    return source, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3.0 frozen VoxTell language-action sanity diagnostics.")
    parser.add_argument("--model-dir", default=str(ProjectPaths().voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(ProjectPaths().voxtell_root))
    parser.add_argument("--data-root", default=str(ProjectPaths().data_root))
    parser.add_argument("--split-json", default=str(ProjectPaths().split_json))
    parser.add_argument("--output-dir", default="outputs/v3_0_language_sanity")
    parser.add_argument("--prompt-pairs", nargs="+", type=parse_prompt_pair, default=DEFAULT_PROMPT_PAIRS)
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--dev-cases", type=int, default=4)
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=12)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
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


def unique_prompts(prompt_pairs: list[tuple[str, str]]) -> list[str]:
    prompts = []
    seen = set()
    for source, target in prompt_pairs:
        for prompt in (source, target):
            if prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)
    return prompts


def flattened_embedding(embedding: torch.Tensor, prompt_index: int) -> torch.Tensor:
    if embedding.ndim == 4:
        item = embedding[:, prompt_index]
    elif embedding.ndim == 3:
        item = embedding[:, prompt_index]
    else:
        raise ValueError(f"Unexpected text embedding shape: {tuple(embedding.shape)}")
    return item.float().flatten()


def text_embedding_metrics(text_embedding: torch.Tensor, source_index: int, target_index: int) -> dict[str, float]:
    source = flattened_embedding(text_embedding.detach().cpu(), source_index)
    target = flattened_embedding(text_embedding.detach().cpu(), target_index)
    delta = target - source
    return {
        "text_embedding_cosine": float(F.cosine_similarity(source[None], target[None]).item()),
        "text_embedding_delta_norm": float(torch.linalg.vector_norm(delta).item()),
        "text_embedding_source_norm": float(torch.linalg.vector_norm(source).item()),
        "text_embedding_target_norm": float(torch.linalg.vector_norm(target).item()),
    }


def soft_dice(source_probability: torch.Tensor, target_probability: torch.Tensor, eps: float = 1e-6) -> float:
    source = source_probability.float()
    target = target_probability.float()
    intersection = (source * target).sum()
    denominator = source.sum() + target.sum()
    return float(((2.0 * intersection + eps) / (denominator + eps)).detach().cpu())


def binary_dice(source_probability: torch.Tensor, target_probability: torch.Tensor, threshold: float, eps: float = 1e-6) -> float:
    source = (source_probability > threshold).float()
    target = (target_probability > threshold).float()
    intersection = (source * target).sum()
    denominator = source.sum() + target.sum()
    return float(((2.0 * intersection + eps) / (denominator + eps)).detach().cpu())


def probability_agreement(source_logits: torch.Tensor, target_logits: torch.Tensor, threshold: float) -> dict[str, float]:
    source_probability = torch.sigmoid(source_logits.detach().float())
    target_probability = torch.sigmoid(target_logits.detach().float())
    source_foreground_voxels = float((source_probability > threshold).sum().detach().cpu())
    target_foreground_voxels = float((target_probability > threshold).sum().detach().cpu())
    foreground_denominator = max(source_foreground_voxels, 1.0)
    return {
        "soft_dice": soft_dice(source_probability, target_probability),
        "binary_dice": binary_dice(source_probability, target_probability, threshold),
        "probability_mae": float((source_probability - target_probability).abs().mean().detach().cpu()),
        "source_probability_mean": float(source_probability.mean().detach().cpu()),
        "target_probability_mean": float(target_probability.mean().detach().cpu()),
        "source_foreground_voxels": source_foreground_voxels,
        "target_foreground_voxels": target_foreground_voxels,
        "target_to_source_foreground_ratio": target_foreground_voxels / foreground_denominator,
        "both_empty": source_foreground_voxels == 0.0 and target_foreground_voxels == 0.0,
        "source_to_empty_collapse": source_foreground_voxels > 0.0 and target_foreground_voxels == 0.0,
    }


@torch.inference_mode()
def evaluate_case(
    interface: VoxTellStateInterface,
    case: Any,
    split: str,
    prompt_pairs: list[tuple[str, str]],
    prompt_to_index: dict[str, int],
    prompts: list[str],
    selected_stage: str,
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
    prediction_threshold: float,
) -> list[dict[str, float | int | str]]:
    image, _, _ = read_image_and_label(case)
    padded, slicers, patch_kinds = select_patch_slicers(
        interface,
        image,
        [prompt_pairs[0][0]],
        patches_per_case,
        foreground_patches_per_case,
        foreground_candidate_patches,
        foreground_threshold,
    )
    text_embedding = interface.embed_text_prompts(prompts)
    embedding_metrics = {
        (source, target): text_embedding_metrics(text_embedding, prompt_to_index[source], prompt_to_index[target])
        for source, target in prompt_pairs
    }
    rows = []
    for patch_index, slicer in enumerate(slicers):
        patch = torch.clone(padded[slicer][None], memory_format=torch.contiguous_format)
        result = interface.forward_with_states(patch, text_embedding)
        stage_states = result["decoder_states"][selected_stage]
        stage_predictions = result["intermediate_predictions"][selected_stage]
        final_predictions = result["final_prediction"]
        for source, target in prompt_pairs:
            source_index = prompt_to_index[source]
            target_index = prompt_to_index[target]
            source_state = stage_states[:, source_index]
            target_state = stage_states[:, target_index]
            source_stage_logits = stage_predictions[:, source_index : source_index + 1]
            target_stage_logits = stage_predictions[:, target_index : target_index + 1]
            source_final_logits = final_predictions[:, source_index : source_index + 1]
            target_final_logits = final_predictions[:, target_index : target_index + 1]
            intermediate_agreement = probability_agreement(source_stage_logits, target_stage_logits, prediction_threshold)
            final_agreement = probability_agreement(source_final_logits, target_final_logits, prediction_threshold)
            rows.append({
                "split": split,
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "source_prompt": source,
                "target_prompt": target,
                "prompt_pair": f"{source} -> {target}",
                **embedding_metrics[(source, target)],
                "state_normalized_mse": float(normalized_mse(source_state, target_state).detach().cpu()),
                "intermediate_mask_logit_normalized_mse": float(
                    normalized_mse(source_stage_logits, target_stage_logits).detach().cpu()
                ),
                "intermediate_soft_dice": intermediate_agreement["soft_dice"],
                "intermediate_binary_dice": intermediate_agreement["binary_dice"],
                "intermediate_probability_mae": intermediate_agreement["probability_mae"],
                "intermediate_source_probability_mean": intermediate_agreement["source_probability_mean"],
                "intermediate_target_probability_mean": intermediate_agreement["target_probability_mean"],
                "intermediate_source_foreground_voxels": intermediate_agreement["source_foreground_voxels"],
                "intermediate_target_foreground_voxels": intermediate_agreement["target_foreground_voxels"],
                "intermediate_target_to_source_foreground_ratio": intermediate_agreement["target_to_source_foreground_ratio"],
                "intermediate_both_empty": int(intermediate_agreement["both_empty"]),
                "intermediate_source_to_empty_collapse": int(intermediate_agreement["source_to_empty_collapse"]),
                "final_soft_dice": final_agreement["soft_dice"],
                "final_binary_dice": final_agreement["binary_dice"],
                "final_probability_mae": final_agreement["probability_mae"],
                "final_source_probability_mean": final_agreement["source_probability_mean"],
                "final_target_probability_mean": final_agreement["target_probability_mean"],
                "final_source_foreground_voxels": final_agreement["source_foreground_voxels"],
                "final_target_foreground_voxels": final_agreement["target_foreground_voxels"],
                "final_target_to_source_foreground_ratio": final_agreement["target_to_source_foreground_ratio"],
                "final_both_empty": int(final_agreement["both_empty"]),
                "final_source_to_empty_collapse": int(final_agreement["source_to_empty_collapse"]),
            })
    return rows


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def sum_metric(rows: list[dict[str, Any]], key: str) -> int:
    return int(np.sum([int(row[key]) for row in rows])) if rows else 0


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[group_key]) for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    metric_keys = [
        "text_embedding_cosine",
        "text_embedding_delta_norm",
        "state_normalized_mse",
        "intermediate_mask_logit_normalized_mse",
        "intermediate_soft_dice",
        "intermediate_binary_dice",
        "intermediate_probability_mae",
        "intermediate_target_to_source_foreground_ratio",
        "final_soft_dice",
        "final_binary_dice",
        "final_probability_mae",
        "final_target_to_source_foreground_ratio",
    ]
    summaries = []
    for key, group_rows in sorted(groups.items()):
        summary: dict[str, float | int | str] = {"num_samples": len(group_rows)}
        for group_key, group_value in zip(group_keys, key, strict=True):
            summary[group_key] = group_value
        for metric_key in metric_keys:
            summary[f"mean_{metric_key}"] = mean_metric(group_rows, metric_key)
        summary["intermediate_both_empty_count"] = sum_metric(group_rows, "intermediate_both_empty")
        summary["intermediate_source_to_empty_collapse_count"] = sum_metric(
            group_rows,
            "intermediate_source_to_empty_collapse",
        )
        summary["final_both_empty_count"] = sum_metric(group_rows, "final_both_empty")
        summary["final_source_to_empty_collapse_count"] = sum_metric(group_rows, "final_source_to_empty_collapse")
        summaries.append(summary)
    return summaries


def foreground_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if int(row["final_both_empty"]) == 0 or int(row["intermediate_both_empty"]) == 0
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
    train_cases = iter_cases(paths, split="train", limit=args.dev_cases)
    val_cases = iter_cases(paths, split="test", limit=args.val_cases) if args.val_cases else []
    prompts = unique_prompts(args.prompt_pairs)
    prompt_to_index = {prompt: index for index, prompt in enumerate(prompts)}

    rows = []
    for split, cases in [("train", train_cases), ("val", val_cases)]:
        for case in cases:
            rows.extend(evaluate_case(
                interface,
                case,
                split,
                args.prompt_pairs,
                prompt_to_index,
                prompts,
                args.selected_stage,
                args.patches_per_case,
                args.foreground_patches_per_case,
                args.foreground_candidate_patches,
                args.foreground_threshold,
                args.prediction_threshold,
            ))

    per_pair = aggregate_rows(rows, ["split", "prompt_pair"])
    per_case = aggregate_rows(rows, ["split", "case", "prompt_pair"])
    per_patch = aggregate_rows(rows, ["split", "case", "patch_index", "patch_kind", "prompt_pair"])
    overall = aggregate_rows(rows, ["prompt_pair"])
    fg_rows = foreground_rows(rows)
    foreground_per_pair = aggregate_rows(fg_rows, ["split", "prompt_pair"])
    foreground_per_case = aggregate_rows(fg_rows, ["split", "case", "prompt_pair"])
    foreground_per_patch = aggregate_rows(fg_rows, ["split", "case", "patch_index", "patch_kind", "prompt_pair"])
    foreground_overall = aggregate_rows(fg_rows, ["prompt_pair"])

    detail_path = output_dir / "language_action_sanity.csv"
    per_pair_path = output_dir / "language_action_sanity_by_pair.csv"
    per_case_path = output_dir / "language_action_sanity_by_case.csv"
    per_patch_path = output_dir / "language_action_sanity_by_patch.csv"
    foreground_per_pair_path = output_dir / "language_action_sanity_foreground_by_pair.csv"
    foreground_per_case_path = output_dir / "language_action_sanity_foreground_by_case.csv"
    foreground_per_patch_path = output_dir / "language_action_sanity_foreground_by_patch.csv"
    write_csv(detail_path, rows)
    write_csv(per_pair_path, per_pair)
    write_csv(per_case_path, per_case)
    write_csv(per_patch_path, per_patch)
    write_csv(foreground_per_pair_path, foreground_per_pair)
    write_csv(foreground_per_case_path, foreground_per_case)
    write_csv(foreground_per_patch_path, foreground_per_patch)

    summary = {
        "args": {
            **vars(args),
            "prompt_pairs": [[source, target] for source, target in args.prompt_pairs],
        },
        "train_cases": [case.case for case in train_cases],
        "val_cases": [case.case for case in val_cases],
        "prompts": prompts,
        "selected_stage": args.selected_stage,
        "num_rows": len(rows),
        "detail_csv": str(detail_path),
        "per_pair_csv": str(per_pair_path),
        "per_case_csv": str(per_case_path),
        "per_patch_csv": str(per_patch_path),
        "foreground_per_pair_csv": str(foreground_per_pair_path),
        "foreground_per_case_csv": str(foreground_per_case_path),
        "foreground_per_patch_csv": str(foreground_per_patch_path),
        "overall_by_prompt_pair": overall,
        "split_by_prompt_pair": per_pair,
        "foreground_overall_by_prompt_pair": foreground_overall,
        "foreground_split_by_prompt_pair": foreground_per_pair,
        "interpretation_note": "Sanity only: nonzero text/latent transition plus foreground-stable prediction agreement is required before designing a V3 language action encoder. Empty source/target patches are counted but should not be used as binary Dice evidence for semantic consistency.",
    }
    summary_path = output_dir / "language_action_sanity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
