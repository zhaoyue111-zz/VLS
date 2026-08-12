from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases
from vls.v2_experiment import (
    build_dataset as build_visual_dataset,
    compute_eval_records as compute_visual_eval_records,
    grouped_indices as visual_grouped_indices,
    macro_records_from_case_records as visual_macro_records,
    prepare_functional_seg_head,
    resolve_device,
    state_to_intermediate_prediction,
    subset_data as visual_subset_data,
)
from vls.v3_language_experiment import (
    SOURCE_PROMPT,
    TARGET_PROMPT,
    build_dataset as build_language_dataset,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, normalized_mse


DEFAULT_GAMMA_STRENGTHS = [-0.3, -0.15, 0.15, 0.3]
DEFAULT_BLUR_SIGMAS = [0.5, 1.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3.2a unified V2 visual and language world predictor.")
    paths = ProjectPaths()
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--v2-checkpoint", default="outputs/v2_final/world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v3_2a_unified")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--dev-cases", type=int, default=8)
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", nargs="+", type=int, default=[0, 100, 200])
    parser.add_argument("--gamma-strengths", nargs="+", type=float, default=DEFAULT_GAMMA_STRENGTHS)
    parser.add_argument("--blur-sigmas", nargs="+", type=float, default=DEFAULT_BLUR_SIGMAS)
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_v2_unified_predictor(
    checkpoint_path: Path,
    in_channels: int,
    text_delta_dim: int,
    hidden_channels: int,
    device: torch.device,
) -> VisualWorldPredictor3D:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["hidden_channels"]) != hidden_channels:
        raise ValueError("V2 checkpoint hidden_channels does not match the requested unified model")
    model = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        action_dim=int(checkpoint.get("action_dim", 3)),
        num_blocks=2,
        use_action=True,
        text_delta_dim=text_delta_dim,
        use_language=True,
        allow_unconditioned=True,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["conditioned_state_dict"], strict=False)
    expected_missing = {"language_action_encoder.weight", "language_action_encoder.bias"}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(f"Unexpected V2 load result: missing={missing}, unexpected={unexpected}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("language_action_encoder.")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable != ["language_action_encoder.weight", "language_action_encoder.bias"]:
        raise RuntimeError(f"V3.2a must train only language encoder, got {trainable}")
    model.eval()
    return model


@torch.inference_mode()
def evaluate_language(
    model: nn.Module,
    data: dict[str, Any],
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
    conditioned: bool,
    wrong_direction: bool = False,
) -> dict[str, float]:
    states = data["states"]
    targets = data["targets"]
    deltas = data["text_deltas"]
    target_predictions = data["target_predictions"]
    state_sse = 0.0
    state_target_sq = 0.0
    mask_sse = 0.0
    mask_target_sq = 0.0
    device = next(model.parameters()).device
    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        state_batch = states[start:end].to(device)
        target_batch = targets[start:end].to(device)
        delta_batch = deltas[start:end].to(device)
        if wrong_direction:
            delta_batch = -delta_batch
        prediction = model(
            state_batch,
            text_delta=delta_batch if conditioned else None,
        )
        target_prediction_batch = target_predictions[start:end].to(device)
        pred_logits = state_to_intermediate_prediction(interface, selected_stage, prediction)
        state_sse += float((prediction.float() - target_batch.float()).pow(2).sum().cpu())
        state_target_sq += float(target_batch.float().pow(2).sum().cpu())
        mask_sse += float((pred_logits.float() - target_prediction_batch.float()).pow(2).sum().cpu())
        mask_target_sq += float(target_prediction_batch.float().pow(2).sum().cpu())
    return {
        "state_normalized_mse": state_sse / max(state_target_sq, 1e-6),
        "mask_logit_normalized_mse": mask_sse / max(mask_target_sq, 1e-6),
    }


@torch.inference_mode()
def evaluate_language_identity(
    data: dict[str, Any],
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> dict[str, float]:
    states = data["states"]
    targets = data["targets"]
    target_predictions = data["target_predictions"]
    state_sse = 0.0
    state_target_sq = 0.0
    mask_sse = 0.0
    mask_target_sq = 0.0
    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        state_batch = states[start:end].to(interface.device)
        target_batch = targets[start:end].to(interface.device)
        target_prediction_batch = target_predictions[start:end].to(interface.device)
        pred_logits = state_to_intermediate_prediction(interface, selected_stage, state_batch)
        state_sse += float((state_batch.float() - target_batch.float()).pow(2).sum().cpu())
        state_target_sq += float(target_batch.float().pow(2).sum().cpu())
        mask_sse += float((pred_logits.float() - target_prediction_batch.float()).pow(2).sum().cpu())
        mask_target_sq += float(target_prediction_batch.float().pow(2).sum().cpu())
    return {
        "state_normalized_mse": state_sse / max(state_target_sq, 1e-6),
        "mask_logit_normalized_mse": mask_sse / max(mask_target_sq, 1e-6),
    }


def grouped_indices(data: dict[str, Any], key: str) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(data[key]):
        groups.setdefault(str(value), []).append(index)
    return sorted(groups.items(), key=lambda item: item[0])


def subset_language_data(data: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    index_tensor = torch.tensor(indices, dtype=torch.long, device=data["states"].device)
    return {
        "states": data["states"].index_select(0, index_tensor),
        "targets": data["targets"].index_select(0, index_tensor),
        "text_deltas": data["text_deltas"].index_select(0, index_tensor),
        "target_predictions": data["target_predictions"].index_select(0, index_tensor),
        "case_ids": [data["case_ids"][i] for i in indices],
        "case_names": [data["case_names"][i] for i in indices],
        "patch_indices": [data["patch_indices"][i] for i in indices],
        "patch_kinds": [data["patch_kinds"][i] for i in indices],
        "directions": [data["directions"][i] for i in indices],
    }


def language_records(
    step: int,
    split: str,
    group: str,
    group_value: str,
    data: dict[str, Any],
    model: nn.Module,
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    records = []
    evaluations = [
        ("identity", evaluate_language_identity(data, interface, selected_stage, batch_size)),
        ("shared_trunk_no_language", evaluate_language(model, data, interface, selected_stage, batch_size, False)),
        ("language_conditioned", evaluate_language(model, data, interface, selected_stage, batch_size, True)),
        (
            "language_conditioned_wrong_direction",
            evaluate_language(model, data, interface, selected_stage, batch_size, True, True),
        ),
    ]
    for model_name, metrics in evaluations:
        records.append({
            "step": step,
            "split": split,
            "group": group,
            "group_value": group_value,
            "model": model_name,
            **metrics,
        })
    return records


def language_macro_records(step: int, split: str, case_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model_name in sorted({str(row["model"]) for row in case_records}):
        model_rows = [row for row in case_records if row["model"] == model_name]
        rows.append({
            "step": step,
            "split": split,
            "group": "overall_macro",
            "group_value": "case_mean",
            "model": model_name,
            "state_normalized_mse": float(np.mean([float(row["state_normalized_mse"]) for row in model_rows])),
            "mask_logit_normalized_mse": float(np.mean([float(row["mask_logit_normalized_mse"]) for row in model_rows])),
        })
    return rows


def language_pass_summary(
    rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def grouped(group: str, value: str) -> dict[str, dict[str, Any]]:
        return {
            str(row["model"]): row
            for row in group_rows
            if row["split"] == "val" and row["group"] == group and row["group_value"] == value
        }

    micro = {
        str(row["model"]): row
        for row in rows
        if row["split"] == "val" and row["group"] == "overall_micro"
    }
    macro = {
        str(row["model"]): row
        for row in rows
        if row["split"] == "val" and row["group"] == "overall_macro"
    }

    def state(items: dict[str, dict[str, Any]], name: str) -> float:
        return float(items[name]["state_normalized_mse"])

    def mask(items: dict[str, dict[str, Any]], name: str) -> float:
        return float(items[name]["mask_logit_normalized_mse"])

    directions = []
    for direction in sorted({str(row["group_value"]) for row in group_rows if row["split"] == "val" and row["group"] == "directions"}):
        items = grouped("directions", direction)
        directions.append({
            "direction": direction,
            "conditioned_lt_no_language": state(items, "language_conditioned") < state(items, "shared_trunk_no_language"),
            "correct_lt_wrong_direction": state(items, "language_conditioned") < state(items, "language_conditioned_wrong_direction"),
        })
    cases = []
    for case in sorted({str(row["group_value"]) for row in group_rows if row["split"] == "val" and row["group"] == "case_names"}):
        items = grouped("case_names", case)
        cases.append({
            "case": case,
            "conditioned_lt_no_language": state(items, "language_conditioned") < state(items, "shared_trunk_no_language"),
            "correct_lt_wrong_direction": state(items, "language_conditioned") < state(items, "language_conditioned_wrong_direction"),
        })
    case_wins = sum(item["conditioned_lt_no_language"] for item in cases)
    direction_pass = all(item["conditioned_lt_no_language"] and item["correct_lt_wrong_direction"] for item in directions)
    return {
        "passed": (
            state(micro, "language_conditioned") < state(micro, "shared_trunk_no_language")
            and state(micro, "language_conditioned") < state(micro, "language_conditioned_wrong_direction")
            and state(macro, "language_conditioned") < state(macro, "shared_trunk_no_language")
            and state(macro, "language_conditioned") < state(macro, "language_conditioned_wrong_direction")
            and mask(micro, "language_conditioned") < mask(micro, "shared_trunk_no_language")
            and direction_pass
            and case_wins >= 3
        ),
        "micro_conditioned_lt_no_language": state(micro, "language_conditioned") < state(micro, "shared_trunk_no_language"),
        "micro_correct_lt_wrong_direction": state(micro, "language_conditioned") < state(micro, "language_conditioned_wrong_direction"),
        "macro_conditioned_lt_no_language": state(macro, "language_conditioned") < state(macro, "shared_trunk_no_language"),
        "macro_correct_lt_wrong_direction": state(macro, "language_conditioned") < state(macro, "language_conditioned_wrong_direction"),
        "micro_mask_improved": mask(micro, "language_conditioned") < mask(micro, "shared_trunk_no_language"),
        "macro_mask_improved": mask(macro, "language_conditioned") < mask(macro, "shared_trunk_no_language"),
        "case_win_count": int(case_wins),
        "case_total": len(cases),
        "direction_wins": directions,
        "case_wins": cases,
        "criterion": "V3.2a language-only encoder: micro/macro state and correct-vs-wrong direction, 2/2 directions, at least 3/4 cases, and overall mask improvement.",
    }


def language_pass_summaries_by_step(
    rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for step in sorted({int(row["step"]) for row in rows}):
        step_rows = [row for row in rows if int(row["step"]) == step]
        step_group_rows = [row for row in group_rows if int(row["step"]) == step]
        summaries[str(step)] = language_pass_summary(step_rows, step_group_rows)
    return summaries


def collect_language_eval(
    step: int,
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    model: nn.Module,
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for split, data in [("train", train_data), ("val", val_data)]:
        rows.extend(language_records(step, split, "overall_micro", "all", data, model, interface, selected_stage, batch_size))
        case_records = []
        for group_name in ["directions", "case_names"]:
            for group_value, indices in grouped_indices(data, group_name):
                records = language_records(
                    step, split, group_name, group_value,
                    subset_language_data(data, indices), model, interface, selected_stage, batch_size,
                )
                group_rows.extend(records)
                if group_name == "case_names":
                    case_records.extend(records)
        rows.extend(language_macro_records(step, split, case_records))
    return rows, group_rows


def collect_visual_eval(
    step: int,
    split: str,
    data: dict[str, Any],
    model: nn.Module,
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    models = {"action_conditioned": model}
    use_action = {"action_conditioned": True}
    rows.extend(compute_visual_eval_records(
        step, split, "overall_micro", "all", data, models, use_action, interface, selected_stage, batch_size,
    ))
    case_records = []
    for group_name in ["action_families", "strengths", "case_names"]:
        for group_value, indices in visual_grouped_indices(data, group_name):
            records = compute_visual_eval_records(
                step, split, group_name, group_value, visual_subset_data(data, indices),
                models, use_action, interface, selected_stage, batch_size,
            )
            group_rows.extend(records)
            if group_name == "case_names":
                case_records.extend(records)
    rows.extend(visual_macro_records(step, split, case_records))
    return rows, group_rows


def train_language_encoder(
    model: nn.Module,
    train_data: dict[str, Any],
    train_eval_data: dict[str, Any],
    val_data: dict[str, Any],
    visual_val_data: dict[str, Any],
    interface: VoxTellStateInterface,
    selected_stage: str,
    max_steps: int,
    eval_steps: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    optimizer = torch.optim.AdamW(model.language_action_encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    eval_set = sorted(set([0, max_steps, *eval_steps]))
    language_rows: list[dict[str, Any]] = []
    language_group_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    visual_group_rows: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}

    def append_eval(step: int) -> None:
        rows, groups = collect_language_eval(step, train_eval_data, val_data, model, interface, selected_stage, batch_size)
        language_rows.extend(rows)
        language_group_rows.extend(groups)
        visual_micro, visual_groups = collect_visual_eval(step, "val", visual_val_data, model, interface, selected_stage, batch_size)
        visual_rows.extend(visual_micro)
        visual_group_rows.extend(visual_groups)
        checkpoints[step] = cpu_state_dict(model)

    append_eval(0)
    states = train_data["states"]
    targets = train_data["targets"]
    deltas = train_data["text_deltas"]
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(torch.initial_seed())
    permutation = torch.randperm(states.shape[0], generator=generator)
    cursor = 0
    for step in range(1, max_steps + 1):
        if cursor + batch_size > states.shape[0]:
            permutation = torch.randperm(states.shape[0], generator=generator)
            cursor = 0
        indices = permutation[cursor:cursor + batch_size]
        cursor += batch_size
        state_batch = states.index_select(0, indices).to(device)
        target_batch = targets.index_select(0, indices).to(device)
        delta_batch = deltas.index_select(0, indices).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(state_batch, text_delta=delta_batch)
        loss = normalized_mse(prediction, target_batch)
        loss.backward()
        optimizer.step()
        model.eval()
        if step in eval_set:
            append_eval(step)
    return language_rows, language_group_rows, visual_rows, visual_group_rows, checkpoints


def rename_visual_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    renamed = []
    for row in rows:
        item = dict(row)
        if row["model"] == "identity":
            item["model"] = "identity"
        elif row["model"] == "action_conditioned":
            item["model"] = prefix
        else:
            item["model"] = prefix + row["model"].removeprefix("action_conditioned")
        renamed.append(item)
    return renamed


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
    val_cases = iter_cases(paths, split="test", limit=args.val_cases)
    visual_values = {"gamma": list(args.gamma_strengths), "blur": list(args.blur_sigmas)}

    language_train = build_language_dataset(
        interface, train_cases, args.selected_stage, args.patches_per_case,
        args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
    )
    language_val = build_language_dataset(
        interface, val_cases, args.selected_stage, args.patches_per_case,
        args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
    )
    visual_val = build_visual_dataset(
        interface, val_cases, DEFAULT_PROMPTS, ["gamma", "blur"], visual_values,
        args.selected_stage, args.patches_per_case, args.foreground_patches_per_case,
        args.foreground_candidate_patches, args.foreground_threshold,
    )
    prepare_functional_seg_head(interface, args.selected_stage)

    checkpoint_path = Path(args.v2_checkpoint)
    model = load_v2_unified_predictor(
        checkpoint_path, language_train["states"].shape[1], language_train["text_delta_dim"],
        args.hidden_channels, device,
    )
    baseline = VisualWorldPredictor3D(
        in_channels=language_train["states"].shape[1], hidden_channels=args.hidden_channels,
        action_dim=3, use_action=True,
    ).to(device)
    v2_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    baseline.load_state_dict(v2_checkpoint["conditioned_state_dict"])
    baseline.eval()

    language_rows, language_group_rows, visual_rows, visual_group_rows, checkpoints = train_language_encoder(
        model, language_train, language_train, language_val, visual_val, interface,
        args.selected_stage, args.max_train_steps, args.eval_steps, args.batch_size,
    )

    fields = ["step", "split", "group", "group_value", "model", "state_normalized_mse", "mask_logit_normalized_mse"]
    language_curve_path = output_dir / "language_training_curve.csv"
    language_group_path = output_dir / "language_grouped_training_curve.csv"
    visual_curve_path = output_dir / "visual_validation_curve.csv"
    visual_group_path = output_dir / "visual_grouped_validation_curve.csv"
    write_csv(language_curve_path, language_rows, fields)
    write_csv(language_group_path, language_group_rows, fields)
    write_csv(visual_curve_path, visual_rows, fields)
    write_csv(visual_group_path, visual_group_rows, fields)

    diagnostic_rows = []
    for split, data in [("train", language_train), ("val", language_val)]:
        diagnostic_rows.extend({"split": split, **row} for row in data["diagnostics"])
    diagnostic_path = output_dir / "language_transition_diagnostics.csv"
    write_csv(diagnostic_path, diagnostic_rows, ["split", "case", "patch_index", "patch_kind", "direction", "state_normalized_mse", "mask_logit_normalized_mse"])

    language_step_summaries = language_pass_summaries_by_step(language_rows, language_group_rows)
    final_step = args.max_train_steps
    final_language_rows = [row for row in language_rows if int(row["step"]) == final_step]
    final_language_groups = [row for row in language_group_rows if int(row["step"]) == final_step]
    language_summary = language_pass_summary(final_language_rows, final_language_groups)
    final_visual_rows = [row for row in visual_rows if int(row["step"]) == final_step]
    final_visual_groups = [row for row in visual_group_rows if int(row["step"]) == final_step]

    baseline_rows, baseline_groups = collect_visual_eval(final_step, "val", visual_val, baseline, interface, args.selected_stage, args.batch_size)
    baseline_rows = rename_visual_rows(baseline_rows, "v2_loaded_baseline")
    baseline_groups = rename_visual_rows(baseline_groups, "v2_loaded_baseline")
    unified_visual_rows = rename_visual_rows(final_visual_rows, "unified_visual")
    unified_visual_groups = rename_visual_rows(final_visual_groups, "unified_visual")
    baseline_micro = {row["model"]: row for row in baseline_rows if row["group"] == "overall_micro"}
    unified_micro = {row["model"]: row for row in unified_visual_rows if row["group"] == "overall_micro"}
    baseline_by_role = {
        ("identity" if name == "identity" else name.removeprefix("v2_loaded_baseline")): row
        for name, row in baseline_micro.items()
    }
    unified_by_role = {
        ("identity" if name == "identity" else name.removeprefix("unified_visual")): row
        for name, row in unified_micro.items()
    }
    visual_comparison = {
        "max_abs_overall_micro_metric_difference": max(
            abs(float(unified_by_role[name][metric]) - float(baseline_by_role[name][metric]))
            for name in unified_by_role
            for metric in ["state_normalized_mse", "mask_logit_normalized_mse"]
            if name in baseline_by_role
        ),
        "baseline_overall_micro": baseline_rows,
        "unified_overall_micro": final_visual_rows,
    }

    checkpoint_paths = {}
    for checkpoint_step in sorted(set([final_step, *args.eval_steps])):
        if checkpoint_step not in checkpoints:
            continue
        checkpoint_output = output_dir / f"unified_world_predictor_step{checkpoint_step}.pt"
        torch.save({
            "selected_step": checkpoint_step,
            "selected_stage": args.selected_stage,
            "hidden_channels": args.hidden_channels,
            "text_delta_dim": language_train["text_delta_dim"],
            "v2_checkpoint": str(checkpoint_path),
            "language_actions": [
                "liver -> the liver: E(the liver) - E(liver)",
                "the liver -> liver: E(liver) - E(the liver)",
            ],
            "language_encoder_zero_init": True,
            "trainable_parameters": ["language_action_encoder.weight", "language_action_encoder.bias"],
            "state_dict": checkpoints[checkpoint_step],
            "args": vars(args),
        }, checkpoint_output)
        checkpoint_paths[str(checkpoint_step)] = str(checkpoint_output)
    final_checkpoint_path = Path(checkpoint_paths[str(final_step)])

    summary = {
        "args": vars(args),
        "train_cases": [case.case for case in train_cases],
        "val_cases": [case.case for case in val_cases],
        "train_language_samples": len(language_train["case_ids"]),
        "val_language_samples": len(language_val["case_ids"]),
        "val_visual_samples": len(visual_val["case_ids"]),
        "selected_stage": args.selected_stage,
        "v2_checkpoint": str(checkpoint_path),
        "text_delta_dim": language_train["text_delta_dim"],
        "text_delta_norm": language_train["text_delta_norm"],
        "language_training_curve_csv": str(language_curve_path),
        "language_grouped_training_curve_csv": str(language_group_path),
        "visual_validation_curve_csv": str(visual_curve_path),
        "visual_grouped_validation_curve_csv": str(visual_group_path),
        "language_transition_diagnostics_csv": str(diagnostic_path),
        "checkpoint_path": str(final_checkpoint_path),
        "checkpoint_paths": checkpoint_paths,
        "language_final_metrics": final_language_rows,
        "language_final_group_metrics": final_language_groups,
        "language_pass_summary": language_summary,
        "language_step_pass_summaries": language_step_summaries,
        "visual_final_metrics": final_visual_rows,
        "visual_final_group_metrics": final_visual_groups,
        "visual_comparison": visual_comparison,
        "scope_note": "V3.2a unified predictor only: V2 trunk/action/output frozen; only zero-initialized language_action_encoder is trained; no new loss or V4 functionality.",
    }
    (output_dir / "v3_2a_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
