from __future__ import annotations

import argparse
import csv
import json
import random
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
    subset_data as visual_subset_data,
    v2_pass_summary,
)
from vls.v3_2a_unified_experiment import (
    DEFAULT_BLUR_SIGMAS,
    DEFAULT_GAMMA_STRENGTHS,
    build_language_dataset,
    collect_language_eval,
    cpu_state_dict,
    language_pass_summary,
    set_seed,
    subset_language_data,
    write_csv,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, normalized_mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3.2b joint visual-language fine-tuning.")
    paths = ProjectPaths()
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--v2-checkpoint", default="outputs/v2_final/world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v3_2b_joint")
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
    parser.add_argument("--language-lr", type=float, default=1e-3)
    parser.add_argument("--shared-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def write_model_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["step", "split", "group", "group_value", "model", "state_normalized_mse", "mask_logit_normalized_mse"]
    write_csv(path, rows, fields)


def make_model(
    checkpoint_path: Path,
    in_channels: int,
    text_delta_dim: int,
    hidden_channels: int,
    device: torch.device,
    state_key: str = "conditioned_state_dict",
) -> VisualWorldPredictor3D:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["hidden_channels"]) != hidden_channels:
        raise ValueError("V2 checkpoint hidden_channels does not match the requested model")
    action_dim = int(checkpoint.get("action_dim", 3))
    if state_key != "conditioned_state_dict":
        action_dim = int(checkpoint[state_key]["action_mlp.0.weight"].shape[1])
    model = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        action_dim=action_dim,
        num_blocks=2,
        use_action=state_key == "conditioned_state_dict",
        text_delta_dim=text_delta_dim if state_key == "conditioned_state_dict" else None,
        use_language=state_key == "conditioned_state_dict",
        allow_unconditioned=True,
    ).to(device)
    state_dict = checkpoint[state_key]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected_missing = (
        {"language_action_encoder.weight", "language_action_encoder.bias"}
        if state_key == "conditioned_state_dict" else set()
    )
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint load: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def make_joint_model(
    checkpoint_path: Path,
    in_channels: int,
    text_delta_dim: int,
    hidden_channels: int,
    device: torch.device,
) -> VisualWorldPredictor3D:
    model = make_model(checkpoint_path, in_channels, text_delta_dim, hidden_channels, device)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def collect_visual_eval(
    step: int,
    split: str,
    data: dict[str, Any],
    joint_model: nn.Module,
    v2_agnostic: nn.Module,
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = {
        "action_agnostic": v2_agnostic,
        "action_conditioned": joint_model,
    }
    use_action = {"action_agnostic": False, "action_conditioned": True}
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    rows.extend(compute_visual_eval_records(
        step, split, "overall_micro", "all", data, models, use_action,
        interface, selected_stage, batch_size,
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


def rename_visual_baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renamed = []
    for row in rows:
        item = dict(row)
        if row["model"] == "action_conditioned":
            item["model"] = "v2_conditioned_baseline"
        elif row["model"] == "action_agnostic":
            item["model"] = "v2_agnostic_baseline"
        else:
            item["model"] = "v2_baseline_" + str(row["model"]).removeprefix("action_conditioned_")
        renamed.append(item)
    return renamed


def language_joint_pass_summary(
    rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = language_pass_summary(rows, group_rows)

    def records(group: str, value: str) -> dict[str, dict[str, Any]]:
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

    def state(items: dict[str, dict[str, Any]], model: str) -> float:
        return float(items[model]["state_normalized_mse"])

    def mask(items: dict[str, dict[str, Any]], model: str) -> float:
        return float(items[model]["mask_logit_normalized_mse"])

    direction_wins = []
    for direction in sorted({str(row["group_value"]) for row in group_rows if row["split"] == "val" and row["group"] == "directions"}):
        items = records("directions", direction)
        direction_wins.append({
            "direction": direction,
            "conditioned_lt_identity": state(items, "language_conditioned") < state(items, "identity"),
            "conditioned_lt_no_language": state(items, "language_conditioned") < state(items, "shared_trunk_no_language"),
            "correct_lt_wrong_direction": state(items, "language_conditioned") < state(items, "language_conditioned_wrong_direction"),
        })
    case_wins = []
    for case in sorted({str(row["group_value"]) for row in group_rows if row["split"] == "val" and row["group"] == "case_names"}):
        items = records("case_names", case)
        case_wins.append({
            "case": case,
            "conditioned_lt_identity": state(items, "language_conditioned") < state(items, "identity"),
            "conditioned_lt_no_language": state(items, "language_conditioned") < state(items, "shared_trunk_no_language"),
            "correct_lt_wrong_direction": state(items, "language_conditioned") < state(items, "language_conditioned_wrong_direction"),
        })
    direction_pass = all(
        item["conditioned_lt_identity"]
        and item["conditioned_lt_no_language"]
        and item["correct_lt_wrong_direction"]
        for item in direction_wins
    )
    case_pass_count = sum(
        item["conditioned_lt_identity"]
        and item["conditioned_lt_no_language"]
        and item["correct_lt_wrong_direction"]
        for item in case_wins
    )
    micro_mask_identity = mask(micro, "language_conditioned") < mask(micro, "identity")
    macro_mask_identity = mask(macro, "language_conditioned") < mask(macro, "identity")
    micro_mask_no_language = mask(micro, "language_conditioned") < mask(micro, "shared_trunk_no_language")
    macro_mask_no_language = mask(macro, "language_conditioned") < mask(macro, "shared_trunk_no_language")
    summary.update({
        "passed": (
            state(micro, "language_conditioned") < state(micro, "identity")
            and state(micro, "language_conditioned") < state(micro, "shared_trunk_no_language")
            and state(micro, "language_conditioned") < state(micro, "language_conditioned_wrong_direction")
            and state(macro, "language_conditioned") < state(macro, "identity")
            and state(macro, "language_conditioned") < state(macro, "shared_trunk_no_language")
            and state(macro, "language_conditioned") < state(macro, "language_conditioned_wrong_direction")
            and direction_pass
            and case_pass_count >= 3
            and micro_mask_identity
            and macro_mask_identity
            and micro_mask_no_language
            and macro_mask_no_language
        ),
        "micro_conditioned_lt_identity": state(micro, "language_conditioned") < state(micro, "identity"),
        "macro_conditioned_lt_identity": state(macro, "language_conditioned") < state(macro, "identity"),
        "micro_mask_improved_vs_identity": micro_mask_identity,
        "macro_mask_improved_vs_identity": macro_mask_identity,
        "micro_mask_improved_vs_no_language": micro_mask_no_language,
        "macro_mask_improved_vs_no_language": macro_mask_no_language,
        "direction_wins": direction_wins,
        "case_wins": case_wins,
        "case_pass_count": int(case_pass_count),
        "case_total": len(case_wins),
        "criterion": "V3.2b language: conditioned < identity and shared-trunk no-language at micro/macro, correct < wrong-direction, 2/2 directions, at least 3/4 cases, and mask improves over both baselines.",
    })
    return summary


def joint_pass_summary(
    language_rows: list[dict[str, Any]],
    language_group_rows: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    visual_group_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    language = language_joint_pass_summary(language_rows, language_group_rows)
    visual = v2_pass_summary(visual_rows, visual_group_rows)
    return {
        "passed": bool(language["passed"] and visual["passed"]),
        "language": language,
        "visual": visual,
        "criterion": "V3.2b requires the new language gate and the complete V2 visual gate at the same checkpoint.",
    }


def train_joint(
    model: nn.Module,
    language_train: dict[str, Any],
    visual_train: dict[str, Any],
    language_eval_train: dict[str, Any],
    language_val: dict[str, Any],
    visual_val: dict[str, Any],
    v2_agnostic: nn.Module,
    interface: VoxTellStateInterface,
    selected_stage: str,
    max_steps: int,
    eval_steps: list[int],
    batch_size: int,
    language_lr: float,
    shared_lr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    language_parameters = list(model.language_action_encoder.parameters())
    language_parameter_ids = {id(parameter) for parameter in language_parameters}
    shared_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in language_parameter_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": language_parameters, "lr": language_lr},
        {"params": shared_parameters, "lr": shared_lr},
    ], weight_decay=1e-4)
    model.eval()
    language_rows: list[dict[str, Any]] = []
    language_group_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    visual_group_rows: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    eval_set = sorted(set([0, max_steps, *eval_steps]))

    def append_eval(step: int) -> None:
        rows, groups = collect_language_eval(
            step, language_eval_train, language_val, model, interface, selected_stage, batch_size,
        )
        language_rows.extend(rows)
        language_group_rows.extend(groups)
        rows, groups = collect_visual_eval(
            step, "val", visual_val, model, v2_agnostic, interface, selected_stage, batch_size,
        )
        visual_rows.extend(rows)
        visual_group_rows.extend(groups)
        checkpoints[step] = cpu_state_dict(model)

    append_eval(0)
    device = next(model.parameters()).device
    language_states = language_train["states"]
    language_targets = language_train["targets"]
    language_deltas = language_train["text_deltas"]
    visual_states = visual_train["states"]
    visual_targets = visual_train["targets"]
    visual_actions = visual_train["actions"]
    language_generator = torch.Generator().manual_seed(torch.initial_seed() + 17)
    visual_generator = torch.Generator().manual_seed(torch.initial_seed() + 31)
    language_permutation = torch.randperm(language_states.shape[0], generator=language_generator)
    visual_permutation = torch.randperm(visual_states.shape[0], generator=visual_generator)
    language_cursor = 0
    visual_cursor = 0

    for step in range(1, max_steps + 1):
        if language_cursor + batch_size > language_states.shape[0]:
            language_permutation = torch.randperm(language_states.shape[0], generator=language_generator)
            language_cursor = 0
        if visual_cursor + batch_size > visual_states.shape[0]:
            visual_permutation = torch.randperm(visual_states.shape[0], generator=visual_generator)
            visual_cursor = 0
        language_indices = language_permutation[language_cursor:language_cursor + batch_size]
        visual_indices = visual_permutation[visual_cursor:visual_cursor + batch_size]
        language_cursor += batch_size
        visual_cursor += batch_size
        language_state_batch = language_states.index_select(0, language_indices).to(device)
        language_target_batch = language_targets.index_select(0, language_indices).to(device)
        language_delta_batch = language_deltas.index_select(0, language_indices).to(device)
        visual_state_batch = visual_states.index_select(0, visual_indices).to(device)
        visual_target_batch = visual_targets.index_select(0, visual_indices).to(device)
        visual_action_batch = visual_actions.index_select(0, visual_indices).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        language_prediction = model(language_state_batch, text_delta=language_delta_batch)
        visual_prediction = model(visual_state_batch, action=visual_action_batch)
        loss = 0.5 * (
            normalized_mse(language_prediction, language_target_batch)
            + normalized_mse(visual_prediction, visual_target_batch)
        )
        loss.backward()
        optimizer.step()
        model.eval()
        if step in eval_set:
            append_eval(step)
    return language_rows, language_group_rows, visual_rows, visual_group_rows, checkpoints


def rows_at_step(rows: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["step"]) == step]


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
    train_cases = iter_cases(paths, split="train", limit=args.dev_cases)
    val_cases = iter_cases(paths, split="test", limit=args.val_cases)
    language_train = build_language_dataset(
        interface, train_cases, args.selected_stage, args.patches_per_case,
        args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
    )
    language_val = build_language_dataset(
        interface, val_cases, args.selected_stage, args.patches_per_case,
        args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
    )
    visual_values = {"gamma": list(args.gamma_strengths), "blur": list(args.blur_sigmas)}
    visual_train = build_visual_dataset(
        interface, train_cases, DEFAULT_PROMPTS, ["gamma", "blur"], visual_values,
        args.selected_stage, args.patches_per_case, args.foreground_patches_per_case,
        args.foreground_candidate_patches, args.foreground_threshold,
        include_identity_anchors=True,
    )
    visual_val = build_visual_dataset(
        interface, val_cases, DEFAULT_PROMPTS, ["gamma", "blur"], visual_values,
        args.selected_stage, args.patches_per_case, args.foreground_patches_per_case,
        args.foreground_candidate_patches, args.foreground_threshold,
    )
    prepare_functional_seg_head(interface, args.selected_stage)
    checkpoint_path = Path(args.v2_checkpoint)
    in_channels = int(language_train["states"].shape[1])
    joint_model = make_joint_model(
        checkpoint_path, in_channels, int(language_train["text_delta_dim"]), args.hidden_channels, device,
    )
    v2_agnostic = make_model(
        checkpoint_path, in_channels, int(language_train["text_delta_dim"]), args.hidden_channels, device,
        state_key="agnostic_state_dict",
    )
    language_rows, language_group_rows, visual_rows, visual_group_rows, checkpoints = train_joint(
        joint_model, language_train, visual_train, language_train, language_val, visual_val,
        v2_agnostic, interface, args.selected_stage, args.max_train_steps, args.eval_steps,
        args.batch_size, args.language_lr, args.shared_lr,
    )

    write_model_csv(output_dir / "language_training_curve.csv", language_rows)
    write_model_csv(output_dir / "language_grouped_training_curve.csv", language_group_rows)
    write_model_csv(output_dir / "visual_validation_curve.csv", visual_rows)
    write_model_csv(output_dir / "visual_grouped_validation_curve.csv", visual_group_rows)
    diagnostic_rows = []
    for split, data in [("train", language_train), ("val", language_val)]:
        diagnostic_rows.extend({"split": split, **row} for row in data["diagnostics"])
    write_csv(
        output_dir / "language_transition_diagnostics.csv", diagnostic_rows,
        ["split", "case", "patch_index", "patch_kind", "direction", "state_normalized_mse", "mask_logit_normalized_mse"],
    )

    step_summaries = {}
    for step in sorted({int(row["step"]) for row in language_rows}):
        step_summaries[str(step)] = joint_pass_summary(
            rows_at_step(language_rows, step), rows_at_step(language_group_rows, step),
            rows_at_step(visual_rows, step), rows_at_step(visual_group_rows, step),
        )
    passing_steps = [
        (int(step), float(summary["language"]["macro_conditioned_lt_identity"]))
        for step, summary in step_summaries.items()
        if summary["passed"]
    ]
    # Among jointly passing checkpoints, select the one with the lowest language val macro state error.
    if passing_steps:
        def language_macro_state(step: int) -> float:
            for row in language_rows:
                if int(row["step"]) == step and row["split"] == "val" and row["group"] == "overall_macro" and row["model"] == "language_conditioned":
                    return float(row["state_normalized_mse"])
            raise ValueError("Missing language macro state metric")
        selected_step = min((step for step, _ in passing_steps), key=lambda step: (language_macro_state(step), step))
    else:
        selected_step = args.max_train_steps

    checkpoint_paths = {}
    for step, state_dict in checkpoints.items():
        path = output_dir / f"unified_world_predictor_step{step}.pt"
        torch.save({
            "selected_step": step,
            "selected_stage": args.selected_stage,
            "hidden_channels": args.hidden_channels,
            "text_delta_dim": language_train["text_delta_dim"],
            "v2_checkpoint": str(checkpoint_path),
            "language_actions": [
                "liver -> the liver: E(the liver) - E(liver)",
                "the liver -> liver: E(liver) - E(the liver)",
            ],
            "language_encoder_zero_init": True,
            "joint_finetuning": True,
            "learning_rates": {"language_action_encoder": args.language_lr, "v2_parameters": args.shared_lr},
            "trainable_parameters": [name for name, parameter in joint_model.named_parameters() if parameter.requires_grad],
            "state_dict": state_dict,
            "args": vars(args),
        }, path)
        checkpoint_paths[str(step)] = str(path)
    best_path = checkpoint_paths.get(str(selected_step)) if step_summaries[str(selected_step)]["passed"] else None

    selected_language_rows = rows_at_step(language_rows, selected_step)
    selected_language_groups = rows_at_step(language_group_rows, selected_step)
    selected_visual_rows = rows_at_step(visual_rows, selected_step)
    selected_visual_groups = rows_at_step(visual_group_rows, selected_step)
    summary = {
        "args": vars(args),
        "train_cases": [case.case for case in train_cases],
        "val_cases": [case.case for case in val_cases],
        "selected_stage": args.selected_stage,
        "v2_checkpoint": str(checkpoint_path),
        "train_language_samples": len(language_train["case_ids"]),
        "train_visual_samples": len(visual_train["case_ids"]),
        "val_language_samples": len(language_val["case_ids"]),
        "val_visual_samples": len(visual_val["case_ids"]),
        "language_training_curve_csv": str(output_dir / "language_training_curve.csv"),
        "language_grouped_training_curve_csv": str(output_dir / "language_grouped_training_curve.csv"),
        "visual_validation_curve_csv": str(output_dir / "visual_validation_curve.csv"),
        "visual_grouped_validation_curve_csv": str(output_dir / "visual_grouped_validation_curve.csv"),
        "language_transition_diagnostics_csv": str(output_dir / "language_transition_diagnostics.csv"),
        "checkpoint_paths": checkpoint_paths,
        "selected_step": selected_step,
        "best_checkpoint_path": best_path,
        "selected_language_metrics": selected_language_rows,
        "selected_language_group_metrics": selected_language_groups,
        "selected_visual_metrics": selected_visual_rows,
        "selected_visual_group_metrics": selected_visual_groups,
        "step_pass_summaries": step_summaries,
        "scope_note": "V3.2b joint visual-language fine-tuning only; same V2 unified predictor, normalized MSE only, no new module or loss beyond the existing language encoder.",
    }
    (output_dir / "v3_2b_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
