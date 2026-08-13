from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases
from vls.v2_experiment import (
    build_dataset as build_visual_dataset,
    prepare_functional_seg_head,
    resolve_device,
    v2_pass_summary,
)
from vls.v3_2a_unified_experiment import (
    DEFAULT_BLUR_SIGMAS,
    DEFAULT_GAMMA_STRENGTHS,
    build_language_dataset,
    collect_language_eval,
    cpu_state_dict,
    set_seed,
    write_csv,
)
from vls.v3_2b_joint_experiment import (
    collect_visual_eval,
    language_joint_pass_summary,
    make_model,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, normalized_mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3.2e joint training with frozen input projection.")
    paths = ProjectPaths()
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--v2-checkpoint", default="outputs/v2_final/world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v3_2e_frozen_input_projection")
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
    parser.add_argument("--eval-steps", nargs="+", type=int, default=[0, 50, 100, 150, 200])
    parser.add_argument("--gamma-strengths", nargs="+", type=float, default=DEFAULT_GAMMA_STRENGTHS)
    parser.add_argument("--blur-sigmas", nargs="+", type=float, default=DEFAULT_BLUR_SIGMAS)
    parser.add_argument("--language-lr", type=float, default=1e-3)
    parser.add_argument("--shared-lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def write_model_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["step", "split", "group", "group_value", "model", "state_normalized_mse", "mask_logit_normalized_mse"]
    write_csv(path, rows, fields)


def load_v3_2e_model(
    checkpoint_path: Path,
    in_channels: int,
    text_delta_dim: int,
    hidden_channels: int,
    device: torch.device,
) -> VisualWorldPredictor3D:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["hidden_channels"]) != hidden_channels:
        raise ValueError("V2 checkpoint hidden_channels does not match the requested model")
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
        parameter.requires_grad = (
            name.startswith("language_action_encoder.")
            or name.startswith("action_mlp.")
            or name.startswith("blocks.")
            or name.startswith("output_projection.")
        )
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if frozen != ["input_projection.weight", "input_projection.bias"]:
        raise RuntimeError(f"V3.2e must freeze only input_projection, got frozen={frozen}")
    expected_trainable = [
        name for name, _ in model.named_parameters()
        if name.startswith(("language_action_encoder.", "action_mlp.", "blocks.", "output_projection."))
    ]
    if trainable != expected_trainable:
        raise RuntimeError(f"Unexpected V3.2e trainable parameters: {trainable}")
    model.eval()
    return model


def train_v3_2e(
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
    shared_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("language_action_encoder.")
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


def joint_summary_at_step(
    language_rows: list[dict[str, Any]],
    language_group_rows: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    visual_group_rows: list[dict[str, Any]],
    step: int,
) -> dict[str, Any]:
    language = language_joint_pass_summary(
        rows_at_step(language_rows, step), rows_at_step(language_group_rows, step),
    )
    visual = v2_pass_summary(
        rows_at_step(visual_rows, step), rows_at_step(visual_group_rows, step),
    )
    return {
        "passed": bool(language["passed"] and visual["passed"]),
        "language": language,
        "visual": visual,
    }


def language_macro_state(rows: list[dict[str, Any]], step: int) -> float:
    for row in rows:
        if (
            int(row["step"]) == step
            and row["split"] == "val"
            and row["group"] == "overall_macro"
            and row["model"] == "language_conditioned"
        ):
            return float(row["state_normalized_mse"])
    raise ValueError(f"Missing language macro state metric at step {step}")


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
        args.foreground_candidate_patches, args.foreground_threshold, include_identity_anchors=True,
    )
    visual_val = build_visual_dataset(
        interface, val_cases, DEFAULT_PROMPTS, ["gamma", "blur"], visual_values,
        args.selected_stage, args.patches_per_case, args.foreground_patches_per_case,
        args.foreground_candidate_patches, args.foreground_threshold,
    )
    prepare_functional_seg_head(interface, args.selected_stage)
    checkpoint_path = Path(args.v2_checkpoint)
    base_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_v3_2e_model(
        checkpoint_path, int(language_train["states"].shape[1]), int(language_train["text_delta_dim"]),
        args.hidden_channels, device,
    )
    v2_agnostic = make_model(
        checkpoint_path, int(language_train["states"].shape[1]), int(language_train["text_delta_dim"]),
        args.hidden_channels, device, state_key="agnostic_state_dict",
    )
    language_rows, language_group_rows, visual_rows, visual_group_rows, checkpoints = train_v3_2e(
        model, language_train, visual_train, language_train, language_val, visual_val,
        v2_agnostic, interface, args.selected_stage, args.max_train_steps, args.eval_steps,
        args.batch_size, args.language_lr, args.shared_lr,
    )

    write_model_csv(output_dir / "language_training_curve.csv", language_rows)
    write_model_csv(output_dir / "language_grouped_training_curve.csv", language_group_rows)
    write_model_csv(output_dir / "visual_validation_curve.csv", visual_rows)
    write_model_csv(output_dir / "visual_grouped_validation_curve.csv", visual_group_rows)
    diagnostics = []
    for split, data in [("train", language_train), ("val", language_val)]:
        diagnostics.extend({"split": split, **row} for row in data["diagnostics"])
    write_csv(
        output_dir / "language_transition_diagnostics.csv", diagnostics,
        ["split", "case", "patch_index", "patch_kind", "direction", "state_normalized_mse", "mask_logit_normalized_mse"],
    )

    eval_steps = sorted(set([0, args.max_train_steps, *args.eval_steps]))
    step_summaries = {
        str(step): joint_summary_at_step(language_rows, language_group_rows, visual_rows, visual_group_rows, step)
        for step in eval_steps
    }
    passing_steps = [
        step for step in eval_steps if step_summaries[str(step)]["passed"]
    ]
    selected_step = min(passing_steps, key=lambda step: (language_macro_state(language_rows, step), step)) if passing_steps else None

    checkpoint_paths = {}
    for step in eval_steps:
        if step not in checkpoints:
            continue
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
            "input_projection_frozen": True,
            "joint_finetuning": True,
            "learning_rates": {
                "language_action_encoder": args.language_lr,
                "action_mlp_blocks_output_projection": args.shared_lr,
            },
            "trainable_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
            "frozen_parameters": [name for name, parameter in model.named_parameters() if not parameter.requires_grad],
            "state_dict": checkpoints[step],
            "args": vars(args),
        }, path)
        checkpoint_paths[str(step)] = str(path)
    best_path = checkpoint_paths.get(str(selected_step)) if selected_step is not None else None

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
        "input_projection_frozen": True,
        "language_encoder_lr": args.language_lr,
        "shared_lr": args.shared_lr,
        "loss": "0.5 * (language normalized MSE + visual normalized MSE)",
        "checkpoint_paths": checkpoint_paths,
        "joint_passing_steps": passing_steps,
        "selected_step": selected_step,
        "best_checkpoint_path": best_path,
        "language_training_curve_csv": str(output_dir / "language_training_curve.csv"),
        "language_grouped_training_curve_csv": str(output_dir / "language_grouped_training_curve.csv"),
        "visual_validation_curve_csv": str(output_dir / "visual_validation_curve.csv"),
        "visual_grouped_validation_curve_csv": str(output_dir / "visual_grouped_validation_curve.csv"),
        "language_transition_diagnostics_csv": str(output_dir / "language_transition_diagnostics.csv"),
        "step_pass_summaries": step_summaries,
        "scope_note": "V3.2e clean confirmation: initialized from V2, input_projection permanently frozen, language/action/trunk/output jointly fine-tuned with normalized MSE only; checkpoint candidates require both strict language and V2 visual gates.",
    }
    (output_dir / "v3_2e_summary.json").write_text(json.dumps(summary, indent=2))
    compact = {
        "summary_path": str(output_dir / "v3_2e_summary.json"),
        "joint_passing_steps": passing_steps,
        "selected_step": selected_step,
        "best_checkpoint_path": best_path,
        "steps": {
            step: {
                "passed": step_summaries[str(step)]["passed"],
                "language_passed": step_summaries[str(step)]["language"]["passed"],
                "visual_passed": step_summaries[str(step)]["visual"]["passed"],
                "language_macro_conditioned_state": language_macro_state(language_rows, step),
                "visual_strength_win_count": step_summaries[str(step)]["visual"]["strength_win_count"],
                "visual_case_win_count": step_summaries[str(step)]["visual"]["case_win_count"],
            }
            for step in eval_steps
        },
    }
    print(json.dumps(compact, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
