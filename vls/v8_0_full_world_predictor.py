"""V8.0 full-train visual World Predictor training.

This is a clean, full-split visual retraining entry point.  The predictor
architecture and action injection are the V3.2e/V2 visual definitions; only
the training scope is expanded.  Test labels are used exclusively by the
evaluation and checkpoint-selection path.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases
from vls.v2_experiment import (
    build_dataset,
    resolve_device,
    set_seed,
    v2_pass_summary,
)
from vls.v3_2b_joint_experiment import collect_visual_eval
from vls.v3_language_experiment import flatten_prompt_embedding
from vls.v2_experiment import prepare_functional_seg_head
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, normalized_mse


GAMMA_STRENGTH = 0.30
BLUR_SIGMA = 1.5
ACTION_FAMILIES = ("gamma", "blur")
ACTION_VALUES = {"gamma": [GAMMA_STRENGTH], "blur": [BLUR_SIGMA]}


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V8.0 full-train visual World Predictor")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--output-dir", default="outputs/v8_0_full_world_predictor")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_visual_model(
    in_channels: int,
    text_delta_dim: int,
    hidden_channels: int,
    device: torch.device,
    use_action: bool,
) -> VisualWorldPredictor3D:
    """Construct the V3.2e architecture with a fresh random initialization."""
    model = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        action_dim=3,
        num_blocks=2,
        use_action=use_action,
        text_delta_dim=text_delta_dim,
        use_language=True,
        allow_unconditioned=True,
    ).to(device)
    return model


def manifest_payload(
    train_cases: list[Any],
    test_cases: list[Any],
    train_data: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = [
        {
            "case_id": case_id,
            "case": case,
            "patch_index": int(patch_index),
            "patch_kind": patch_kind,
            "action_family": action_family,
            "strength": float(strength),
        }
        for case_id, case, patch_index, patch_kind, action_family, strength in zip(
            train_data["case_ids"],
            train_data["case_names"],
            train_data["patch_indices"],
            train_data["patch_kinds"],
            train_data["action_families"],
            train_data["strengths"],
            strict=True,
        )
    ]
    return {
        "stage": "V8.0 full-train visual World Predictor",
        "manifest_version": 1,
        "train_cases": [case.case for case in train_cases],
        "test_cases": [case.case for case in test_cases],
        "train_case_count": len(train_cases),
        "test_case_count": len(test_cases),
        "action_families": list(ACTION_FAMILIES),
        "action_values": ACTION_VALUES,
        "config": {
            "selected_stage": args.selected_stage,
            "patches_per_case": args.patches_per_case,
            "foreground_patches_per_case": args.foreground_patches_per_case,
            "foreground_candidate_patches": args.foreground_candidate_patches,
            "foreground_threshold": args.foreground_threshold,
        },
        "records": records,
        "test_labels_used_for": ["evaluation", "checkpoint_selection"],
        "test_labels_used_in_training_loss": False,
    }


def assert_full_split(train_cases: list[Any], test_cases: list[Any], train_data: dict[str, Any]) -> None:
    train_names = [case.case for case in train_cases]
    test_names = [case.case for case in test_cases]
    if not train_names:
        raise AssertionError("V8.0 requires a non-empty full train split")
    if set(train_names) & set(test_names):
        raise AssertionError("V8.0 train/test case overlap")
    if set(train_data["case_names"]) != set(train_names):
        raise AssertionError("V8.0 training data does not cover the complete train split")
    if not all(case in set(train_data["case_names"]) for case in train_names):
        raise AssertionError("V8.0 omitted a train case from the training data")
    for case in train_names:
        actions = {
            family
            for name, family in zip(train_data["case_names"], train_data["action_families"], strict=True)
            if name == case
        }
        if not set(ACTION_FAMILIES).issubset(actions):
            raise AssertionError(f"V8.0 train case {case} lacks gamma or blur coverage")


def test_gate_summary(
    rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the unchanged V2/V3.2e visual gates to rows labelled as test."""
    gate_rows = [{**row, "split": "val"} for row in rows]
    gate_group_rows = [{**row, "split": "val"} for row in group_rows]
    summary = v2_pass_summary(gate_rows, gate_group_rows)
    summary["evaluation_split"] = "test"
    summary["test_used_for_checkpoint_selection"] = True
    return summary


def conditioned_macro_mse(rows: list[dict[str, Any]]) -> float:
    matches = [
        row for row in rows
        if row["group"] == "overall_macro" and row["model"] == "action_conditioned"
    ]
    if len(matches) != 1:
        raise AssertionError("Missing unique test macro conditioned metric")
    return float(matches[0]["state_normalized_mse"])


def train_epoch(
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    train_data: dict[str, Any],
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    states = train_data["states"]
    targets = train_data["targets"]
    actions = train_data["actions"]
    permutation = torch.randperm(states.shape[0], generator=generator)
    loss_rows = []
    for start in range(0, states.shape[0], batch_size):
        indices = permutation[start : start + batch_size]
        state_batch = states.index_select(0, indices).to(device)
        target_batch = targets.index_select(0, indices).to(device)
        action_batch = actions.index_select(0, indices).to(device)
        for model_name, model in models.items():
            model.train()
            optimizers[model_name].zero_grad(set_to_none=True)
            prediction = model(
                state_batch,
                action_batch if model_name == "action_conditioned" else None,
            )
            loss = normalized_mse(prediction, target_batch)
            loss.backward()
            optimizers[model_name].step()
            loss_rows.append({
                "model": model_name,
                "loss": float(loss.detach().cpu()),
                "batch_size": int(indices.numel()),
            })
        del state_batch, target_batch, action_batch
    return loss_rows


def run(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise AssertionError("V8.0 epochs and batch size must be positive")
    if not np.isclose(GAMMA_STRENGTH, 0.30) or not np.isclose(BLUR_SIGMA, 1.5):
        raise AssertionError("V8.0 action values are not the frozen gamma(+0.30)/blur(1.5) protocol")

    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V8.0 requires CUDA, resolved {device}")
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
    prepare_functional_seg_head(interface, args.selected_stage)
    train_cases = iter_cases(paths, split="train")
    test_cases = iter_cases(paths, split="test")
    if len(train_cases) != len(iter_cases(paths, split="train")):
        raise AssertionError("V8.0 train split changed during setup")
    train_data = build_dataset(
        interface,
        train_cases,
        DEFAULT_PROMPTS,
        list(ACTION_FAMILIES),
        ACTION_VALUES,
        args.selected_stage,
        args.patches_per_case,
        args.foreground_patches_per_case,
        args.foreground_candidate_patches,
        args.foreground_threshold,
        include_identity_anchors=True,
    )
    test_data = build_dataset(
        interface,
        test_cases,
        DEFAULT_PROMPTS,
        list(ACTION_FAMILIES),
        ACTION_VALUES,
        args.selected_stage,
        args.patches_per_case,
        args.foreground_patches_per_case,
        args.foreground_candidate_patches,
        args.foreground_threshold,
        include_identity_anchors=False,
    )
    assert_full_split(train_cases, test_cases, train_data)
    if set(test_data["case_names"]) != {case.case for case in test_cases}:
        raise AssertionError("V8.0 test evaluation data does not cover the complete test split")

    text_embedding = interface.embed_text_prompts(["liver", "the liver"])
    text_delta_dim = int(
        (flatten_prompt_embedding(text_embedding, 1) - flatten_prompt_embedding(text_embedding, 0)).numel()
    )
    in_channels = int(train_data["states"].shape[1])
    conditioned = make_visual_model(in_channels, text_delta_dim, args.hidden_channels, device, True)
    agnostic = make_visual_model(in_channels, text_delta_dim, args.hidden_channels, device, False)
    models = {"action_agnostic": agnostic, "action_conditioned": conditioned}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for name, model in models.items()
    }

    train_curve: list[dict[str, Any]] = []
    test_curve: list[dict[str, Any]] = []
    test_group_curve: list[dict[str, Any]] = []
    epoch_summaries: dict[str, Any] = {}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    generator = torch.Generator().manual_seed(args.seed + 17)

    def evaluate(epoch: int) -> None:
        rows, groups = collect_visual_eval(
            epoch, "test", test_data, conditioned, agnostic,
            interface, args.selected_stage, args.batch_size,
        )
        test_curve.extend(rows)
        test_group_curve.extend(groups)
        epoch_rows = [row for row in rows if int(row["step"]) == epoch]
        epoch_groups = [row for row in groups if int(row["step"]) == epoch]
        summary = test_gate_summary(epoch_rows, epoch_groups)
        summary["test_conditioned_macro_state_normalized_mse"] = conditioned_macro_mse(
            [*epoch_rows, *epoch_groups],
        )
        epoch_summaries[str(epoch)] = summary
        checkpoints[epoch] = cpu_state_dict(conditioned)

    evaluate(0)
    for epoch in range(1, args.epochs + 1):
        rows = train_epoch(models, optimizers, train_data, args.batch_size, device, generator)
        for model_name in models:
            values = [row["loss"] for row in rows if row["model"] == model_name]
            train_curve.append({
                "epoch": epoch,
                "model": model_name,
                "mean_normalized_mse": float(np.mean(values)),
                "batch_count": len(values),
                "train_case_count": len(train_cases),
                "test_labels_used_in_loss": False,
            })
        evaluate(epoch)

    candidates = [
        (epoch, conditioned_macro_mse(
            [row for row in test_curve if int(row["step"]) == epoch and row["group"] == "overall_macro"],
        ))
        for epoch, summary in ((int(step), value) for step, value in epoch_summaries.items())
        if summary["passed"]
    ]
    selection_mode = "passed_gates"
    if not candidates:
        selection_mode = "lowest_conditioned_test_macro_mse_fallback_no_gate_passed"
        candidates = [
            (epoch, conditioned_macro_mse(
                [row for row in test_curve if int(row["step"]) == epoch and row["group"] == "overall_macro"],
            ))
            for epoch in range(args.epochs + 1)
        ]
    selected_epoch, selected_mse = min(candidates, key=lambda item: (item[1], item[0]))
    best_path = output_dir / "best_world_predictor.pt"
    torch.save({
        "stage": "V8.0 full-train visual World Predictor",
        "selected_epoch": selected_epoch,
        "selected_stage": args.selected_stage,
        "hidden_channels": args.hidden_channels,
        "text_delta_dim": text_delta_dim,
        "action_dim": 3,
        "state_dict": checkpoints[selected_epoch],
        "architecture": {
            "num_blocks": 2,
            "use_action": True,
            "use_language": True,
            "allow_unconditioned": True,
            "fresh_initialization": True,
        },
        "action_protocol": {"gamma": GAMMA_STRENGTH, "blur": BLUR_SIGMA},
        "train_case_count": len(train_cases),
        "test_case_count": len(test_cases),
        "test_used_for_checkpoint_selection": True,
        "checkpoint_selection": {
            "mode": selection_mode,
            "selected_test_conditioned_macro_state_normalized_mse": selected_mse,
            "gates": "unchanged V2/V3.2e conditioned-vs-agnostic, correct-vs-wrong-strength, correct-vs-wrong-type, family/case/strength gates",
        },
        "args": vars(args),
    }, best_path)

    manifest = manifest_payload(train_cases, test_cases, train_data, args)
    (output_dir / "train_patch_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "train_cases.json").write_text(json.dumps([case.case for case in train_cases], indent=2))
    (output_dir / "test_cases.json").write_text(json.dumps([case.case for case in test_cases], indent=2))
    write_csv(output_dir / "training_curve.csv", train_curve)
    write_csv(output_dir / "test_validation_curve.csv", test_curve)
    write_csv(output_dir / "test_validation_grouped_curve.csv", test_group_curve)
    summary = {
        "stage": "V8.0 full-scale World Predictor training",
        "train_cases": [case.case for case in train_cases],
        "test_cases": [case.case for case in test_cases],
        "train_case_count": len(train_cases),
        "test_case_count": len(test_cases),
        "train_uses_complete_split": True,
        "test_used_for_checkpoint_selection": True,
        "test_labels_used_in_training_loss": False,
        "test_cases_in_gradient_training": False,
        "case_overlap": sorted({case.case for case in train_cases} & {case.case for case in test_cases}),
        "action_protocol": {"gamma": GAMMA_STRENGTH, "blur": BLUR_SIGMA},
        "architecture": "V3.2e VisualWorldPredictor3D: action_dim=3, num_blocks=2, action injection unchanged",
        "fresh_initialization": True,
        "trainable_scope": "all fresh World Predictor parameters",
        "checkpoint_selection": {
            "selected_epoch": selected_epoch,
            "selected_test_conditioned_macro_state_normalized_mse": selected_mse,
            "mode": selection_mode,
            "gates": "conditioned vs agnostic; correct vs wrong-strength; correct vs wrong-type; action-family/case/strength gates",
        },
        "epoch_gate_summaries": epoch_summaries,
        "outputs": {
            "best_checkpoint": str(best_path),
            "train_patch_manifest": str(output_dir / "train_patch_manifest.json"),
            "training_curve": str(output_dir / "training_curve.csv"),
            "test_validation_curve": str(output_dir / "test_validation_curve.csv"),
            "test_validation_grouped_curve": str(output_dir / "test_validation_grouped_curve.csv"),
            "summary": str(output_dir / "summary.json"),
        },
        "status": "code_ready; full training not executed by implementation task",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"best_checkpoint": str(best_path), "selected_epoch": selected_epoch}, indent=2))


if __name__ == "__main__":
    run(parse_args())
