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

from vls.config import ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import (
    prepare_functional_seg_head,
    resolve_device,
    select_patch_slicers,
    state_to_intermediate_prediction,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import LanguageWorldPredictor3D, normalized_mse


SOURCE_PROMPT = "liver"
TARGET_PROMPT = "the liver"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3.1 minimal language world prediction.")
    parser.add_argument("--model-dir", default=str(ProjectPaths().voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(ProjectPaths().voxtell_root))
    parser.add_argument("--data-root", default=str(ProjectPaths().data_root))
    parser.add_argument("--split-json", default=str(ProjectPaths().split_json))
    parser.add_argument("--output-dir", default="outputs/v3_1_language")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--dev-cases", type=int, default=8)
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=300)
    parser.add_argument("--eval-steps", nargs="+", type=int, default=[0, 100, 200, 300])
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


def grouped_indices(data: dict[str, Any], key: str) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(data[key]):
        groups.setdefault(str(value), []).append(index)
    return sorted(groups.items(), key=lambda item: item[0])


def subset_data(data: dict[str, Any], indices: list[int]) -> dict[str, Any]:
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


@torch.inference_mode()
def build_dataset(
    interface: VoxTellStateInterface,
    cases: list[Any],
    selected_stage: str,
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
) -> dict[str, Any]:
    prompts = [SOURCE_PROMPT, TARGET_PROMPT]
    text_embedding = interface.embed_text_prompts(prompts)
    liver_to_the_liver = (
        flatten_prompt_embedding(text_embedding, 1) - flatten_prompt_embedding(text_embedding, 0)
    ).cpu()
    the_liver_to_liver = -liver_to_the_liver

    states = []
    targets = []
    text_deltas = []
    target_predictions = []
    case_ids = []
    case_names = []
    patch_indices = []
    patch_kinds_all = []
    directions = []
    diagnostics = []
    for case in cases:
        image, _, _ = read_image_and_label(case)
        padded, slicers, patch_kinds = select_patch_slicers(
            interface,
            image,
            prompts,
            patches_per_case,
            foreground_patches_per_case,
            foreground_candidate_patches,
            foreground_threshold,
        )
        for patch_index, slicer in enumerate(slicers):
            patch = torch.clone(padded[slicer][None], memory_format=torch.contiguous_format)
            result = interface.forward_with_states(patch, text_embedding)
            stage_states = result["decoder_states"][selected_stage]
            stage_predictions = result["intermediate_predictions"][selected_stage]
            liver_state = stage_states[:, 0].detach().float()
            the_liver_state = stage_states[:, 1].detach().float()
            liver_prediction = stage_predictions[:, 0:1].detach().float()
            the_liver_prediction = stage_predictions[:, 1:2].detach().float()
            examples = [
                (
                    "liver_to_the_liver",
                    liver_state,
                    the_liver_state,
                    liver_to_the_liver,
                    the_liver_prediction,
                ),
                (
                    "the_liver_to_liver",
                    the_liver_state,
                    liver_state,
                    the_liver_to_liver,
                    liver_prediction,
                ),
            ]
            for direction, source_state, target_state, delta, target_prediction in examples:
                states.append(source_state.cpu())
                targets.append(target_state.cpu())
                text_deltas.append(delta[None])
                target_predictions.append(target_prediction.cpu())
                case_ids.append(f"{case.case}:patch{patch_index}:{direction}")
                case_names.append(case.case)
                patch_indices.append(patch_index)
                patch_kinds_all.append(patch_kinds[patch_index])
                directions.append(direction)
                diagnostics.append({
                    "case": case.case,
                    "patch_index": patch_index,
                    "patch_kind": patch_kinds[patch_index],
                    "direction": direction,
                    "state_normalized_mse": float(normalized_mse(source_state, target_state).detach().cpu()),
                    "mask_logit_normalized_mse": float(
                        normalized_mse(
                            liver_prediction if direction == "liver_to_the_liver" else the_liver_prediction,
                            target_prediction,
                        ).detach().cpu()
                    ),
                })
    return {
        "states": torch.cat(states, dim=0),
        "targets": torch.cat(targets, dim=0),
        "text_deltas": torch.cat(text_deltas, dim=0),
        "target_predictions": torch.cat(target_predictions, dim=0),
        "case_ids": case_ids,
        "case_names": case_names,
        "patch_indices": patch_indices,
        "patch_kinds": patch_kinds_all,
        "directions": directions,
        "diagnostics": diagnostics,
        "text_delta_dim": int(liver_to_the_liver.numel()),
        "text_delta_norm": float(torch.linalg.vector_norm(liver_to_the_liver).item()),
    }


def make_predictors(
    in_channels: int,
    text_delta_dim: int,
    hidden_channels: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    conditioned = LanguageWorldPredictor3D(
        in_channels=in_channels,
        text_delta_dim=text_delta_dim,
        hidden_channels=hidden_channels,
        use_language=True,
    ).to(device)
    agnostic = LanguageWorldPredictor3D(
        in_channels=in_channels,
        text_delta_dim=text_delta_dim,
        hidden_channels=hidden_channels,
        use_language=False,
    ).to(device)
    conditioned_state = conditioned.state_dict()
    agnostic_state = agnostic.state_dict()
    shared_state = {
        key: value
        for key, value in conditioned_state.items()
        if key in agnostic_state and agnostic_state[key].shape == value.shape
    }
    agnostic.load_state_dict({**agnostic_state, **deepcopy(shared_state)})
    return agnostic, conditioned


@torch.inference_mode()
def evaluate_identity(
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
        state_sse += float((state_batch.float() - target_batch.float()).pow(2).sum().detach().cpu())
        state_target_sq += float(target_batch.float().pow(2).sum().detach().cpu())
        mask_sse += float((pred_logits.float() - target_prediction_batch.float()).pow(2).sum().detach().cpu())
        mask_target_sq += float(target_prediction_batch.float().pow(2).sum().detach().cpu())
    return {
        "state_normalized_mse": state_sse / max(state_target_sq, 1e-6),
        "mask_logit_normalized_mse": mask_sse / max(mask_target_sq, 1e-6),
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data: dict[str, Any],
    use_language: bool,
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
    wrong_direction: bool = False,
) -> dict[str, float]:
    states = data["states"]
    targets = data["targets"]
    text_deltas = data["text_deltas"]
    target_predictions = data["target_predictions"]
    device = next(model.parameters()).device
    state_sse = 0.0
    state_target_sq = 0.0
    mask_sse = 0.0
    mask_target_sq = 0.0
    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        state_batch = states[start:end].to(device)
        target_batch = targets[start:end].to(device)
        delta_batch = text_deltas[start:end].to(device)
        if wrong_direction:
            delta_batch = -delta_batch
        target_prediction_batch = target_predictions[start:end].to(device)
        prediction = model(state_batch, delta_batch if use_language else None)
        pred_logits = state_to_intermediate_prediction(interface, selected_stage, prediction)
        state_sse += float((prediction.float() - target_batch.float()).pow(2).sum().detach().cpu())
        state_target_sq += float(target_batch.float().pow(2).sum().detach().cpu())
        mask_sse += float((pred_logits.float() - target_prediction_batch.float()).pow(2).sum().detach().cpu())
        mask_target_sq += float(target_prediction_batch.float().pow(2).sum().detach().cpu())
    return {
        "state_normalized_mse": state_sse / max(state_target_sq, 1e-6),
        "mask_logit_normalized_mse": mask_sse / max(mask_target_sq, 1e-6),
    }


@torch.inference_mode()
def compute_eval_records(
    step: int,
    split_name: str,
    group: str,
    group_value: str,
    data: dict[str, Any],
    models: dict[str, nn.Module],
    use_language: dict[str, bool],
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> list[dict[str, float | int | str]]:
    records = [{
        "step": step,
        "split": split_name,
        "group": group,
        "group_value": group_value,
        "model": "identity",
        **evaluate_identity(data, interface, selected_stage, batch_size),
    }]
    for model_name, model in models.items():
        records.append({
            "step": step,
            "split": split_name,
            "group": group,
            "group_value": group_value,
            "model": model_name,
            **evaluate_model(model, data, use_language[model_name], interface, selected_stage, batch_size),
        })
    records.append({
        "step": step,
        "split": split_name,
        "group": group,
        "group_value": group_value,
        "model": "language_conditioned_wrong_direction",
        **evaluate_model(
            models["language_conditioned"],
            data,
            True,
            interface,
            selected_stage,
            batch_size,
            wrong_direction=True,
        ),
    })
    return records


def macro_records_from_case_records(
    step: int,
    split_name: str,
    case_records: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    models = sorted({str(row["model"]) for row in case_records})
    rows = []
    for model_name in models:
        model_rows = [row for row in case_records if row["model"] == model_name]
        rows.append({
            "step": step,
            "split": split_name,
            "group": "overall_macro",
            "group_value": "case_mean",
            "model": model_name,
            "state_normalized_mse": float(np.mean([float(row["state_normalized_mse"]) for row in model_rows])),
            "mask_logit_normalized_mse": float(np.mean([float(row["mask_logit_normalized_mse"]) for row in model_rows])),
        })
    return rows


def records_by_model(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int | str]]:
    return {str(row["model"]): row for row in rows}


def pass_summary(
    final_rows: list[dict[str, float | int | str]],
    final_group_rows: list[dict[str, float | int | str]],
) -> dict[str, Any]:
    val_micro = records_by_model([
        row for row in final_rows
        if row["split"] == "val" and row["group"] == "overall_micro"
    ])
    val_macro = records_by_model([
        row for row in final_rows
        if row["split"] == "val" and row["group"] == "overall_macro"
    ])

    def state(rows: dict[str, dict[str, float | int | str]], model: str) -> float:
        return float(rows[model]["state_normalized_mse"])

    micro_conditioned_lt_agnostic = state(val_micro, "language_conditioned") < state(val_micro, "language_agnostic")
    micro_correct_lt_wrong = (
        state(val_micro, "language_conditioned") < state(val_micro, "language_conditioned_wrong_direction")
    )
    macro_conditioned_lt_agnostic = state(val_macro, "language_conditioned") < state(val_macro, "language_agnostic")
    macro_correct_lt_wrong = (
        state(val_macro, "language_conditioned") < state(val_macro, "language_conditioned_wrong_direction")
    )
    case_wins = []
    for case_name in sorted({
        str(row["group_value"])
        for row in final_group_rows
        if row["split"] == "val" and row["group"] == "case_names"
    }):
        rows = records_by_model([
            row for row in final_group_rows
            if row["split"] == "val" and row["group"] == "case_names" and row["group_value"] == case_name
        ])
        case_wins.append({
            "case": case_name,
            "conditioned_lt_agnostic": state(rows, "language_conditioned") < state(rows, "language_agnostic"),
            "correct_lt_wrong_direction": (
                state(rows, "language_conditioned") < state(rows, "language_conditioned_wrong_direction")
            ),
        })
    direction_wins = []
    for direction in sorted({
        str(row["group_value"])
        for row in final_group_rows
        if row["split"] == "val" and row["group"] == "directions"
    }):
        rows = records_by_model([
            row for row in final_group_rows
            if row["split"] == "val" and row["group"] == "directions" and row["group_value"] == direction
        ])
        direction_wins.append({
            "direction": direction,
            "conditioned_lt_agnostic": state(rows, "language_conditioned") < state(rows, "language_agnostic"),
            "correct_lt_wrong_direction": (
                state(rows, "language_conditioned") < state(rows, "language_conditioned_wrong_direction")
            ),
        })
    return {
        "passed": (
            micro_conditioned_lt_agnostic
            and micro_correct_lt_wrong
            and macro_conditioned_lt_agnostic
            and macro_correct_lt_wrong
        ),
        "micro_conditioned_lt_agnostic": micro_conditioned_lt_agnostic,
        "micro_correct_lt_wrong_direction": micro_correct_lt_wrong,
        "macro_conditioned_lt_agnostic": macro_conditioned_lt_agnostic,
        "macro_correct_lt_wrong_direction": macro_correct_lt_wrong,
        "case_wins": case_wins,
        "direction_wins": direction_wins,
        "criterion": "V3.1 sanity: unseen micro/macro conditioned < agnostic and correct delta < wrong-direction.",
    }


def val_macro_conditioned_state(
    rows: list[dict[str, float | int | str]],
) -> float:
    for row in rows:
        if (
            row["split"] == "val"
            and row["group"] == "overall_macro"
            and row["model"] == "language_conditioned"
        ):
            return float(row["state_normalized_mse"])
    raise ValueError("Missing val overall_macro language_conditioned metric")


def pass_summaries_by_step(
    curve_rows: list[dict[str, float | int | str]],
    group_rows: list[dict[str, float | int | str]],
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for step in sorted({int(row["step"]) for row in curve_rows}):
        step_rows = [row for row in curve_rows if int(row["step"]) == step]
        step_group_rows = [row for row in group_rows if int(row["step"]) == step]
        summary = pass_summary(step_rows, step_group_rows)
        summary["val_macro_conditioned_state_normalized_mse"] = val_macro_conditioned_state(step_rows)
        summaries[str(step)] = summary
    return summaries


def select_language_checkpoint_step(
    step_summaries: dict[str, dict[str, Any]],
    fallback_step: int,
) -> int:
    passing_steps = [
        (int(step), float(summary["val_macro_conditioned_state_normalized_mse"]))
        for step, summary in step_summaries.items()
        if summary["passed"]
    ]
    if not passing_steps:
        return fallback_step
    return min(passing_steps, key=lambda item: (item[1], item[0]))[0]


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_models(
    agnostic: nn.Module,
    conditioned: nn.Module,
    train_data: dict[str, Any],
    train_eval_data: dict[str, Any],
    val_data: dict[str, Any],
    interface: VoxTellStateInterface,
    selected_stage: str,
    max_steps: int,
    eval_steps: list[int],
    batch_size: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], dict[int, dict[str, Any]]]:
    optimizers = {
        "language_agnostic": torch.optim.AdamW(agnostic.parameters(), lr=1e-3, weight_decay=1e-4),
        "language_conditioned": torch.optim.AdamW(conditioned.parameters(), lr=1e-3, weight_decay=1e-4),
    }
    models = {"language_agnostic": agnostic, "language_conditioned": conditioned}
    use_language = {"language_agnostic": False, "language_conditioned": True}
    eval_set = sorted(set([0, max_steps, *eval_steps]))
    rows: list[dict[str, float | int | str]] = []
    group_rows: list[dict[str, float | int | str]] = []
    checkpoints: dict[int, dict[str, Any]] = {}

    def append_eval(step: int) -> None:
        for split_name, data in [("train", train_eval_data), ("val", val_data)]:
            rows.extend(compute_eval_records(
                step,
                split_name,
                "overall_micro",
                "all",
                data,
                models,
                use_language,
                interface,
                selected_stage,
                batch_size,
            ))
            case_records_for_macro = []
            for group_name in ["directions", "case_names"]:
                for group_value, indices in grouped_indices(data, group_name):
                    records = compute_eval_records(
                        step,
                        split_name,
                        group_name,
                        group_value,
                        subset_data(data, indices),
                        models,
                        use_language,
                        interface,
                        selected_stage,
                        batch_size,
                    )
                    group_rows.extend(records)
                    if group_name == "case_names":
                        case_records_for_macro.extend(records)
            rows.extend(macro_records_from_case_records(step, split_name, case_records_for_macro))
        checkpoints[step] = {
            "language_agnostic": cpu_state_dict(agnostic),
            "language_conditioned": cpu_state_dict(conditioned),
        }

    append_eval(0)
    device = next(agnostic.parameters()).device
    states = train_data["states"]
    targets = train_data["targets"]
    text_deltas = train_data["text_deltas"]
    num_samples = states.shape[0]
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    permutation = torch.randperm(num_samples, generator=generator)
    cursor = 0
    for step in range(1, max_steps + 1):
        if cursor + batch_size > num_samples:
            permutation = torch.randperm(num_samples, generator=generator)
            cursor = 0
        index_tensor = permutation[cursor : cursor + batch_size]
        cursor += batch_size
        state_batch = states.index_select(0, index_tensor).to(device)
        target_batch = targets.index_select(0, index_tensor).to(device)
        delta_batch = text_deltas.index_select(0, index_tensor).to(device)
        for model_name, model in models.items():
            model.train()
            optimizers[model_name].zero_grad(set_to_none=True)
            prediction = model(state_batch, delta_batch if use_language[model_name] else None)
            loss = normalized_mse(prediction, target_batch)
            loss.backward()
            optimizers[model_name].step()
            model.eval()
        if step in eval_set:
            append_eval(step)
    return rows, group_rows, checkpoints


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    val_cases = iter_cases(paths, split="test", limit=args.val_cases) if args.val_cases else train_cases
    train_eval_data = build_dataset(
        interface,
        train_cases,
        args.selected_stage,
        args.patches_per_case,
        args.foreground_patches_per_case,
        args.foreground_candidate_patches,
        args.foreground_threshold,
    )
    train_data = train_eval_data
    val_data = build_dataset(
        interface,
        val_cases,
        args.selected_stage,
        args.patches_per_case,
        args.foreground_patches_per_case,
        args.foreground_candidate_patches,
        args.foreground_threshold,
    )
    prepare_functional_seg_head(interface, args.selected_stage)

    agnostic, conditioned = make_predictors(
        in_channels=train_data["states"].shape[1],
        text_delta_dim=train_data["text_delta_dim"],
        hidden_channels=args.hidden_channels,
        device=device,
    )
    curve_rows, group_rows, checkpoints = train_models(
        agnostic,
        conditioned,
        train_data,
        train_eval_data,
        val_data,
        interface,
        args.selected_stage,
        args.max_train_steps,
        args.eval_steps,
        args.batch_size,
    )

    curve_path = output_dir / "training_curve.csv"
    fieldnames = [
        "step",
        "split",
        "group",
        "group_value",
        "model",
        "state_normalized_mse",
        "mask_logit_normalized_mse",
    ]
    write_csv(curve_path, curve_rows, fieldnames)
    group_curve_path = output_dir / "grouped_training_curve.csv"
    write_csv(group_curve_path, group_rows, fieldnames)

    diagnostic_rows = []
    for split_name, data in [("train", train_data), ("val", val_data)]:
        for row in data["diagnostics"]:
            diagnostic_rows.append({"split": split_name, **row})
    diagnostic_path = output_dir / "transition_diagnostics.csv"
    write_csv(
        diagnostic_path,
        diagnostic_rows,
        [
            "split",
            "case",
            "patch_index",
            "patch_kind",
            "direction",
            "state_normalized_mse",
            "mask_logit_normalized_mse",
        ],
    )

    step_summaries = pass_summaries_by_step(curve_rows, group_rows)
    selected_step = select_language_checkpoint_step(step_summaries, args.max_train_steps)
    final_rows = [row for row in curve_rows if int(row["step"]) == selected_step]
    final_group_rows = [row for row in group_rows if int(row["step"]) == selected_step]
    final_pass_summary = step_summaries[str(selected_step)]
    checkpoint_path = output_dir / f"language_world_predictor_step{selected_step}.pt"
    torch.save(
        {
            "selected_step": selected_step,
            "selected_stage": args.selected_stage,
            "hidden_channels": args.hidden_channels,
            "text_delta_dim": train_data["text_delta_dim"],
            "language_actions": [
                "liver -> the liver: delta_text = E(the liver) - E(liver)",
                "the liver -> liver: delta_text = E(liver) - E(the liver)",
            ],
            "conditioned_state_dict": checkpoints[selected_step]["language_conditioned"],
            "agnostic_state_dict": checkpoints[selected_step]["language_agnostic"],
            "args": vars(args),
        },
        checkpoint_path,
    )
    summary = {
        "args": vars(args),
        "train_cases": [case.case for case in train_cases],
        "val_cases": [case.case for case in val_cases],
        "selected_stage": args.selected_stage,
        "train_samples": len(train_data["case_ids"]),
        "val_samples": len(val_data["case_ids"]),
        "text_delta_dim": train_data["text_delta_dim"],
        "text_delta_norm": train_data["text_delta_norm"],
        "training_curve_csv": str(curve_path),
        "grouped_training_curve_csv": str(group_curve_path),
        "transition_diagnostics_csv": str(diagnostic_path),
        "checkpoint_path": str(checkpoint_path),
        "selected_step": selected_step,
        "final_metrics": final_rows,
        "final_group_metrics": final_group_rows,
        "step_pass_summaries": step_summaries,
        "pass_summary": final_pass_summary,
        "scope_note": "V3.1 minimal language world prediction only; VoxTell is frozen and no V3.1 visual rollout, SFDA, or extra loss is implemented.",
    }
    summary_path = output_dir / "v3_language_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
