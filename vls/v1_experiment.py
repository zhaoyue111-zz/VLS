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
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from torch import nn

from vls.augmentations import gamma_augment
from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, gamma_action, normalized_mse


DEFAULT_STRENGTHS = [-0.3, -0.15, 0.15, 0.3]
DEFAULT_EVAL_STEPS = [0, 30, 100, 300]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1 visual world model experiment.")
    parser.add_argument("--model-dir", default=str(ProjectPaths().voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(ProjectPaths().voxtell_root))
    parser.add_argument("--data-root", default=str(ProjectPaths().data_root))
    parser.add_argument("--split-json", default=str(ProjectPaths().split_json))
    parser.add_argument("--output-dir", default="outputs/v1")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--candidate-stages", nargs="+", default=["decoder_stage_1_low_to_high", "decoder_stage_2_low_to_high"])
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--dev-cases", type=int, default=8)
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--max-train-steps", type=int, default=300)
    parser.add_argument("--eval-steps", nargs="+", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--gamma-strengths", nargs="+", type=float, default=DEFAULT_STRENGTHS)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
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


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def padded_image_and_slicers(predictor: Any, image: np.ndarray) -> tuple[torch.Tensor, list[tuple]]:
    preprocessed, _, _ = predictor.preprocess(image)
    padded, _ = pad_nd_image(preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None)
    return padded, predictor._internal_get_sliding_window_slicers(padded.shape[1:])


def uniform_slicer_candidates(slicers: list[tuple], num_candidates: int) -> list[tuple]:
    if num_candidates >= len(slicers):
        return list(slicers)
    indices = np.linspace(0, len(slicers) - 1, num_candidates, dtype=np.int64)
    return [slicers[int(index)] for index in indices]


def score_foreground_slicers(
    interface: VoxTellStateInterface,
    padded: torch.Tensor,
    slicers: list[tuple],
    prompts: list[str],
    num_select: int,
    max_candidates: int,
    foreground_threshold: float,
) -> list[tuple]:
    if num_select <= 0:
        return []
    candidates = uniform_slicer_candidates(slicers, max(num_select, min(max_candidates, len(slicers))))
    scored = []
    for slicer in candidates:
        patch = torch.clone(padded[slicer][None], memory_format=torch.contiguous_format)
        result = interface.forward_with_states(patch, prompts)
        probability = torch.sigmoid(result["final_prediction"])
        foreground_voxels = float((probability > foreground_threshold).sum().detach().cpu())
        if foreground_voxels > 0:
            score = foreground_voxels
        else:
            flat_probability = probability.flatten()
            topk = min(4096, flat_probability.numel())
            score = float(torch.topk(flat_probability, topk).values.mean().detach().cpu())
        scored.append((score, slicer))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [slicer for _, slicer in scored[:num_select]]


def select_patch_slicers(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    prompts: list[str],
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
) -> tuple[torch.Tensor, list[tuple], list[str]]:
    padded, slicers = padded_image_and_slicers(interface.predictor, image)
    base_count = max(0, patches_per_case - foreground_patches_per_case)
    context_candidates = uniform_slicer_candidates(slicers, max(base_count, patches_per_case))
    selected = list(context_candidates[:base_count])
    patch_kinds = ["context"] * len(selected)
    foreground = score_foreground_slicers(
        interface,
        padded,
        slicers,
        prompts,
        foreground_patches_per_case,
        foreground_candidate_patches,
        foreground_threshold,
    )
    seen = {repr(s) for s in selected}
    for slicer in foreground:
        if repr(slicer) not in seen:
            selected.append(slicer)
            patch_kinds.append("foreground")
            seen.add(repr(slicer))
    for slicer in context_candidates + slicers:
        if len(selected) >= patches_per_case:
            break
        if repr(slicer) not in seen:
            selected.append(slicer)
            patch_kinds.append("context_fill")
            seen.add(repr(slicer))
    return padded, selected[:patches_per_case], patch_kinds[:patches_per_case]


@torch.inference_mode()
def extract_patch_pair(
    interface: VoxTellStateInterface,
    original_padded: torch.Tensor,
    gamma_padded: torch.Tensor,
    slicer: tuple,
    prompts: list[str],
    original: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
    gamma_patch = torch.clone(gamma_padded[slicer][None], memory_format=torch.contiguous_format)
    if original is None:
        original = interface.forward_with_states(original_patch, prompts)
    target = interface.forward_with_states(gamma_patch, prompts)
    return {
        "original": original,
        "target": target,
        "original_patch": original_patch,
        "gamma_patch": gamma_patch,
    }


def tensor_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024.0 ** 2)


def compare_candidate_stages(pair: dict[str, Any], stages: list[str]) -> list[dict[str, float | str]]:
    rows = []
    for stage in stages:
        source = pair["original"]["decoder_states"][stage]
        target = pair["target"]["decoder_states"][stage]
        pred_source = pair["original"]["intermediate_predictions"][stage]
        pred_target = pair["target"]["intermediate_predictions"][stage]
        rows.append({
            "stage": stage,
            "state_shape": "x".join(str(x) for x in source.shape),
            "state_mb": tensor_mb(source),
            "identity_normalized_mse": float(normalized_mse(source, target).detach().cpu()),
            "intermediate_prediction_normalized_mse": float(normalized_mse(pred_source, pred_target).detach().cpu()),
        })
    return rows


def stage_index(stage: str) -> int:
    return int(stage.split("_")[2])


@torch.inference_mode()
def state_to_intermediate_prediction(
    interface: VoxTellStateInterface,
    stage: str,
    state: torch.Tensor,
) -> torch.Tensor:
    if hasattr(interface, "functional_seg_head"):
        return interface.functional_seg_head(state)
    idx = stage_index(stage)
    decoder = interface.network.decoder
    if idx >= len(decoder.seg_layers):
        raise ValueError(f"Stage index {idx} is outside decoder seg layer range")
    return decoder.seg_layers[idx](state)


def build_dataset(
    interface: VoxTellStateInterface,
    cases: list[Any],
    prompts: list[str],
    strengths: list[float],
    selected_stage: str,
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
) -> dict[str, torch.Tensor | list[str] | list[float]]:
    states = []
    targets = []
    actions = []
    target_predictions = []
    case_ids = []
    case_names = []
    patch_indices = []
    patch_kinds_all = []
    strengths_all = []
    diagnostic_rows = []
    for case in cases:
        image, _, _ = read_image_and_label(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface,
            image,
            prompts,
            patches_per_case,
            foreground_patches_per_case,
            foreground_candidate_patches,
            foreground_threshold,
        )
        gamma_padded_by_strength = {
            strength: padded_image_and_slicers(interface.predictor, gamma_augment(image, 1.0 + strength))[0]
            for strength in strengths
        }
        for patch_index, slicer in enumerate(slicers):
            original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            original = interface.forward_with_states(original_patch, prompts)
            for strength in strengths:
                pair = extract_patch_pair(
                    interface,
                    original_padded,
                    gamma_padded_by_strength[strength],
                    slicer,
                    prompts,
                    original,
                )
                state = pair["original"]["decoder_states"][selected_stage][:, 0].detach().float()
                target = pair["target"]["decoder_states"][selected_stage][:, 0].detach().float()
                target_prediction = pair["target"]["intermediate_predictions"][selected_stage][:, 0:1].detach().float()
                states.append(state.cpu())
                targets.append(target.cpu())
                actions.append(gamma_action(strength, torch.device("cpu")))
                target_predictions.append(target_prediction.cpu())
                case_ids.append(f"{case.case}:patch{patch_index}:gamma{strength:+.2f}")
                case_names.append(case.case)
                patch_indices.append(patch_index)
                patch_kinds_all.append(patch_kinds[patch_index])
                strengths_all.append(float(strength))
                source_prediction = pair["original"]["intermediate_predictions"][selected_stage][:, 0:1].detach().float()
                diagnostic_rows.append({
                    "case": case.case,
                    "patch_index": patch_index,
                    "patch_kind": patch_kinds[patch_index],
                    "gamma_strength": float(strength),
                    "input_normalized_mse": float(normalized_mse(pair["original_patch"], pair["gamma_patch"]).detach().cpu()),
                    "state_normalized_mse": float(normalized_mse(state, target).detach().cpu()),
                    "mask_logit_normalized_mse": float(normalized_mse(source_prediction, target_prediction).detach().cpu()),
                })
    return {
        "states": torch.cat(states, dim=0),
        "targets": torch.cat(targets, dim=0),
        "actions": torch.cat(actions, dim=0),
        "target_predictions": torch.cat(target_predictions, dim=0),
        "case_ids": case_ids,
        "case_names": case_names,
        "patch_indices": patch_indices,
        "patch_kinds": patch_kinds_all,
        "strengths": strengths_all,
        "diagnostics": diagnostic_rows,
    }


def make_predictors(in_channels: int, hidden_channels: int, device: torch.device) -> tuple[nn.Module, nn.Module]:
    conditioned = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        use_action=True,
    ).to(device)
    agnostic = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        use_action=False,
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


def prepare_functional_seg_head(interface: VoxTellStateInterface, selected_stage: str) -> None:
    idx = stage_index(selected_stage)
    interface.functional_seg_head = deepcopy(interface.network.decoder.seg_layers[idx]).to(interface.device).eval()
    interface.network.to("cpu")
    if interface.device.type == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data: dict[str, torch.Tensor | list[str]],
    use_action: bool,
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> dict[str, float]:
    states = data["states"]
    targets = data["targets"]
    actions = data["actions"]
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
        action_batch = actions[start:end].to(device)
        target_prediction_batch = target_predictions[start:end].to(device)
        prediction = model(state_batch, action_batch if use_action else None)
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
def evaluate_identity(
    data: dict[str, torch.Tensor | list[str]],
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
    device = interface.device
    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        state_batch = states[start:end].to(device)
        target_batch = targets[start:end].to(device)
        target_prediction_batch = target_predictions[start:end].to(device)
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
def evaluate_wrong_action(
    model: nn.Module,
    data: dict[str, torch.Tensor | list[str]],
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> dict[str, float]:
    wrong_actions = data["actions"].clone()
    wrong_actions[:, 1] = -wrong_actions[:, 1]
    wrong_data = {**data, "actions": wrong_actions}
    return evaluate_model(model, wrong_data, True, interface, selected_stage, batch_size)


def subset_data(data: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    index_tensor = torch.tensor(indices, dtype=torch.long, device=data["states"].device)
    return {
        "states": data["states"].index_select(0, index_tensor),
        "targets": data["targets"].index_select(0, index_tensor),
        "actions": data["actions"].index_select(0, index_tensor),
        "target_predictions": data["target_predictions"].index_select(0, index_tensor),
        "case_ids": [data["case_ids"][i] for i in indices],
        "case_names": [data["case_names"][i] for i in indices],
        "patch_indices": [data["patch_indices"][i] for i in indices],
        "patch_kinds": [data["patch_kinds"][i] for i in indices],
        "strengths": [data["strengths"][i] for i in indices],
    }


def grouped_indices(data: dict[str, Any], group_by: str) -> list[tuple[str, list[int]]]:
    values = data[group_by]
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        if isinstance(value, float):
            key = f"{value:+.2f}"
        else:
            key = str(value)
        groups.setdefault(key, []).append(index)
    return sorted(groups.items(), key=lambda item: item[0])


@torch.inference_mode()
def append_group_evals(
    rows: list[dict[str, float | int | str]],
    step: int,
    split_name: str,
    data: dict[str, Any],
    group_name: str,
    group_value: str,
    models: dict[str, nn.Module],
    use_action: dict[str, bool],
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> None:
    rows.extend(compute_eval_records(
        step,
        split_name,
        group_name,
        group_value,
        data,
        models,
        use_action,
        interface,
        selected_stage,
        batch_size,
    ))


@torch.inference_mode()
def compute_eval_records(
    step: int,
    split_name: str,
    group_name: str,
    group_value: str,
    data: dict[str, Any],
    models: dict[str, nn.Module],
    use_action: dict[str, bool],
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> list[dict[str, float | int | str]]:
    records = []
    identity = evaluate_identity(data, interface, selected_stage, batch_size)
    records.append({
        "step": step,
        "split": split_name,
        "group": group_name,
        "group_value": group_value,
        "model": "identity",
        **identity,
    })
    for model_name, model in models.items():
        metrics = evaluate_model(model, data, use_action[model_name], interface, selected_stage, batch_size)
        records.append({
            "step": step,
            "split": split_name,
            "group": group_name,
            "group_value": group_value,
            "model": model_name,
            **metrics,
        })
    wrong = evaluate_wrong_action(models["action_conditioned"], data, interface, selected_stage, batch_size)
    records.append({
        "step": step,
        "split": split_name,
        "group": group_name,
        "group_value": group_value,
        "model": "action_conditioned_wrong_action",
        **wrong,
    })
    return records


def macro_records_from_case_records(
    step: int,
    split_name: str,
    case_records: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    models = sorted({str(row["model"]) for row in case_records})
    macro_rows = []
    for model_name in models:
        model_rows = [row for row in case_records if row["model"] == model_name]
        macro_rows.append({
            "step": step,
            "split": split_name,
            "group": "overall_macro",
            "group_value": "case_mean",
            "model": model_name,
            "state_normalized_mse": float(np.mean([float(row["state_normalized_mse"]) for row in model_rows])),
            "mask_logit_normalized_mse": float(np.mean([float(row["mask_logit_normalized_mse"]) for row in model_rows])),
        })
    return macro_rows


def records_by_model(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int | str]]:
    return {str(row["model"]): row for row in rows}


def v13_pass_summary(
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

    def state(model_rows: dict[str, dict[str, float | int | str]], model: str) -> float:
        return float(model_rows[model]["state_normalized_mse"])

    micro_conditioned_lt_agnostic = state(val_micro, "action_conditioned") < state(val_micro, "action_agnostic")
    micro_correct_lt_wrong = state(val_micro, "action_conditioned") < state(val_micro, "action_conditioned_wrong_action")
    macro_conditioned_lt_agnostic = state(val_macro, "action_conditioned") < state(val_macro, "action_agnostic")
    macro_correct_lt_wrong = state(val_macro, "action_conditioned") < state(val_macro, "action_conditioned_wrong_action")

    val_case_groups = sorted({
        str(row["group_value"])
        for row in final_group_rows
        if row["split"] == "val" and row["group"] == "case_names"
    })
    case_wins = []
    for case_name in val_case_groups:
        rows = records_by_model([
            row for row in final_group_rows
            if row["split"] == "val" and row["group"] == "case_names" and row["group_value"] == case_name
        ])
        case_wins.append({
            "case": case_name,
            "conditioned_lt_agnostic": state(rows, "action_conditioned") < state(rows, "action_agnostic"),
        })

    val_action_groups = sorted({
        str(row["group_value"])
        for row in final_group_rows
        if row["split"] == "val" and row["group"] == "strengths"
    })
    action_wins = []
    for strength in val_action_groups:
        rows = records_by_model([
            row for row in final_group_rows
            if row["split"] == "val" and row["group"] == "strengths" and row["group_value"] == strength
        ])
        action_wins.append({
            "strength": strength,
            "conditioned_lt_agnostic": state(rows, "action_conditioned") < state(rows, "action_agnostic"),
            "correct_lt_wrong": state(rows, "action_conditioned") < state(rows, "action_conditioned_wrong_action"),
            "conditioned_lt_identity": state(rows, "action_conditioned") < state(rows, "identity"),
        })

    case_win_count = sum(1 for item in case_wins if item["conditioned_lt_agnostic"])
    action_win_count = sum(
        1 for item in action_wins
        if item["conditioned_lt_agnostic"] and item["correct_lt_wrong"]
    )
    strong_intervention_conditioned_lt_identity = any(
        item["strength"] in {"+0.30", "-0.30"} and item["conditioned_lt_identity"]
        for item in action_wins
    )
    passed = (
        micro_conditioned_lt_agnostic
        and micro_correct_lt_wrong
        and macro_conditioned_lt_agnostic
        and macro_correct_lt_wrong
        and case_win_count >= 3
        and action_win_count >= 3
        and strong_intervention_conditioned_lt_identity
    )

    return {
        "passed": passed,
        "micro_conditioned_lt_agnostic": micro_conditioned_lt_agnostic,
        "micro_correct_lt_wrong": micro_correct_lt_wrong,
        "macro_conditioned_lt_agnostic": macro_conditioned_lt_agnostic,
        "macro_correct_lt_wrong": macro_correct_lt_wrong,
        "case_win_count": case_win_count,
        "case_total": len(case_wins),
        "action_win_count": action_win_count,
        "action_total": len(action_wins),
        "strong_intervention_conditioned_lt_identity": strong_intervention_conditioned_lt_identity,
        "case_wins": case_wins,
        "action_wins": action_wins,
        "criterion": "micro and macro conditioned < agnostic and correct < wrong; >=3/4 val cases; >=3/4 actions; at least one strong intervention conditioned < identity",
    }


def train_models(
    agnostic: nn.Module,
    conditioned: nn.Module,
    train_data: dict[str, torch.Tensor | list[str]],
    val_data: dict[str, torch.Tensor | list[str]],
    interface: VoxTellStateInterface,
    selected_stage: str,
    max_steps: int,
    eval_steps: list[int],
    batch_size: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    optimizers = {
        "action_agnostic": torch.optim.AdamW(agnostic.parameters(), lr=1e-3, weight_decay=1e-4),
        "action_conditioned": torch.optim.AdamW(conditioned.parameters(), lr=1e-3, weight_decay=1e-4),
    }
    models = {"action_agnostic": agnostic, "action_conditioned": conditioned}
    use_action = {"action_agnostic": False, "action_conditioned": True}
    eval_set = sorted(set([0, max_steps, *eval_steps]))
    rows: list[dict[str, float | int | str]] = []
    group_rows: list[dict[str, float | int | str]] = []

    def append_eval(step: int) -> None:
        for split_name, data in [("train", train_data), ("val", val_data)]:
            append_group_evals(
                rows,
                step,
                split_name,
                data,
                "overall_micro",
                "all",
                models,
                use_action,
                interface,
                selected_stage,
                batch_size,
            )
            case_records_for_macro = []
            for group_name in ["strengths", "case_names"]:
                for group_value, indices in grouped_indices(data, group_name):
                    records = compute_eval_records(
                        step,
                        split_name,
                        group_name,
                        group_value,
                        subset_data(data, indices),
                        models,
                        use_action,
                        interface,
                        selected_stage,
                        batch_size,
                    )
                    group_rows.extend(records)
                    if group_name == "case_names":
                        case_records_for_macro.extend(records)
            rows.extend(macro_records_from_case_records(step, split_name, case_records_for_macro))

    append_eval(0)
    device = next(agnostic.parameters()).device
    states = train_data["states"]
    targets = train_data["targets"]
    actions = train_data["actions"]
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
        action_batch = actions.index_select(0, index_tensor).to(device)
        for model_name, model in models.items():
            model.train()
            optimizers[model_name].zero_grad(set_to_none=True)
            prediction = model(state_batch, action_batch if use_action[model_name] else None)
            loss = normalized_mse(prediction, target_batch)
            loss.backward()
            optimizers[model_name].step()
            model.eval()
        if step in eval_set:
            append_eval(step)
    return rows, group_rows


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

    first_image, _, _ = read_image_and_label(train_cases[0])
    padded, slicers, _ = select_patch_slicers(
        interface,
        first_image,
        args.prompts,
        patches_per_case=1,
        foreground_patches_per_case=0,
        foreground_candidate_patches=args.foreground_candidate_patches,
        foreground_threshold=args.foreground_threshold,
    )
    gamma_padded, _ = padded_image_and_slicers(interface.predictor, gamma_augment(first_image, 1.0 + args.gamma_strengths[0]))
    first_pair = extract_patch_pair(interface, padded, gamma_padded, slicers[0], args.prompts)
    candidate_rows = compare_candidate_stages(first_pair, args.candidate_stages)

    train_data = build_dataset(
        interface,
        train_cases,
        args.prompts,
        args.gamma_strengths,
        args.selected_stage,
        args.patches_per_case,
        args.foreground_patches_per_case,
        args.foreground_candidate_patches,
        args.foreground_threshold,
    )
    val_data = build_dataset(
        interface,
        val_cases,
        args.prompts,
        args.gamma_strengths,
        args.selected_stage,
        args.patches_per_case,
        args.foreground_patches_per_case,
        args.foreground_candidate_patches,
        args.foreground_threshold,
    )
    prepare_functional_seg_head(interface, args.selected_stage)

    agnostic, conditioned = make_predictors(
        in_channels=train_data["states"].shape[1],
        hidden_channels=args.hidden_channels,
        device=device,
    )
    curve_rows, group_rows = train_models(
        agnostic,
        conditioned,
        train_data,
        val_data,
        interface,
        args.selected_stage,
        args.max_train_steps,
        args.eval_steps,
        args.batch_size,
    )

    candidate_path = output_dir / "candidate_stage_metrics.csv"
    with candidate_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)

    curve_path = output_dir / "training_curve.csv"
    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "split",
                "group",
                "group_value",
                "model",
                "state_normalized_mse",
                "mask_logit_normalized_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    group_curve_path = output_dir / "grouped_training_curve.csv"
    with group_curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "split",
                "group",
                "group_value",
                "model",
                "state_normalized_mse",
                "mask_logit_normalized_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(group_rows)

    diagnostic_rows = []
    for split_name, data in [("train", train_data), ("val", val_data)]:
        for row in data["diagnostics"]:
            diagnostic_rows.append({"split": split_name, **row})
    diagnostic_path = output_dir / "transition_diagnostics.csv"
    with diagnostic_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "case",
                "patch_index",
                "patch_kind",
                "gamma_strength",
                "input_normalized_mse",
                "state_normalized_mse",
                "mask_logit_normalized_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    final_rows = [row for row in curve_rows if row["step"] == args.max_train_steps]
    final_group_rows = [row for row in group_rows if row["step"] == args.max_train_steps]
    pass_summary = v13_pass_summary(final_rows, final_group_rows)
    summary = {
        "args": vars(args),
        "train_cases": [case.case for case in train_cases],
        "val_cases": [case.case for case in val_cases],
        "selected_stage": args.selected_stage,
        "train_samples": len(train_data["case_ids"]),
        "val_samples": len(val_data["case_ids"]),
        "candidate_stage_csv": str(candidate_path),
        "training_curve_csv": str(curve_path),
        "grouped_training_curve_csv": str(group_curve_path),
        "transition_diagnostics_csv": str(diagnostic_path),
        "final_metrics": final_rows,
        "final_group_metrics": final_group_rows,
        "v1_3_pass_summary": pass_summary,
        "expected_sanity_order": "action_conditioned < action_agnostic < identity, plus correct action < wrong action",
    }
    summary_path = output_dir / "v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
