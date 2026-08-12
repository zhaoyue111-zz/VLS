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

from vls.augmentations import gamma_augment, gaussian_blur_augment
from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, normalized_mse


DEFAULT_GAMMA_STRENGTHS = [-0.3, -0.15, 0.15, 0.3]
DEFAULT_BLUR_SIGMAS = [0.5, 1.5]
DEFAULT_EVAL_STEPS = [0, 100, 200, 300]
DEFAULT_ACTION_FAMILIES = ["gamma", "blur"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 visual world model experiment with gamma and blur actions.")
    parser.add_argument("--model-dir", default=str(ProjectPaths().voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(ProjectPaths().voxtell_root))
    parser.add_argument("--data-root", default=str(ProjectPaths().data_root))
    parser.add_argument("--split-json", default=str(ProjectPaths().split_json))
    parser.add_argument("--output-dir", default="outputs/v2")
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
    parser.add_argument("--gamma-strengths", nargs="+", type=float, default=DEFAULT_GAMMA_STRENGTHS)
    parser.add_argument("--blur-sigmas", nargs="+", type=float, default=DEFAULT_BLUR_SIGMAS)
    parser.add_argument("--action-families", nargs="+", default=DEFAULT_ACTION_FAMILIES)
    parser.add_argument("--identity-action-anchors", action=argparse.BooleanOptionalAction, default=True)
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
    return padded_preprocessed_and_slicers(predictor, preprocessed)


def padded_preprocessed_and_slicers(predictor: Any, preprocessed: torch.Tensor) -> tuple[torch.Tensor, list[tuple]]:
    padded, _ = pad_nd_image(preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None)
    return padded, predictor._internal_get_sliding_window_slicers(padded.shape[1:])


def padded_visual_action_and_slicers(
    predictor: Any,
    image: np.ndarray,
    action_family: str,
    strength: float,
) -> tuple[torch.Tensor, list[tuple]]:
    if action_family == "gamma":
        return padded_image_and_slicers(predictor, gamma_augment(image, 1.0 + strength))
    if action_family == "blur":
        preprocessed, _, _ = predictor.preprocess(image)
        blurred = gaussian_blur_augment(preprocessed.numpy(), strength)
        return padded_preprocessed_and_slicers(predictor, torch.from_numpy(blurred))
    raise ValueError(f"Unsupported action family: {action_family}")


def visual_action(action_family: str, strength: float, device: torch.device) -> torch.Tensor:
    if action_family == "gamma":
        values = [1.0, 0.0, float(strength)]
    elif action_family == "blur":
        values = [0.0, 1.0, float(strength)]
    else:
        raise ValueError(f"Unsupported action family: {action_family}")
    return torch.tensor([values], dtype=torch.float32, device=device)


def swapped_action_family(action_family: str) -> str:
    if action_family == "gamma":
        return "blur"
    if action_family == "blur":
        return "gamma"
    raise ValueError(f"Unsupported action family: {action_family}")


def action_values_by_family(args: argparse.Namespace) -> dict[str, list[float]]:
    values = {
        "gamma": [float(value) for value in args.gamma_strengths],
        "blur": [float(value) for value in args.blur_sigmas],
    }
    unsupported = sorted(set(args.action_families) - set(values))
    if unsupported:
        raise ValueError(f"Unsupported action families: {unsupported}")
    return values


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
    action_families: list[str],
    action_values_by_family: dict[str, list[float]],
    selected_stage: str,
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
    include_identity_anchors: bool = False,
) -> dict[str, torch.Tensor | list[str] | list[float]]:
    states = []
    targets = []
    actions = []
    target_predictions = []
    case_ids = []
    case_names = []
    patch_indices = []
    patch_kinds_all = []
    action_families_all = []
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
        padded_by_action = {
            (action_family, strength): padded_visual_action_and_slicers(
                interface.predictor,
                image,
                action_family,
                strength,
            )[0]
            for action_family in action_families
            for strength in action_values_by_family[action_family]
        }
        for patch_index, slicer in enumerate(slicers):
            original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            original = interface.forward_with_states(original_patch, prompts)
            original_state = original["decoder_states"][selected_stage][:, 0].detach().float()
            source_prediction = original["intermediate_predictions"][selected_stage][:, 0:1].detach().float()
            if include_identity_anchors:
                for action_family in action_families:
                    states.append(original_state.cpu())
                    targets.append(original_state.cpu())
                    actions.append(visual_action(action_family, 0.0, torch.device("cpu")))
                    target_predictions.append(source_prediction.cpu())
                    case_ids.append(f"{case.case}:patch{patch_index}:{action_family}+0.00_anchor")
                    case_names.append(case.case)
                    patch_indices.append(patch_index)
                    patch_kinds_all.append(patch_kinds[patch_index])
                    action_families_all.append(action_family)
                    strengths_all.append(0.0)
                    diagnostic_rows.append({
                        "case": case.case,
                        "patch_index": patch_index,
                        "patch_kind": patch_kinds[patch_index],
                        "action_family": action_family,
                        "strength": 0.0,
                        "input_normalized_mse": 0.0,
                        "state_normalized_mse": 0.0,
                        "mask_logit_normalized_mse": 0.0,
                    })
            for action_family in action_families:
                for strength in action_values_by_family[action_family]:
                    pair = extract_patch_pair(
                        interface,
                        original_padded,
                        padded_by_action[(action_family, strength)],
                        slicer,
                        prompts,
                        original,
                    )
                    state = pair["original"]["decoder_states"][selected_stage][:, 0].detach().float()
                    target = pair["target"]["decoder_states"][selected_stage][:, 0].detach().float()
                    target_prediction = pair["target"]["intermediate_predictions"][selected_stage][:, 0:1].detach().float()
                    states.append(state.cpu())
                    targets.append(target.cpu())
                    actions.append(visual_action(action_family, strength, torch.device("cpu")))
                    target_predictions.append(target_prediction.cpu())
                    case_ids.append(f"{case.case}:patch{patch_index}:{action_family}{strength:+.2f}")
                    case_names.append(case.case)
                    patch_indices.append(patch_index)
                    patch_kinds_all.append(patch_kinds[patch_index])
                    action_families_all.append(action_family)
                    strengths_all.append(float(strength))
                    diagnostic_rows.append({
                        "case": case.case,
                        "patch_index": patch_index,
                        "patch_kind": patch_kinds[patch_index],
                        "action_family": action_family,
                        "strength": float(strength),
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
        "action_families": action_families_all,
        "strengths": strengths_all,
        "diagnostics": diagnostic_rows,
    }


def make_predictors(in_channels: int, hidden_channels: int, device: torch.device) -> tuple[nn.Module, nn.Module]:
    conditioned = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        action_dim=3,
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


def wrong_strength_actions(actions: torch.Tensor) -> torch.Tensor:
    wrong_actions = actions.clone()
    gamma_mask = actions[:, 0] > actions[:, 1]
    blur_mask = actions[:, 1] > actions[:, 0]
    wrong_actions[gamma_mask, 2] = -wrong_actions[gamma_mask, 2]
    wrong_actions[blur_mask, 2] = torch.where(
        actions[blur_mask, 2] < 1.0,
        torch.full_like(actions[blur_mask, 2], 1.5),
        torch.full_like(actions[blur_mask, 2], 0.5),
    )
    return wrong_actions


def wrong_type_matched_data(data: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor] | None:
    indices = []
    replacement_actions = []
    for index, (action_family, strength) in enumerate(zip(data["action_families"], data["strengths"], strict=True)):
        strength_value = float(strength)
        if action_family == "gamma" and np.isclose(strength_value, 0.15):
            indices.append(index)
            replacement_actions.append([0.0, 1.0, 0.5])
        elif action_family == "gamma" and np.isclose(strength_value, 0.30):
            indices.append(index)
            replacement_actions.append([0.0, 1.0, 1.5])
        elif action_family == "blur" and np.isclose(strength_value, 0.5):
            indices.append(index)
            replacement_actions.append([1.0, 0.0, 0.15])
        elif action_family == "blur" and np.isclose(strength_value, 1.5):
            indices.append(index)
            replacement_actions.append([1.0, 0.0, 0.30])
    if not indices:
        return None
    matched_data = subset_data(data, indices)
    actions = torch.tensor(replacement_actions, dtype=torch.float32, device=matched_data["actions"].device)
    return matched_data, actions


@torch.inference_mode()
def evaluate_with_replaced_actions(
    model: nn.Module,
    data: dict[str, torch.Tensor | list[str]],
    replacement_actions: torch.Tensor,
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
) -> dict[str, float]:
    wrong_data = {**data, "actions": replacement_actions}
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
        "action_families": [data["action_families"][i] for i in indices],
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
    wrong_strength = evaluate_with_replaced_actions(
        models["action_conditioned"],
        data,
        wrong_strength_actions(data["actions"]),
        interface,
        selected_stage,
        batch_size,
    )
    records.append({
        "step": step,
        "split": split_name,
        "group": group_name,
        "group_value": group_value,
        "model": "action_conditioned_wrong_strength",
        **wrong_strength,
    })
    wrong_type_pair = wrong_type_matched_data(data)
    if wrong_type_pair is not None:
        wrong_type_data, wrong_type_actions = wrong_type_pair
        correct_for_wrong_type = evaluate_model(
            models["action_conditioned"],
            wrong_type_data,
            True,
            interface,
            selected_stage,
            batch_size,
        )
        records.append({
            "step": step,
            "split": split_name,
            "group": group_name,
            "group_value": group_value,
            "model": "action_conditioned_correct_for_wrong_type",
            **correct_for_wrong_type,
        })
        wrong_type = evaluate_with_replaced_actions(
            models["action_conditioned"],
            wrong_type_data,
            wrong_type_actions,
            interface,
            selected_stage,
            batch_size,
        )
        records.append({
            "step": step,
            "split": split_name,
            "group": group_name,
            "group_value": group_value,
            "model": "action_conditioned_wrong_type",
            **wrong_type,
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


def v2_pass_summary(
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

    def optional_state(model_rows: dict[str, dict[str, float | int | str]], model: str) -> float | None:
        if model not in model_rows:
            return None
        return float(model_rows[model]["state_normalized_mse"])

    micro_conditioned_lt_agnostic = state(val_micro, "action_conditioned") < state(val_micro, "action_agnostic")
    micro_correct_lt_wrong_strength = state(val_micro, "action_conditioned") < state(val_micro, "action_conditioned_wrong_strength")
    micro_wrong_type_correct = optional_state(val_micro, "action_conditioned_correct_for_wrong_type")
    micro_correct_lt_wrong_type = (
        micro_wrong_type_correct is not None
        and micro_wrong_type_correct < state(val_micro, "action_conditioned_wrong_type")
    )
    macro_conditioned_lt_agnostic = state(val_macro, "action_conditioned") < state(val_macro, "action_agnostic")
    macro_correct_lt_wrong_strength = state(val_macro, "action_conditioned") < state(val_macro, "action_conditioned_wrong_strength")
    macro_wrong_type_correct = optional_state(val_macro, "action_conditioned_correct_for_wrong_type")
    macro_correct_lt_wrong_type = (
        macro_wrong_type_correct is not None
        and macro_wrong_type_correct < state(val_macro, "action_conditioned_wrong_type")
    )

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

    val_family_groups = sorted({
        str(row["group_value"])
        for row in final_group_rows
        if row["split"] == "val" and row["group"] == "action_families"
    })
    family_wins = []
    for family in val_family_groups:
        rows = records_by_model([
            row for row in final_group_rows
            if row["split"] == "val" and row["group"] == "action_families" and row["group_value"] == family
        ])
        family_wins.append({
            "action_family": family,
            "conditioned_lt_agnostic": state(rows, "action_conditioned") < state(rows, "action_agnostic"),
            "correct_lt_wrong_strength": state(rows, "action_conditioned") < state(rows, "action_conditioned_wrong_strength"),
            "correct_lt_wrong_type": (
                optional_state(rows, "action_conditioned_correct_for_wrong_type") is not None
                and optional_state(rows, "action_conditioned_correct_for_wrong_type") < state(rows, "action_conditioned_wrong_type")
            ),
        })

    val_strength_groups = sorted({
        str(row["group_value"])
        for row in final_group_rows
        if row["split"] == "val" and row["group"] == "strengths"
    })
    strength_wins = []
    for strength in val_strength_groups:
        rows = records_by_model([
            row for row in final_group_rows
            if row["split"] == "val" and row["group"] == "strengths" and row["group_value"] == strength
        ])
        wrong_type_state = optional_state(rows, "action_conditioned_wrong_type")
        wrong_type_correct_state = optional_state(rows, "action_conditioned_correct_for_wrong_type")
        strength_wins.append({
            "strength": strength,
            "conditioned_lt_agnostic": state(rows, "action_conditioned") < state(rows, "action_agnostic"),
            "correct_lt_wrong_strength": state(rows, "action_conditioned") < state(rows, "action_conditioned_wrong_strength"),
            "correct_lt_wrong_type": (
                True if wrong_type_state is None else wrong_type_correct_state is not None and wrong_type_correct_state < wrong_type_state
            ),
            "wrong_type_applicable": wrong_type_state is not None,
        })

    case_win_count = sum(1 for item in case_wins if item["conditioned_lt_agnostic"])
    family_win_count = sum(
        1 for item in family_wins
        if item["conditioned_lt_agnostic"]
    )
    strength_win_count = sum(
        1 for item in strength_wins
        if item["conditioned_lt_agnostic"]
        and item["correct_lt_wrong_strength"]
        and item["correct_lt_wrong_type"]
    )
    passed = (
        micro_conditioned_lt_agnostic
        and micro_correct_lt_wrong_strength
        and micro_correct_lt_wrong_type
        and macro_conditioned_lt_agnostic
        and macro_correct_lt_wrong_strength
        and macro_correct_lt_wrong_type
        and family_win_count == len(family_wins)
        and case_win_count > len(case_wins) / 2
        and strength_win_count > len(strength_wins) / 2
    )

    return {
        "passed": passed,
        "micro_conditioned_lt_agnostic": micro_conditioned_lt_agnostic,
        "micro_correct_lt_wrong_strength": micro_correct_lt_wrong_strength,
        "micro_correct_lt_wrong_type": micro_correct_lt_wrong_type,
        "macro_conditioned_lt_agnostic": macro_conditioned_lt_agnostic,
        "macro_correct_lt_wrong_strength": macro_correct_lt_wrong_strength,
        "macro_correct_lt_wrong_type": macro_correct_lt_wrong_type,
        "case_win_count": case_win_count,
        "case_total": len(case_wins),
        "action_family_win_count": family_win_count,
        "action_family_total": len(family_wins),
        "strength_win_count": strength_win_count,
        "strength_total": len(strength_wins),
        "case_wins": case_wins,
        "action_family_wins": family_wins,
        "strength_wins": strength_wins,
        "criterion": "micro and macro conditioned < agnostic; correct < wrong-strength; correct < wrong-type; gamma and blur families each conditioned < agnostic; case and strength win rates must be majority",
    }


def val_macro_conditioned_state(rows: list[dict[str, float | int | str]]) -> float:
    for row in rows:
        if row["split"] == "val" and row["group"] == "overall_macro" and row["model"] == "action_conditioned":
            return float(row["state_normalized_mse"])
    raise ValueError("Missing val overall_macro action_conditioned metric")


def pass_summaries_by_step(
    curve_rows: list[dict[str, float | int | str]],
    group_rows: list[dict[str, float | int | str]],
) -> dict[str, dict[str, Any]]:
    steps = sorted({int(row["step"]) for row in curve_rows})
    summaries = {}
    for step in steps:
        step_rows = [row for row in curve_rows if int(row["step"]) == step]
        step_group_rows = [row for row in group_rows if int(row["step"]) == step]
        if not step_rows or not step_group_rows:
            continue
        summary = v2_pass_summary(step_rows, step_group_rows)
        summary["val_macro_conditioned_state_normalized_mse"] = val_macro_conditioned_state(step_rows)
        summaries[str(step)] = summary
    return summaries


def select_v2_checkpoint_step(step_summaries: dict[str, dict[str, Any]], fallback_step: int) -> int:
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
    train_data: dict[str, torch.Tensor | list[str]],
    train_eval_data: dict[str, torch.Tensor | list[str]],
    val_data: dict[str, torch.Tensor | list[str]],
    interface: VoxTellStateInterface,
    selected_stage: str,
    max_steps: int,
    eval_steps: list[int],
    batch_size: int,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    dict[int, dict[str, dict[str, torch.Tensor]]],
]:
    optimizers = {
        "action_agnostic": torch.optim.AdamW(agnostic.parameters(), lr=1e-3, weight_decay=1e-4),
        "action_conditioned": torch.optim.AdamW(conditioned.parameters(), lr=1e-3, weight_decay=1e-4),
    }
    models = {"action_agnostic": agnostic, "action_conditioned": conditioned}
    use_action = {"action_agnostic": False, "action_conditioned": True}
    eval_set = sorted(set([0, max_steps, *eval_steps]))
    rows: list[dict[str, float | int | str]] = []
    group_rows: list[dict[str, float | int | str]] = []
    checkpoints: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    def append_eval(step: int) -> None:
        for split_name, data in [("train", train_eval_data), ("val", val_data)]:
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
            for group_name in ["action_families", "strengths", "case_names"]:
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
        checkpoints[step] = {
            "action_agnostic": cpu_state_dict(agnostic),
            "action_conditioned": cpu_state_dict(conditioned),
        }

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
    return rows, group_rows, checkpoints


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
    family_values = action_values_by_family(args)

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
    gamma_padded, _ = padded_image_and_slicers(interface.predictor, gamma_augment(first_image, 1.0 + family_values["gamma"][0]))
    first_pair = extract_patch_pair(interface, padded, gamma_padded, slicers[0], args.prompts)
    candidate_rows = compare_candidate_stages(first_pair, args.candidate_stages)

    train_eval_data = build_dataset(
        interface,
        train_cases,
        args.prompts,
        args.action_families,
        family_values,
        args.selected_stage,
        args.patches_per_case,
        args.foreground_patches_per_case,
        args.foreground_candidate_patches,
        args.foreground_threshold,
    )
    train_data = build_dataset(
        interface,
        train_cases,
        args.prompts,
        args.action_families,
        family_values,
        args.selected_stage,
        args.patches_per_case,
        args.foreground_patches_per_case,
        args.foreground_candidate_patches,
        args.foreground_threshold,
        include_identity_anchors=args.identity_action_anchors,
    )
    val_data = build_dataset(
        interface,
        val_cases,
        args.prompts,
        args.action_families,
        family_values,
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
                "action_family",
                "strength",
                "input_normalized_mse",
                "state_normalized_mse",
                "mask_logit_normalized_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    step_summaries = pass_summaries_by_step(curve_rows, group_rows)
    selected_step = select_v2_checkpoint_step(step_summaries, args.max_train_steps)
    final_rows = [row for row in curve_rows if int(row["step"]) == selected_step]
    final_group_rows = [row for row in group_rows if int(row["step"]) == selected_step]
    pass_summary = step_summaries[str(selected_step)]
    checkpoint_path = output_dir / f"world_predictor_step{selected_step}.pt"
    checkpoint = {
        "selected_step": selected_step,
        "selected_stage": args.selected_stage,
        "action_dim": 3,
        "hidden_channels": args.hidden_channels,
        "action_encoding": {
            "gamma": "[1, 0, strength]",
            "blur": "[0, 1, sigma]",
        },
        "gamma_strengths": [float(value) for value in args.gamma_strengths],
        "blur_sigmas": [float(value) for value in args.blur_sigmas],
        "identity_action_anchors": bool(args.identity_action_anchors),
        "conditioned_state_dict": checkpoints[selected_step]["action_conditioned"],
        "agnostic_state_dict": checkpoints[selected_step]["action_agnostic"],
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "args": vars(args),
        "train_cases": [case.case for case in train_cases],
        "val_cases": [case.case for case in val_cases],
        "selected_stage": args.selected_stage,
        "train_samples": len(train_data["case_ids"]),
        "train_eval_samples": len(train_eval_data["case_ids"]),
        "val_samples": len(val_data["case_ids"]),
        "candidate_stage_csv": str(candidate_path),
        "training_curve_csv": str(curve_path),
        "grouped_training_curve_csv": str(group_curve_path),
        "transition_diagnostics_csv": str(diagnostic_path),
        "final_metrics": final_rows,
        "final_group_metrics": final_group_rows,
        "step_pass_summaries": step_summaries,
        "selected_step": selected_step,
        "selected_checkpoint_path": str(checkpoint_path),
        "v2_pass_summary": pass_summary,
        "expected_sanity_order": "action_conditioned < action_agnostic; correct action < wrong-strength and wrong-type",
    }
    summary_path = output_dir / "v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
