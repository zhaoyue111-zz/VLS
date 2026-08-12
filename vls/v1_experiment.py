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
    parser.add_argument("--dev-cases", type=int, default=1)
    parser.add_argument("--val-cases", type=int, default=0)
    parser.add_argument("--patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-patches-per-case", type=int, default=0)
    parser.add_argument("--foreground-candidate-patches", type=int, default=12)
    parser.add_argument("--max-train-steps", type=int, default=300)
    parser.add_argument("--eval-steps", nargs="+", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--gamma-strengths", nargs="+", type=float, default=DEFAULT_STRENGTHS)
    parser.add_argument("--hidden-channels", type=int, default=32)
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


def score_foreground_slicers(
    interface: VoxTellStateInterface,
    padded: torch.Tensor,
    slicers: list[tuple],
    prompts: list[str],
    num_select: int,
    max_candidates: int,
) -> list[tuple]:
    if num_select <= 0:
        return []
    candidates = slicers[: max(num_select, min(max_candidates, len(slicers)))]
    scored = []
    for slicer in candidates:
        patch = torch.clone(padded[slicer][None], memory_format=torch.contiguous_format)
        result = interface.forward_with_states(patch, prompts)
        score = float(torch.sigmoid(result["final_prediction"]).sum().detach().cpu())
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
) -> tuple[torch.Tensor, list[tuple]]:
    padded, slicers = padded_image_and_slicers(interface.predictor, image)
    base_count = max(0, patches_per_case - foreground_patches_per_case)
    selected = list(slicers[:base_count])
    foreground = score_foreground_slicers(
        interface,
        padded,
        slicers,
        prompts,
        foreground_patches_per_case,
        foreground_candidate_patches,
    )
    seen = {repr(s) for s in selected}
    for slicer in foreground:
        if repr(slicer) not in seen:
            selected.append(slicer)
            seen.add(repr(slicer))
    for slicer in slicers:
        if len(selected) >= patches_per_case:
            break
        if repr(slicer) not in seen:
            selected.append(slicer)
            seen.add(repr(slicer))
    return padded, selected[:patches_per_case]


@torch.inference_mode()
def extract_patch_pair(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    slicer: tuple,
    prompts: list[str],
    strength: float,
) -> dict[str, Any]:
    gamma_value = 1.0 + strength
    original_padded, _ = padded_image_and_slicers(interface.predictor, image)
    gamma_padded, _ = padded_image_and_slicers(interface.predictor, gamma_augment(image, gamma_value))
    original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
    gamma_patch = torch.clone(gamma_padded[slicer][None], memory_format=torch.contiguous_format)
    original = interface.forward_with_states(original_patch, prompts)
    target = interface.forward_with_states(gamma_patch, prompts)
    return {"original": original, "target": target}


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
) -> dict[str, torch.Tensor | list[str]]:
    states = []
    targets = []
    actions = []
    target_predictions = []
    case_ids = []
    for case in cases:
        image, _, _ = read_image_and_label(case)
        _, slicers = select_patch_slicers(
            interface,
            image,
            prompts,
            patches_per_case,
            foreground_patches_per_case,
            foreground_candidate_patches,
        )
        for patch_index, slicer in enumerate(slicers):
            for strength in strengths:
                pair = extract_patch_pair(interface, image, slicer, prompts, strength)
                states.append(pair["original"]["decoder_states"][selected_stage][:, 0].detach().float())
                targets.append(pair["target"]["decoder_states"][selected_stage][:, 0].detach().float())
                actions.append(gamma_action(strength, interface.device))
                target_predictions.append(pair["target"]["intermediate_predictions"][selected_stage][:, 0:1].detach().float())
                case_ids.append(f"{case.case}:patch{patch_index}:gamma{strength:+.2f}")
    return {
        "states": torch.cat(states, dim=0),
        "targets": torch.cat(targets, dim=0),
        "actions": torch.cat(actions, dim=0),
        "target_predictions": torch.cat(target_predictions, dim=0),
        "case_ids": case_ids,
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


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data: dict[str, torch.Tensor | list[str]],
    use_action: bool,
    interface: VoxTellStateInterface,
    selected_stage: str,
) -> dict[str, float]:
    states = data["states"]
    targets = data["targets"]
    actions = data["actions"]
    target_predictions = data["target_predictions"]
    prediction = model(states, actions if use_action else None)
    state_loss = normalized_mse(prediction, targets)
    pred_logits = state_to_intermediate_prediction(interface, selected_stage, prediction)
    mask_loss = normalized_mse(pred_logits, target_predictions)
    return {
        "state_normalized_mse": float(state_loss.detach().cpu()),
        "mask_logit_normalized_mse": float(mask_loss.detach().cpu()),
    }


@torch.inference_mode()
def evaluate_identity(
    data: dict[str, torch.Tensor | list[str]],
    interface: VoxTellStateInterface,
    selected_stage: str,
) -> dict[str, float]:
    states = data["states"]
    targets = data["targets"]
    target_predictions = data["target_predictions"]
    pred_logits = state_to_intermediate_prediction(interface, selected_stage, states)
    return {
        "state_normalized_mse": float(normalized_mse(states, targets).detach().cpu()),
        "mask_logit_normalized_mse": float(normalized_mse(pred_logits, target_predictions).detach().cpu()),
    }


@torch.inference_mode()
def evaluate_wrong_action(
    model: nn.Module,
    data: dict[str, torch.Tensor | list[str]],
    interface: VoxTellStateInterface,
    selected_stage: str,
) -> dict[str, float]:
    actions = data["actions"].clone()
    actions[:, 1] = -actions[:, 1]
    prediction = model(data["states"], actions)
    pred_logits = state_to_intermediate_prediction(interface, selected_stage, prediction)
    return {
        "state_normalized_mse": float(normalized_mse(prediction, data["targets"]).detach().cpu()),
        "mask_logit_normalized_mse": float(normalized_mse(pred_logits, data["target_predictions"]).detach().cpu()),
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
) -> list[dict[str, float | int | str]]:
    optimizers = {
        "action_agnostic": torch.optim.AdamW(agnostic.parameters(), lr=1e-3, weight_decay=1e-4),
        "action_conditioned": torch.optim.AdamW(conditioned.parameters(), lr=1e-3, weight_decay=1e-4),
    }
    models = {"action_agnostic": agnostic, "action_conditioned": conditioned}
    use_action = {"action_agnostic": False, "action_conditioned": True}
    eval_set = sorted(set([0, max_steps, *eval_steps]))
    rows: list[dict[str, float | int | str]] = []

    def append_eval(step: int) -> None:
        for split_name, data in [("train", train_data), ("val", val_data)]:
            identity = evaluate_identity(data, interface, selected_stage)
            rows.append({"step": step, "split": split_name, "model": "identity", **identity})
            for model_name, model in models.items():
                metrics = evaluate_model(model, data, use_action[model_name], interface, selected_stage)
                rows.append({"step": step, "split": split_name, "model": model_name, **metrics})
            wrong = evaluate_wrong_action(conditioned, data, interface, selected_stage)
            rows.append({"step": step, "split": split_name, "model": "action_conditioned_wrong_action", **wrong})

    append_eval(0)
    states = train_data["states"]
    targets = train_data["targets"]
    actions = train_data["actions"]
    for step in range(1, max_steps + 1):
        for model_name, model in models.items():
            model.train()
            optimizers[model_name].zero_grad(set_to_none=True)
            prediction = model(states, actions if use_action[model_name] else None)
            loss = normalized_mse(prediction, targets)
            loss.backward()
            optimizers[model_name].step()
            model.eval()
        if step in eval_set:
            append_eval(step)
    return rows


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
    padded, slicers = select_patch_slicers(
        interface,
        first_image,
        args.prompts,
        patches_per_case=1,
        foreground_patches_per_case=0,
        foreground_candidate_patches=args.foreground_candidate_patches,
    )
    first_pair = extract_patch_pair(interface, first_image, slicers[0], args.prompts, args.gamma_strengths[0])
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
    )

    agnostic, conditioned = make_predictors(
        in_channels=train_data["states"].shape[1],
        hidden_channels=args.hidden_channels,
        device=device,
    )
    curve_rows = train_models(
        agnostic,
        conditioned,
        train_data,
        val_data,
        interface,
        args.selected_stage,
        args.max_train_steps,
        args.eval_steps,
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
            fieldnames=["step", "split", "model", "state_normalized_mse", "mask_logit_normalized_mse"],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    final_rows = [row for row in curve_rows if row["step"] == args.max_train_steps]
    summary = {
        "args": vars(args),
        "train_cases": [case.case for case in train_cases],
        "val_cases": [case.case for case in val_cases],
        "selected_stage": args.selected_stage,
        "train_samples": len(train_data["case_ids"]),
        "val_samples": len(val_data["case_ids"]),
        "candidate_stage_csv": str(candidate_path),
        "training_curve_csv": str(curve_path),
        "final_metrics": final_rows,
        "expected_sanity_order": "action_conditioned < action_agnostic < identity, plus correct action < wrong action",
    }
    summary_path = output_dir / "v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
