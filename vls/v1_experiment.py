from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from torch import nn

from vls.augmentations import gamma_augment
from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, gamma_action, normalized_mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1 minimal visual world model experiment.")
    parser.add_argument("--model-dir", default=str(ProjectPaths().voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(ProjectPaths().voxtell_root))
    parser.add_argument("--data-root", default=str(ProjectPaths().data_root))
    parser.add_argument("--split-json", default=str(ProjectPaths().split_json))
    parser.add_argument("--output-dir", default="outputs/v1")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--candidate-stages", nargs="+", default=["decoder_stage_1_low_to_high", "decoder_stage_2_low_to_high"])
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--dev-cases", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=3)
    parser.add_argument("--gamma-strength", type=float, default=0.2)
    parser.add_argument("--gamma-strengths", nargs="+", type=float, default=None)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def first_patch(predictor, image: np.ndarray) -> torch.Tensor:
    preprocessed, _, _ = predictor.preprocess(image)
    padded, _ = pad_nd_image(preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None)
    slicer = predictor._internal_get_sliding_window_slicers(padded.shape[1:])[0]
    return torch.clone(padded[slicer][None], memory_format=torch.contiguous_format)


@torch.inference_mode()
def extract_pair(interface: VoxTellStateInterface, image: np.ndarray, prompts: list[str], gamma_strength: float) -> dict:
    gamma_value = 1.0 + gamma_strength
    original_patch = first_patch(interface.predictor, image)
    gamma_patch = first_patch(interface.predictor, gamma_augment(image, gamma_value))

    original = interface.forward_with_states(original_patch, prompts)
    target = interface.forward_with_states(gamma_patch, prompts)
    return {"original": original, "target": target}


def tensor_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024.0 ** 2)


def compare_candidate_stages(pair: dict, stages: list[str]) -> list[dict[str, float | str]]:
    rows = []
    for stage in stages:
        source = pair["original"]["decoder_states"][stage]
        target = pair["target"]["decoder_states"][stage]
        identity_loss = float(normalized_mse(source, target).detach().cpu())
        pred_source = pair["original"]["intermediate_predictions"][stage]
        pred_target = pair["target"]["intermediate_predictions"][stage]
        prediction_delta = float(normalized_mse(pred_source, pred_target).detach().cpu())
        rows.append({
            "stage": stage,
            "state_shape": "x".join(str(x) for x in source.shape),
            "state_mb": tensor_mb(source),
            "identity_normalized_mse": identity_loss,
            "intermediate_prediction_normalized_mse": prediction_delta,
        })
    return rows


def train_predictor(
    states: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    use_action: bool,
    steps: int,
    hidden_channels: int,
) -> tuple[nn.Module, float]:
    model = VisualWorldPredictor3D(
        in_channels=states.shape[1],
        hidden_channels=hidden_channels,
        use_action=use_action,
    ).to(states.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    state_bn = states.detach().float()
    target_bn = targets.detach().float()
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(state_bn, actions if use_action else None)
        loss = normalized_mse(prediction, target_bn)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        final_loss = normalized_mse(model(state_bn, actions if use_action else None), target_bn)
    return model, float(final_loss.detach().cpu())


def run(args: argparse.Namespace) -> None:
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
    cases = iter_cases(paths, split="train", limit=args.dev_cases)
    image, _, _ = read_image_and_label(cases[0])
    strengths = args.gamma_strengths or [args.gamma_strength]
    pair = extract_pair(interface, image, args.prompts, strengths[0])

    candidate_rows = compare_candidate_stages(pair, args.candidate_stages)
    selected_stage = args.selected_stage
    state_samples = []
    target_samples = []
    action_samples = []
    for strength in strengths:
        strength_pair = pair if strength == strengths[0] else extract_pair(interface, image, args.prompts, strength)
        state_samples.append(strength_pair["original"]["decoder_states"][selected_stage][:, 0])
        target_samples.append(strength_pair["target"]["decoder_states"][selected_stage][:, 0])
        action_samples.append(gamma_action(strength, device))
    states = torch.cat(state_samples, dim=0)
    targets = torch.cat(target_samples, dim=0)
    actions = torch.cat(action_samples, dim=0)

    identity_loss = float(normalized_mse(states, targets).detach().cpu())
    _, agnostic_loss = train_predictor(
        states, targets, actions, use_action=False, steps=args.max_train_steps, hidden_channels=args.hidden_channels
    )
    _, conditioned_loss = train_predictor(
        states, targets, actions, use_action=True, steps=args.max_train_steps, hidden_channels=args.hidden_channels
    )

    candidate_path = output_dir / "candidate_stage_metrics.csv"
    with candidate_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)

    summary = {
        "args": vars(args),
        "dev_case": cases[0].case,
        "selected_stage": selected_stage,
        "gamma_strengths": strengths,
        "identity_normalized_mse": identity_loss,
        "action_agnostic_normalized_mse": agnostic_loss,
        "action_conditioned_normalized_mse": conditioned_loss,
        "candidate_stage_csv": str(candidate_path),
        "note": "Minimal V1 smoke uses one train case and the first sliding-window patch.",
    }
    summary_path = output_dir / "v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
