"""Train a full-window, blur-free hierarchical multi-action World Predictor.

The frozen VoxTell encoder supplies source and real-action target features.
Every native sliding-window tile is visited once per epoch. Gradients are
accumulated over all tiles/actions in a case before one optimizer update, so
large volumes do not dominate only because they contain more tiles.

V0-screened visual actions are used. Intensity scale/shift are excluded because
VoxTell preprocessing nearly cancels them. Random noise/bias realizations are
stable per case/action/seed, which makes a run reproducible; their WP transition
quality must still be checked because the realization is not fully observable
from a compact action vector.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import ProjectPaths
from vls.data import iter_cases, read_image
from torch import nn

from vls.v2_experiment import padded_image_and_slicers, resolve_device
from vls.v7_0d_protocol_sanity import set_seed
from vls.v3_language_experiment import flatten_prompt_embedding
from vls.v9_2_encoder_transition_diagnostic import encode_patch
from vls.v9_3_hierarchical_residual_world_predictor import (
    LEVEL_NAMES,
    hierarchical_loss,
    selected_level_names,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D
from vls_2.v0_mri_action_screening import ActionVariant, apply_action


OUTPUT_DIR = Path("output_2/v1_multi_action_world_predictor")
ACTION_DIM = 4


@dataclass(frozen=True)
class WpAction:
    family: str
    strength: float

    @property
    def label(self) -> str:
        return f"{self.family}:{self.strength:+.2f}"

    @property
    def v0_variant(self) -> ActionVariant:
        if self.family == "gamma":
            return ActionVariant(
                "gamma",
                "mild" if abs(self.strength) < 0.2 else "moderate",
                "brighten" if self.strength < 0 else "darken",
                "gamma",
                1.0 + self.strength,
            )
        if self.family == "gaussian_noise":
            return ActionVariant(
                "gaussian_noise",
                "mild" if self.strength < 0.04 else "moderate",
                "fixed",
                "std_range_fraction",
                self.strength,
            )
        if self.family == "bias_field":
            return ActionVariant(
                "bias_field",
                "mild" if self.strength < 0.10 else "moderate",
                "fixed",
                "amplitude",
                self.strength,
            )
        raise ValueError(f"Unsupported action family: {self.family}")


DEFAULT_ACTIONS = (
    "gamma:-0.30",
    "gamma:+0.30",
    "gaussian_noise:+0.02",
    "gaussian_noise:+0.05",
    "bias_field:+0.05",
    "bias_field:+0.15",
)
ALLOWED_ACTIONS = frozenset(DEFAULT_ACTIONS)
FAMILY_INDEX = {"gamma": 0, "gaussian_noise": 1, "bias_field": 2}
FAMILY_SCALE = {"gamma": 0.30, "gaussian_noise": 0.05, "bias_field": 0.15}


def zero_text_delta(
    interface: VoxTellStateInterface,
    device: torch.device,
) -> torch.Tensor:
    """Keep the inherited language-conditioning branch exactly neutral."""
    source_embedding = interface.embed_text_prompts(["liver"])
    flattened = flatten_prompt_embedding(source_embedding, 0)
    return torch.zeros_like(flattened, device=device)[None]


class MultiActionHierarchicalResidualWorldPredictor(nn.Module):
    """V9.3 residual WP bank with a 4D multi-family action code."""

    def __init__(
        self,
        level_channels: dict[str, int],
        hidden_channels: int,
        text_delta_dim: int,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.level_names = tuple(level_channels)
        self.level_channels = dict(level_channels)
        self.action_dim = int(action_dim)
        self.predictors = nn.ModuleDict({
            name: VisualWorldPredictor3D(
                in_channels=channels,
                hidden_channels=hidden_channels,
                action_dim=self.action_dim,
                num_blocks=2,
                use_action=True,
                text_delta_dim=text_delta_dim,
                use_language=True,
                allow_unconditioned=True,
            )
            for name, channels in level_channels.items()
        })

    def forward(
        self,
        features: dict[str, torch.Tensor],
        action: torch.Tensor,
        text_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tuple(features) != self.level_names:
            raise AssertionError(
                f"Feature level order mismatch: {tuple(features)} vs {self.level_names}"
            )
        if int(action.shape[-1]) != self.action_dim:
            raise AssertionError(
                f"Expected action_dim={self.action_dim}, got {tuple(action.shape)}"
            )
        predictions: dict[str, torch.Tensor] = {}
        for name in self.level_names:
            predictor = self.predictors[name]
            state = features[name]
            x = predictor.input_projection(state.float())
            action_bias = predictor.action_mlp(action.float()).type_as(x)
            language_bias = predictor.language_action_encoder(text_delta.float()).type_as(x)
            scale, bias = (action_bias + language_bias).chunk(2, dim=1)
            x = x * (1.0 + scale[:, :, None, None, None]) + bias[:, :, None, None, None]
            residual = predictor.output_projection(predictor.blocks(x))
            predictions[name] = state.float() + residual
        return predictions


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(
        description="Full-window hierarchical WP for V0-screened visual actions."
    )
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--actions",
        default=",".join(DEFAULT_ACTIONS),
        help="Comma-separated family:strength actions; blur/scale/shift are forbidden.",
    )
    parser.add_argument("--bias-sigma-fraction", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-cases", type=int, default=0, help="0 uses all train cases")
    parser.add_argument("--patch-limit-per-case", type=int, default=0, help="debug only; 0 uses every tile")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Use one case, one tile, one epoch; never a formal result.",
    )
    return parser.parse_args()


def make_paths(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )


def serialize_slicer(slicer: tuple[slice, ...]) -> list[list[int | None]]:
    return [[item.start, item.stop, item.step] for item in slicer]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_actions(value: str) -> tuple[WpAction, ...]:
    actions: list[WpAction] = []
    labels: list[str] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        family, separator, strength = raw.partition(":")
        if separator != ":":
            raise ValueError(f"Action must be family:strength, got {raw!r}")
        action = WpAction(family.strip(), float(strength))
        canonical = f"{action.family}:{action.strength:+.2f}"
        labels.append(canonical)
        actions.append(action)
    if len(actions) < 2:
        raise ValueError("At least two actions are required for uncertainty estimation")
    if len(set(labels)) != len(labels):
        raise ValueError("--actions contains duplicates")
    unsupported = sorted(set(labels) - ALLOWED_ACTIONS)
    if unsupported:
        raise ValueError(
            f"Unsupported V1 actions {unsupported}; allowed={sorted(ALLOWED_ACTIONS)}"
        )
    return tuple(actions)


def action_vector(action: WpAction, device: torch.device) -> torch.Tensor:
    values = [0.0] * ACTION_DIM
    values[FAMILY_INDEX[action.family]] = 1.0
    values[-1] = float(action.strength / FAMILY_SCALE[action.family])
    return torch.tensor([values], dtype=torch.float32, device=device)


def assert_same_topology(
    source_slicers: list[tuple],
    action_slicers: list[tuple],
    case_name: str,
    action: WpAction,
) -> None:
    source_serialized = [serialize_slicer(item) for item in source_slicers]
    action_serialized = [serialize_slicer(item) for item in action_slicers]
    if source_serialized != action_serialized:
        raise AssertionError(
            f"Action changed sliding-window topology for {case_name}, action={action.label}"
        )


def source_tensor_and_slicers(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    case_name: str,
    patch_limit: int,
) -> tuple[torch.Tensor, list[tuple]]:
    source_padded, slicers = padded_image_and_slicers(interface.predictor, image)
    if patch_limit:
        slicers = slicers[:patch_limit]
    if not slicers:
        raise AssertionError(f"No sliding-window tiles for {case_name}")
    return source_padded, slicers


def action_tensor_and_slicers(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    case_name: str,
    action: WpAction,
    seed: int,
    bias_sigma_fraction: float,
) -> tuple[torch.Tensor, list[tuple]]:
    action_image = apply_action(
        image,
        action.v0_variant,
        case_name=case_name,
        seed=seed,
        bias_sigma_fraction=bias_sigma_fraction,
    )
    padded, slicers = padded_image_and_slicers(interface.predictor, action_image)
    return padded, slicers


def infer_level_channels(
    interface: VoxTellStateInterface,
    case: Any,
) -> tuple[tuple[str, ...], dict[str, int]]:
    image, _ = read_image(case)
    source_padded, slicers = source_tensor_and_slicers(
        interface, image, case.case, patch_limit=1,
    )
    patch = torch.clone(source_padded[slicers[0]][None], memory_format=torch.contiguous_format)
    features = encode_patch(interface, patch)
    names = selected_level_names(len(features))
    channels = {
        name: int(features[index].shape[1])
        for index, name in enumerate(names, start=1)
    }
    del image, source_padded, patch, features
    return names, channels


def train_case(
    interface: VoxTellStateInterface,
    model: MultiActionHierarchicalResidualWorldPredictor,
    optimizer: torch.optim.Optimizer,
    case: Any,
    actions: tuple[WpAction, ...],
    level_names: tuple[str, ...],
    text_delta: torch.Tensor,
    device: torch.device,
    patch_limit: int,
    seed: int,
    bias_sigma_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image, _ = read_image(case)
    source_padded, slicers = source_tensor_and_slicers(
        interface, image, case.case, patch_limit,
    )
    optimizer.zero_grad(set_to_none=True)
    action_losses: dict[str, list[float]] = {action.label: [] for action in actions}
    manifest_rows = [
        {
            "case": case.case,
            "patch_index": patch_index,
            "slicer": serialize_slicer(slicer),
        }
        for patch_index, slicer in enumerate(slicers)
    ]
    denominator = float(len(slicers) * len(actions))

    # Stream one action volume at a time. Holding all six preprocessed 3D
    # volumes simultaneously can exhaust host RAM on large cases.
    for action in actions:
        action_padded, action_slicers = action_tensor_and_slicers(
            interface,
            image,
            case.case,
            action,
            seed,
            bias_sigma_fraction,
        )
        assert_same_topology(slicers, action_slicers[: len(slicers)], case.case, action)
        if tuple(action_padded.shape) != tuple(source_padded.shape):
            raise AssertionError(
                f"Action changed preprocessed shape for {case.case}, {action.label}: "
                f"{tuple(source_padded.shape)} -> {tuple(action_padded.shape)}"
            )
        for slicer in slicers:
            source_patch = torch.clone(
                source_padded[slicer][None], memory_format=torch.contiguous_format,
            )
            source_all = encode_patch(interface, source_patch)
            source_features = {
                name: source_all[index].detach().clone()
                for index, name in enumerate(level_names, start=1)
            }
            target_patch = torch.clone(
                action_padded[slicer][None], memory_format=torch.contiguous_format,
            )
            target_all = encode_patch(interface, target_patch)
            targets = {
                name: target_all[index].detach()
                for index, name in enumerate(level_names, start=1)
            }
            predictions = model(
                source_features,
                action_vector(action, device),
                text_delta,
            )
            loss = hierarchical_loss(predictions, targets)
            (loss / denominator).backward()
            action_losses[action.label].append(float(loss.detach().cpu()))
            del target_patch, target_all, targets, predictions, loss
            del source_patch, source_all, source_features
        del action_padded

    gradient_norm = float(torch.sqrt(sum(
        parameter.grad.detach().float().pow(2).sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    )).detach().cpu())
    optimizer.step()
    all_losses = [value for values in action_losses.values() for value in values]
    row: dict[str, Any] = {
        "case": case.case,
        "patch_count": len(slicers),
        "action_count": len(actions),
        "mean_normalized_mse": float(np.mean(all_losses, dtype=np.float64)),
        "gradient_norm": gradient_norm,
        "optimizer_updates": 1,
    }
    for label, values in action_losses.items():
        safe_label = label.replace(":", "_").replace("+", "plus").replace("-", "minus")
        row[f"{safe_label}_mean_normalized_mse"] = float(np.mean(values, dtype=np.float64))

    del image, source_padded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, manifest_rows


def run(args: argparse.Namespace) -> None:
    if args.smoke_only:
        args.train_cases = 1
        args.patch_limit_per_case = 1
        args.epochs = 1
    if args.epochs <= 0:
        raise AssertionError("--epochs must be positive")
    if args.patch_limit_per_case < 0:
        raise AssertionError("--patch-limit-per-case must be non-negative")
    actions = parse_actions(args.actions)
    if not 0.0 < args.bias_sigma_fraction < 0.5:
        raise AssertionError("--bias-sigma-fraction must be in (0,0.5)")

    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V1 multi-action WP requires CUDA, resolved {device}")
    paths = make_paths(args)
    train_cases = iter_cases(
        paths, split="train", limit=args.train_cases if args.train_cases else None,
    )
    if not train_cases:
        raise AssertionError("No train cases")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    text_delta = zero_text_delta(interface, device)
    level_names, level_channels = infer_level_channels(interface, train_cases[0])
    if level_names != LEVEL_NAMES:
        raise AssertionError(f"Unexpected encoder levels: {level_names}")
    model = MultiActionHierarchicalResidualWorldPredictor(
        level_channels=level_channels,
        hidden_channels=args.hidden_channels,
        text_delta_dim=int(text_delta.shape[-1]),
        action_dim=ACTION_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    rng = np.random.default_rng(args.seed)
    curve_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_cases)).tolist()
        epoch_rows: list[dict[str, Any]] = []
        epoch_manifest: list[dict[str, Any]] = []
        for case_position, case_index in enumerate(order, start=1):
            case = train_cases[case_index]
            case_row, case_manifest = train_case(
                interface, model, optimizer, case, actions, level_names,
                text_delta, device, args.patch_limit_per_case, args.seed,
                args.bias_sigma_fraction,
            )
            case_row.update({"epoch": epoch, "case_position": case_position})
            epoch_rows.append(case_row)
            if epoch == 1:
                epoch_manifest.extend(case_manifest)
            print(
                f"[v1_multi_action_wp] epoch={epoch}/{args.epochs} "
                f"case={case_position}/{len(train_cases)} {case.case} "
                f"patches={case_row['patch_count']} loss={case_row['mean_normalized_mse']:.6f}",
                flush=True,
            )
        epoch_loss = float(np.mean(
            [row["mean_normalized_mse"] for row in epoch_rows], dtype=np.float64,
        ))
        for row in epoch_rows:
            row["epoch_mean_normalized_mse"] = epoch_loss
        curve_rows.extend(epoch_rows)
        if epoch == 1:
            manifest_rows = epoch_manifest
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        write_csv(output_dir / "training_curve.csv", curve_rows)

    if best_state is None:
        raise AssertionError("No World Predictor checkpoint selected")
    per_case_patch_counts: dict[str, int] = {}
    for row in manifest_rows:
        per_case_patch_counts[row["case"]] = per_case_patch_counts.get(row["case"], 0) + 1
    checkpoint = {
        "stage": "V1 full-window multi-action World Predictor",
        "selected_epoch": best_epoch,
        "selected_encoder_levels": list(level_names),
        "level_channels": level_channels,
        "hidden_channels": args.hidden_channels,
        "text_delta_dim": int(text_delta.shape[-1]),
        "action_dim": ACTION_DIM,
        "action_protocol": {
            "actions": [action.label for action in actions],
            "families": sorted({action.family for action in actions}),
            "includes_blur": False,
            "language_action_used": False,
            "excluded_as_preprocessing_invariant": ["intensity_scale", "intensity_shift"],
            "stochastic_realization": {
                "gaussian_noise": "stable per case/action/seed",
                "bias_field": "stable per case/action/seed",
                "warning": (
                    "compact action code does not reveal the full random field; "
                    "inspect per-action WP identity bias"
                ),
            },
        },
        "patch_protocol": {
            "full_native_sliding_window": args.patch_limit_per_case == 0,
            "case_balanced_optimizer_step": True,
            "patch_limit_per_case": args.patch_limit_per_case,
            "patch_count": len(manifest_rows),
            "per_case_patch_counts": per_case_patch_counts,
        },
        "state_dict": best_state,
        "train_case_count": len(train_cases),
        "train_cases": [case.case for case in train_cases],
        "smoke_only": bool(args.smoke_only),
    }
    checkpoint_path = output_dir / "best_multi_action_world_predictor.pt"
    torch.save(checkpoint, checkpoint_path)
    (output_dir / "train_patch_manifest.json").write_text(json.dumps({
        "full_native_sliding_window": args.patch_limit_per_case == 0,
        "records": manifest_rows,
    }, indent=2))
    summary = {
        "stage": checkpoint["stage"],
        "status": "smoke_complete" if args.smoke_only else "complete",
        "models_trained": True,
        "voxtell_frozen": True,
        "blur_used": False,
        "actions": [action.label for action in actions],
        "action_families": sorted({action.family for action in actions}),
        "bias_sigma_fraction": args.bias_sigma_fraction,
        "train_case_count": len(train_cases),
        "train_cases": checkpoint["train_cases"],
        "epochs": args.epochs,
        "selected_epoch": best_epoch,
        "best_case_macro_normalized_mse": best_loss,
        "patch_protocol": checkpoint["patch_protocol"],
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "updates_per_epoch": len(train_cases),
        },
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "training_curve": str(output_dir / "training_curve.csv"),
            "train_patch_manifest": str(output_dir / "train_patch_manifest.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

'''
python -u vls_2/v1_train_multi_action_world_predictor.py \
  --gpu 0 \
  --epochs 5 \
  --actions gamma:-0.30,gamma:+0.30,gaussian_noise:+0.02,gaussian_noise:+0.05,bias_field:+0.05,bias_field:+0.15 \
  --output-dir output_2/v1_multi_action_world_predictor
'''
