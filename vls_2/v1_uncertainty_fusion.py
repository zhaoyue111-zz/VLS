"""Full-window uncertainty-gated pseudo-label fusion for VoxTell SFDA.

A frozen, blur-free multi-action World Predictor produces multiple imagined
predictions. Their voxel-wise variance gates a soft interpolation from the
original hard source pseudo-label to the multi-prediction mean. No TP/FP/FN
routing rule or prompt action is used during adaptation. GT is
diagnostic/evaluation only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import sys
import types
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths


OUTPUT_DIR = Path("output_2/v1_uncertainty_weighted_pseudo")
DEFAULT_WP_ACTIONS = (
    "gamma:-0.30",
    "gamma:+0.30",
    "gaussian_noise:+0.02",
    "gaussian_noise:+0.05",
    "bias_field:+0.05",
    "bias_field:+0.15",
)
ALLOWED_WP_ACTIONS = frozenset(DEFAULT_WP_ACTIONS)
FAMILY_INDEX = {"gamma": 0, "gaussian_noise": 1, "bias_field": 2}
FAMILY_SCALE = {"gamma": 0.30, "gaussian_noise": 0.05, "bias_field": 0.15}
ACTION_DIM = 4
VARIANTS = ("B_source_hard_pseudo", "C_uncertainty_weighted_soft_pseudo")


def load_runtime_dependencies() -> None:
    """Delay VoxTell/nnUNet imports until an experiment actually runs."""
    global CaseRecord, SOURCE_PROMPT, LEVEL_NAMES
    global VoxTellStateInterface, MultiActionHierarchicalResidualWorldPredictor
    global zero_text_delta
    global class_balanced_loss, evaluate_full_volume, inject_lora_qkv, iter_cases
    global level_features, lora_parameters, native_decoder_from_predicted_skips
    global pad_label_like_image, padded_image_and_slicers, pool_full_volume
    global read_image_and_label, resolve_device, set_seed

    from vls.data import CaseRecord, iter_cases, read_image_and_label
    from vls.v2_experiment import padded_image_and_slicers, resolve_device
    from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, pad_label_like_image
    from vls.v7_0d_protocol_sanity import set_seed
    from vls.v7_1a_lora_qkv_smoke import inject_lora_qkv, lora_parameters
    from vls.v7_1b_protocol_consolidation import evaluate_full_volume, pool_full_volume
    from vls.v7_1c_class_balanced_loss_sanity import class_balanced_loss
    if sys.platform == "win32" and "resource" not in sys.modules:
        resource_shim = types.ModuleType("resource")
        resource_shim.RUSAGE_SELF = 0
        resource_shim.getrusage = lambda _who: types.SimpleNamespace(ru_maxrss=0)
        sys.modules["resource"] = resource_shim
    from vls.v9_3_hierarchical_residual_world_predictor import (
        LEVEL_NAMES,
        level_features,
        native_decoder_from_predicted_skips,
    )
    from vls.voxtell_states import VoxTellStateInterface
    from vls_2.v1_train_multi_action_world_predictor import (
        MultiActionHierarchicalResidualWorldPredictor,
        zero_text_delta,
    )


@dataclass(frozen=True)
class WpAction:
    family: str
    strength: float

    @property
    def label(self) -> str:
        return f"{self.family}:{self.strength:+.2f}"


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(
        description="Full-window uncertainty-gated pseudo-label fusion."
    )
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default=(
            "output_2/v1_multi_action_world_predictor/"
            "best_multi_action_world_predictor.pt"
        ),
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--wp-actions", default=",".join(DEFAULT_WP_ACTIONS))
    parser.add_argument("--train-cases", type=int, default=0, help="0 uses all train cases")
    parser.add_argument("--evaluation-cases", type=int, default=0, help="0 uses all test cases")
    parser.add_argument("--training-epochs", type=int, default=1)
    parser.add_argument("--patch-limit-per-case", type=int, default=0, help="debug only; 0 uses every tile")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--calibration-percentile", type=float, default=99.0)
    parser.add_argument("--calibration-voxels-per-patch", type=int, default=4096)
    parser.add_argument("--minimum-calibration-scale", type=float, default=1e-4)
    parser.add_argument("--uncertainty-power", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Use one train/eval case, one tile, one epoch; never a formal result.",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Measure per-action/family WP disagreement on train tiles, then stop before SFDA.",
    )
    return parser.parse_args()


def make_paths(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )


def parse_wp_actions(value: str) -> list[WpAction]:
    actions: list[WpAction] = []
    labels: list[str] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        family, separator, strength = raw.partition(":")
        if separator != ":":
            raise ValueError(f"WP action must be family:strength, got {raw!r}")
        action = WpAction(family.strip(), float(strength))
        labels.append(action.label)
        actions.append(action)
    if len(actions) < 2:
        raise ValueError("At least two distinct WP actions are required")
    if len(set(labels)) != len(labels):
        raise ValueError("--wp-actions contains duplicates")
    unsupported = sorted(set(labels) - ALLOWED_WP_ACTIONS)
    if unsupported:
        raise ValueError(
            f"Unsupported V1 actions {unsupported}; allowed={sorted(ALLOWED_WP_ACTIONS)}"
        )
    return actions


def action_vector(action: WpAction, device: torch.device) -> torch.Tensor:
    values = [0.0] * ACTION_DIM
    values[FAMILY_INDEX[action.family]] = 1.0
    values[-1] = float(action.strength / FAMILY_SCALE[action.family])
    return torch.tensor([values], dtype=torch.float32, device=device)


def serialize_slicer(slicer: tuple[slice, ...]) -> list[list[int | None]]:
    return [[item.start, item.stop, item.step] for item in slicer]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def load_multi_action_world_predictor(
    checkpoint_path: Path,
    actions: list[WpAction],
    device: torch.device,
    allow_smoke: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Multi-action WP checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "V1 full-window multi-action World Predictor":
        raise AssertionError(f"Unexpected checkpoint stage: {checkpoint.get('stage')}")
    protocol = checkpoint.get("action_protocol", {})
    if protocol.get("includes_blur") is not False:
        raise AssertionError(f"Checkpoint is not blur-free: {protocol}")
    if protocol.get("language_action_used") is not False:
        raise AssertionError(f"Checkpoint contains a language/prompt action: {protocol}")
    trained = tuple(str(value) for value in protocol.get("actions", ()))
    requested = tuple(action.label for action in actions)
    if trained != requested:
        raise AssertionError(
            "Requested actions must exactly match checkpoint training actions; "
            f"trained={trained}, requested={requested}"
        )
    if checkpoint.get("smoke_only") and not allow_smoke:
        raise AssertionError("A smoke-only WP checkpoint cannot be used for a formal fusion run")
    levels = tuple(checkpoint.get("selected_encoder_levels", ()))
    if levels != LEVEL_NAMES:
        raise AssertionError(f"Checkpoint levels differ from V1: {levels}")
    level_channels = {str(name): int(value) for name, value in checkpoint["level_channels"].items()}
    if int(checkpoint.get("action_dim", -1)) != ACTION_DIM:
        raise AssertionError(
            f"Checkpoint action_dim={checkpoint.get('action_dim')} != {ACTION_DIM}"
        )
    model = MultiActionHierarchicalResidualWorldPredictor(
        level_channels=level_channels,
        hidden_channels=int(checkpoint["hidden_channels"]),
        text_delta_dim=int(checkpoint["text_delta_dim"]),
        action_dim=ACTION_DIM,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "path": str(checkpoint_path),
        "stage": checkpoint["stage"],
        "selected_epoch": checkpoint.get("selected_epoch"),
        "action_protocol": protocol,
        "patch_protocol": checkpoint.get("patch_protocol"),
        "train_case_count": checkpoint.get("train_case_count"),
        "smoke_only": bool(checkpoint.get("smoke_only")),
    }
    return model, metadata


@contextmanager
def lora_disabled(network: torch.nn.Module):
    """Temporarily recover the unchanged source teacher from a LoRA student."""
    modules: list[tuple[torch.nn.Module, float]] = []
    for module in network.modules():
        if all(hasattr(module, name) for name in ("lora_A_q", "lora_B_q", "scaling")):
            modules.append((module, float(module.scaling)))
            module.scaling = 0.0
    try:
        yield
    finally:
        for module, scaling in modules:
            module.scaling = scaling


@torch.inference_mode()
def source_context_from_network(
    interface: VoxTellStateInterface,
    network: torch.nn.Module,
    patch: torch.Tensor,
    embedding: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    network.eval()
    autocast = torch.autocast(device.type, enabled=True) if device.type == "cuda" else nullcontext()
    with autocast:
        return interface._network_forward_with_audit_context(
            network, patch.to(device), embedding.to(device).float(),
        )


@torch.inference_mode()
def imagined_predictions(
    interface: VoxTellStateInterface,
    world_predictor: torch.nn.Module,
    source_context: dict[str, Any],
    actions: list[WpAction],
    text_delta: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[WpAction, torch.Tensor]]]:
    source_probability = torch.sigmoid(source_context["final_prediction"][:, :1].detach().float())
    source_features = level_features(source_context["decoder_audit"]["skips"], LEVEL_NAMES)
    outputs: list[tuple[WpAction, torch.Tensor]] = []
    for action in actions:
        predicted_features = world_predictor(
            source_features, action_vector(action, device), text_delta,
        )
        logits = native_decoder_from_predicted_skips(
            interface, source_context, predicted_features, LEVEL_NAMES,
        )
        outputs.append((action, torch.sigmoid(logits.detach().float())))
        del predicted_features, logits
    return source_probability, outputs


def prediction_moments(
    predictions: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Population variance and mean without allocating a full N-volume stack."""
    if len(predictions) < 2:
        raise AssertionError("At least two predictions are required")
    mean = torch.zeros_like(predictions[0], dtype=torch.float32)
    for prediction in predictions:
        mean.add_(prediction.float(), alpha=1.0 / len(predictions))
    variance = torch.zeros_like(mean)
    for prediction in predictions:
        variance.add_((prediction.float() - mean).square(), alpha=1.0 / len(predictions))
    return variance, mean


def uncertainty_and_pseudo(
    predictions: list[torch.Tensor],
    source_threshold: float,
    calibration_scale: float,
    uncertainty_power: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(predictions) < 3:
        raise AssertionError("Expected source plus at least two action predictions")
    raw, ensemble = prediction_moments(predictions)
    normalized = torch.clamp(raw / max(calibration_scale, 1e-12), 0.0, 1.0)
    gate = normalized.pow(uncertainty_power)
    source_hard = (predictions[0] > source_threshold).float()
    pseudo = (1.0 - gate) * source_hard + gate * ensemble
    return raw, gate, pseudo.clamp(0.0, 1.0)


def case_setup(
    interface: VoxTellStateInterface,
    case: CaseRecord,
    patch_limit: int,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, list[tuple]]:
    image, label, _ = read_image_and_label(case)
    padded, slicers = padded_image_and_slicers(interface.predictor, image)
    if patch_limit:
        slicers = slicers[:patch_limit]
    if not slicers:
        raise AssertionError(f"No native sliding-window tiles for {case.case}")
    label_padded = pad_label_like_image(interface, label)
    return image, label, padded, label_padded, slicers


def overlap_weights(
    interface: VoxTellStateInterface,
    padded: torch.Tensor,
    slicers: list[tuple],
) -> list[torch.Tensor]:
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    gaussian = compute_gaussian(
        tuple(interface.predictor.patch_size), sigma_scale=1.0 / 8,
        value_scaling_factor=10, device=torch.device("cpu"),
    ).float()
    coverage = torch.zeros(tuple(int(size) for size in padded.shape[1:]), dtype=torch.float32)
    for slicer in slicers:
        coverage[slicer[1:]] += gaussian
    return [
        (gaussian / coverage[slicer[1:]].clamp_min(1e-12))[None, None]
        for slicer in slicers
    ]


def calibration_sample(values: torch.Tensor, count: int, rng: np.random.Generator) -> np.ndarray:
    flat = values.detach().flatten().cpu().numpy().astype(np.float32)
    if len(flat) <= count:
        return flat
    return flat[rng.choice(len(flat), size=count, replace=False)]


@torch.inference_mode()
def calibrate_uncertainty(
    interface: VoxTellStateInterface,
    network: torch.nn.Module,
    world_predictor: torch.nn.Module,
    cases: list[CaseRecord],
    embedding: torch.Tensor,
    text_delta: torch.Tensor,
    actions: list[WpAction],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(args.seed)
    sampled_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    prediction_count_by_group: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        image, label, padded, label_padded, slicers = case_setup(
            interface, case, args.patch_limit_per_case,
        )
        for patch_index, slicer in enumerate(slicers):
            patch = torch.clone(padded[slicer][None], memory_format=torch.contiguous_format)
            context = source_context_from_network(interface, network, patch, embedding, device)
            source_probability, action_outputs = imagined_predictions(
                interface, world_predictor, context, actions, text_delta, device,
            )
            grouped_predictions: dict[str, list[torch.Tensor]] = defaultdict(
                lambda: [source_probability]
            )
            grouped_predictions["combined"] = [source_probability]
            for action, prediction in action_outputs:
                grouped_predictions[action.family].append(prediction)
                grouped_predictions["combined"].append(prediction)
                grouped_predictions[f"action/{action.label}"] = [
                    source_probability, prediction,
                ]
            for group, predictions in grouped_predictions.items():
                prediction_count_by_group[group] = len(predictions)
                raw, group_mean = prediction_moments(predictions)
                sampled_by_group[group].append(calibration_sample(
                    raw, args.calibration_voxels_per_patch, rng,
                ))
                del raw, group_mean
            manifest.append({
                "case": case.case,
                "case_index": case_index,
                "patch_index": patch_index,
                "slicer": serialize_slicer(slicer),
            })
            del patch, context, source_probability, action_outputs, grouped_predictions
        del image, label, padded, label_padded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    group_values = {
        group: np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        for group, parts in sampled_by_group.items()
    }
    values = group_values["combined"]
    if not len(values):
        raise AssertionError("No uncertainty values collected")
    empirical_scale = float(np.percentile(values, args.calibration_percentile))
    raw_max = float(np.max(values))
    if not np.isfinite(empirical_scale) or not np.isfinite(raw_max) or raw_max <= 1e-12:
        raise AssertionError(
            "WP disagreement is numerically zero; pseudo-label fusion is not meaningful"
        )
    scale = max(empirical_scale, args.minimum_calibration_scale)
    stats = {
        "calibration_scope": "all train cases and all native sliding-window tiles",
        "percentile": args.calibration_percentile,
        "scale": scale,
        "empirical_percentile_scale": empirical_scale,
        "minimum_scale": args.minimum_calibration_scale,
        "minimum_scale_applied": bool(empirical_scale < args.minimum_calibration_scale),
        "sample_count": int(len(values)),
        "patch_count": len(manifest),
        "raw_mean": float(np.mean(values, dtype=np.float64)),
        "raw_std": float(np.std(values, dtype=np.float64)),
        "raw_p50": float(np.percentile(values, 50)),
        "raw_p90": float(np.percentile(values, 90)),
        "raw_p95": float(np.percentile(values, 95)),
        "raw_p99": float(np.percentile(values, 99)),
        "group_uncertainty": {
            group_name: {
                "prediction_count": prediction_count_by_group[group_name],
                "raw_mean": float(np.mean(values_for_group, dtype=np.float64)),
                "raw_p95": float(np.percentile(values_for_group, 95)),
                "raw_p99": float(np.percentile(values_for_group, 99)),
            }
            for group_name, values_for_group in group_values.items()
        },
        "per_patch_minmax_used": False,
        "gt_used": False,
    }
    return scale, stats, manifest


def empty_diagnostic() -> dict[str, float]:
    return defaultdict(float)


def update_diagnostic(
    accumulator: dict[str, float],
    source_hard: torch.Tensor,
    pseudo: torch.Tensor,
    gt: torch.Tensor,
    raw: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    source = source_hard.detach().flatten().cpu().numpy().astype(bool)
    fused = pseudo.detach().flatten().cpu().numpy() > 0.5
    truth = gt.detach().flatten().cpu().numpy().astype(bool)
    accumulator["voxel_count"] += len(source)
    accumulator["source_correct"] += int(np.count_nonzero(source == truth))
    accumulator["fused_correct"] += int(np.count_nonzero(fused == truth))
    accumulator["source_positive"] += int(np.count_nonzero(source))
    accumulator["fused_positive"] += int(np.count_nonzero(fused))
    accumulator["background_to_foreground"] += int(np.count_nonzero((~source) & fused))
    accumulator["foreground_to_background"] += int(np.count_nonzero(source & (~fused)))
    accumulator["source_intersection"] += int(np.count_nonzero(source & truth))
    accumulator["source_union"] += int(np.count_nonzero(source | truth))
    accumulator["source_dice_den"] += int(np.count_nonzero(source) + np.count_nonzero(truth))
    accumulator["fused_intersection"] += int(np.count_nonzero(fused & truth))
    accumulator["fused_union"] += int(np.count_nonzero(fused | truth))
    accumulator["fused_dice_den"] += int(np.count_nonzero(fused) + np.count_nonzero(truth))
    accumulator["raw_sum"] += float(raw.detach().sum().cpu())
    accumulator["gate_sum"] += float(gate.detach().sum().cpu())
    accumulator["soft_sum"] += float(pseudo.detach().sum().cpu())


def finalize_diagnostic(
    accumulator: dict[str, float], variant: str, epoch: int, case: str,
) -> dict[str, Any]:
    voxels = int(accumulator["voxel_count"])
    source_den, fused_den = int(accumulator["source_dice_den"]), int(accumulator["fused_dice_den"])
    source_union, fused_union = int(accumulator["source_union"]), int(accumulator["fused_union"])
    source_dice = 1.0 if source_den == 0 else 2.0 * accumulator["source_intersection"] / source_den
    fused_dice = 1.0 if fused_den == 0 else 2.0 * accumulator["fused_intersection"] / fused_den
    source_iou = 1.0 if source_union == 0 else accumulator["source_intersection"] / source_union
    fused_iou = 1.0 if fused_union == 0 else accumulator["fused_intersection"] / fused_union
    return {
        "variant": variant,
        "epoch": epoch,
        "case": case,
        "voxel_count_with_overlap": voxels,
        "source_pseudo_accuracy": accumulator["source_correct"] / max(voxels, 1),
        "fused_pseudo_accuracy": accumulator["fused_correct"] / max(voxels, 1),
        "source_pseudo_dice": source_dice,
        "fused_pseudo_dice": fused_dice,
        "source_pseudo_iou": source_iou,
        "fused_pseudo_iou": fused_iou,
        "delta_pseudo_dice_fused_minus_source": fused_dice - source_dice,
        "delta_pseudo_iou_fused_minus_source": fused_iou - source_iou,
        "source_positive_voxels": int(accumulator["source_positive"]),
        "fused_positive_voxels": int(accumulator["fused_positive"]),
        "background_to_foreground_voxels": int(accumulator["background_to_foreground"]),
        "foreground_to_background_voxels": int(accumulator["foreground_to_background"]),
        "raw_uncertainty_mean": accumulator["raw_sum"] / max(voxels, 1),
        "uncertainty_gate_mean": accumulator["gate_sum"] / max(voxels, 1),
        "soft_pseudo_mean": accumulator["soft_sum"] / max(voxels, 1),
        "gt_usage": "diagnostic only",
    }


def prepare_gt_patch(
    label_padded: torch.Tensor,
    slicer: tuple,
    probability: torch.Tensor,
    label_value: int,
    device: torch.device,
) -> torch.Tensor:
    gt = (label_padded[slicer][None].to(device) == label_value).float()
    if tuple(gt.shape[-3:]) != tuple(probability.shape[-3:]):
        import torch.nn.functional as F

        gt = F.interpolate(gt, size=probability.shape[-3:], mode="nearest")
    return gt


def gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    terms = [parameter.grad.detach().float().pow(2).sum() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not terms else float(torch.sqrt(sum(terms)).detach().cpu())


def train_variant(
    variant: str,
    base_network: torch.nn.Module,
    world_predictor: torch.nn.Module,
    train_cases: list[CaseRecord],
    eval_cases: list[CaseRecord],
    full_data: dict[str, tuple[np.ndarray, np.ndarray]],
    interface: VoxTellStateInterface,
    embedding: torch.Tensor,
    text_delta: torch.Tensor,
    actions: list[WpAction],
    calibration_scale: float,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    target_names: list[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(args.seed)
    student = copy.deepcopy(base_network)
    targets = inject_lora_qkv(student, args.lora_rank, args.lora_alpha, args.lora_dropout)
    if target_names is not None and targets != target_names:
        raise AssertionError("LoRA target modules differ across variants")
    student = student.to(device)
    trainable = lora_parameters(student)
    base_trainable = sum(
        parameter.numel() for name, parameter in student.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    if base_trainable:
        raise AssertionError(f"Base VoxTell parameters must remain frozen, got {base_trainable}")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    uses_uncertainty = variant == "C_uncertainty_weighted_soft_pseudo"
    world_predictor.to(device if uses_uncertainty else torch.device("cpu")).eval()

    loss_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    global_step = 0
    for epoch in range(1, args.training_epochs + 1):
        for case_position, case_index in enumerate(rng.permutation(len(train_cases)).tolist(), start=1):
            case = train_cases[case_index]
            image, label, padded, label_padded, slicers = case_setup(
                interface, case, args.patch_limit_per_case,
            )
            coverage_weights = overlap_weights(interface, padded, slicers)
            optimizer.zero_grad(set_to_none=True)
            diagnostic = empty_diagnostic()
            patch_losses: list[float] = []

            for slicer, coverage_weight in zip(slicers, coverage_weights, strict=True):
                patch = torch.clone(padded[slicer][None], memory_format=torch.contiguous_format).to(device)
                with lora_disabled(student):
                    source_context = source_context_from_network(interface, student, patch, embedding, device)
                if uses_uncertainty:
                    source_probability, action_outputs = imagined_predictions(
                        interface, world_predictor, source_context, actions, text_delta, device,
                    )
                    predictions = [
                        source_probability,
                        *(prediction for _, prediction in action_outputs),
                    ]
                    raw, gate, pseudo = uncertainty_and_pseudo(
                        predictions, args.prediction_threshold,
                        calibration_scale, args.uncertainty_power,
                    )
                else:
                    source_probability = torch.sigmoid(
                        source_context["final_prediction"][:, :1].detach().float()
                    )
                    action_outputs = None
                    predictions = None
                    raw, gate = (
                        torch.zeros_like(source_probability),
                        torch.zeros_like(source_probability),
                    )
                source_hard = (source_probability > args.prediction_threshold).float()
                if not uses_uncertainty:
                    pseudo = source_hard
                gt = prepare_gt_patch(label_padded, slicer, source_probability, args.label_value, device)
                update_diagnostic(diagnostic, source_hard, pseudo, gt, raw, gate)

                # Keep VoxTell in inference mode during SFDA; gradients still
                # flow to LoRA parameters, while normalization/dropout behavior
                # remains identical to the frozen source model.
                student.eval()
                autocast = torch.autocast(device.type, enabled=True) if device.type == "cuda" else nullcontext()
                with autocast:
                    result = interface._network_forward_with_states(
                        student, patch, embedding.to(device).float(),
                    )
                    logits = result["final_prediction"][:, :1].float()
                    weight = coverage_weight.to(device=device, dtype=torch.float32).expand_as(pseudo)
                    loss, _ = class_balanced_loss(logits, pseudo.detach(), weight)
                    scaled_loss = loss / float(len(slicers))
                scaled_loss.backward()
                patch_losses.append(float(loss.detach().cpu()))
                del patch, source_context, source_probability, source_hard, raw, gate
                del pseudo, gt, result, logits, weight, loss, scaled_loss
                del action_outputs, predictions

            grad_norm = gradient_norm(trainable)
            before = [parameter.detach().clone() for parameter in trainable]
            optimizer.step()
            update_delta = float(torch.sqrt(sum(
                (parameter.detach() - previous).float().pow(2).sum()
                for parameter, previous in zip(trainable, before, strict=True)
            )).detach().cpu())
            global_step += 1
            loss_rows.append({
                "variant": variant,
                "epoch": epoch,
                "case_position": case_position,
                "step": global_step,
                "case": case.case,
                "patch_count": len(slicers),
                "mean_patch_loss": float(np.mean(patch_losses, dtype=np.float64)),
                "gradient_norm": grad_norm,
                "update_delta_norm": update_delta,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "optimizer_updates": 1,
                "base_trainable_parameters": base_trainable,
                "lora_trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
            })
            pseudo_rows.append(finalize_diagnostic(diagnostic, variant, epoch, case.case))
            print(
                f"[v1_uncertainty_pseudo] {variant} epoch={epoch}/{args.training_epochs} "
                f"case={case_position}/{len(train_cases)} {case.case} "
                f"patches={len(slicers)} loss={loss_rows[-1]['mean_patch_loss']:.6f}",
                flush=True,
            )
            del image, label, padded, label_padded, coverage_weights, before
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    student.eval()
    full_rows = evaluate_full_volume(
        interface, student, eval_cases, full_data, embedding, variant,
        "full_window_case_balanced", global_step,
        args.label_value, args.prediction_threshold,
    )
    for row in full_rows:
        enrich_metric_row(row)
        row["pseudo_label_method"] = "uncertainty_weighted_soft" if uses_uncertainty else "source_hard"
    lora_state = {
        key: value.detach().cpu() for key, value in student.state_dict().items()
        if "lora_" in key
    }
    lora_path = output_dir / f"{variant}_lora.pt"
    torch.save({
        "variant": variant,
        "target_modules": targets,
        "state_dict": lora_state,
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
    }, lora_path)
    stats = {
        "variant": variant,
        "target_modules": targets,
        "base_trainable_parameters": base_trainable,
        "lora_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
        "lora_checkpoint": str(lora_path),
        "updates": global_step,
    }
    del student, optimizer, trainable
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_rows, pseudo_rows, full_rows, stats


def binary_iou_from_counts(metrics: dict[str, Any]) -> float:
    tp, fp, fn = int(metrics["tp"]), int(metrics["fp"]), int(metrics["fn"])
    return 1.0 if tp + fp + fn == 0 else float(tp / max(tp + fp + fn, 1))


def enrich_metric_row(row: dict[str, Any]) -> None:
    iou = binary_iou_from_counts(row)
    row["iou"] = iou
    row["foreground_dice"] = row["dice"]
    row["foreground_iou"] = iou


def aggregate_metrics(
    full_rows: list[dict[str, Any]], pseudo_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    curve = pool_full_volume(full_rows)
    baseline = next(row for row in curve if row["variant"] == "A_init_no_adaptation")
    pseudo_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pseudo_rows:
        pseudo_by_variant[row["variant"]].append(row)
    output: list[dict[str, Any]] = []
    for row in curve:
        variant = row["variant"]
        diagnostics = pseudo_by_variant.get(variant, [])

        def mean_field(field: str) -> float | None:
            return (
                float(np.mean([float(item[field]) for item in diagnostics], dtype=np.float64))
                if diagnostics else None
            )

        output.append({
            **row,
            "mean_iou": row["mean_foreground_iou"],
            "mean_foreground_dice": row["mean_dice"],
            "delta_mean_dice_vs_A_init": row["mean_dice"] - baseline["mean_dice"],
            "delta_mean_iou_vs_A_init": row["mean_foreground_iou"] - baseline["mean_foreground_iou"],
            "source_pseudo_dice_macro": mean_field("source_pseudo_dice"),
            "fused_pseudo_dice_macro": mean_field("fused_pseudo_dice"),
            "delta_pseudo_dice_macro": mean_field("delta_pseudo_dice_fused_minus_source"),
            "background_to_foreground_voxels_mean": mean_field("background_to_foreground_voxels"),
            "foreground_to_background_voxels_mean": mean_field("foreground_to_background_voxels"),
            "uncertainty_gate_mean": mean_field("uncertainty_gate_mean"),
        })
    return output


def add_baseline_deltas(full_rows: list[dict[str, Any]], initial_rows: list[dict[str, Any]]) -> None:
    baseline_by_case = {row["case"]: row for row in initial_rows}
    for row in full_rows:
        baseline = baseline_by_case[row["case"]]
        row["delta_dice_vs_A_init"] = row["dice"] - baseline["dice"]
        row["delta_iou_vs_A_init"] = row["iou"] - baseline["iou"]


def validate_args(args: argparse.Namespace) -> None:
    if args.training_epochs <= 0:
        raise AssertionError("--training-epochs must be positive")
    if args.patch_limit_per_case < 0:
        raise AssertionError("--patch-limit-per-case must be non-negative")
    if not 0.0 < args.calibration_percentile < 100.0:
        raise AssertionError("--calibration-percentile must be in (0,100)")
    if args.calibration_voxels_per_patch <= 0:
        raise AssertionError("--calibration-voxels-per-patch must be positive")
    if args.minimum_calibration_scale <= 0:
        raise AssertionError("--minimum-calibration-scale must be positive")
    if args.uncertainty_power <= 0:
        raise AssertionError("--uncertainty-power must be positive")


def run(args: argparse.Namespace) -> None:
    if args.smoke_only:
        args.train_cases = 1
        args.evaluation_cases = 1
        args.patch_limit_per_case = 1
        args.training_epochs = 1
        args.calibration_voxels_per_patch = min(args.calibration_voxels_per_patch, 1024)
    validate_args(args)
    actions = parse_wp_actions(args.wp_actions)

    load_runtime_dependencies()
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V1 uncertainty pseudo-label fusion requires CUDA, resolved {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = make_paths(args)
    train_cases = iter_cases(paths, split="train", limit=args.train_cases if args.train_cases else None)
    eval_cases = (
        []
        if args.calibration_only
        else iter_cases(
            paths,
            split="test",
            limit=args.evaluation_cases if args.evaluation_cases else None,
        )
    )
    train_names = [case.case for case in train_cases]
    eval_names = [case.case for case in eval_cases]
    overlap = sorted(set(train_names) & set(eval_names))
    if overlap:
        raise AssertionError(f"Train/evaluation case overlap: {overlap}")
    if not train_cases or (not args.calibration_only and not eval_cases):
        raise AssertionError(
            "Non-empty train cases are required; evaluation cases are also required for SFDA"
        )

    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    base_network = interface.network.to(device).eval()
    for parameter in base_network.parameters():
        parameter.requires_grad_(False)
    embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    text_delta = zero_text_delta(interface, device)
    world_predictor, checkpoint_metadata = load_multi_action_world_predictor(
        Path(args.world_checkpoint), actions, device, allow_smoke=args.smoke_only,
    )
    calibration_scale, calibration_stats, manifest = calibrate_uncertainty(
        interface, base_network, world_predictor, train_cases, embedding,
        text_delta, actions, args, device,
    )
    write_json(output_dir / "uncertainty_calibration.json", calibration_stats)
    write_json(output_dir / "train_patch_manifest.json", {
        "selection": "all native sliding-window tiles; no GT selection",
        "full_native_sliding_window": args.patch_limit_per_case == 0,
        "patch_limit_per_case": args.patch_limit_per_case,
        "records": manifest,
    })
    if args.calibration_only:
        diagnostic_summary = {
            "stage": "V1 multi-action WP disagreement calibration",
            "status": "calibration_complete",
            "sfda_run": False,
            "gt_used": False,
            "prompt_actions_used": False,
            "blur_used": False,
            "wp_actions": [action.label for action in actions],
            "world_checkpoint": checkpoint_metadata,
            "uncertainty_calibration": calibration_stats,
        }
        write_json(output_dir / "summary.json", diagnostic_summary)
        print(json.dumps(diagnostic_summary, indent=2), flush=True)
        return

    full_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in eval_cases:
        image, label, _ = read_image_and_label(case)
        full_data[case.case] = (image, label)
    initial_rows = evaluate_full_volume(
        interface, base_network, eval_cases, full_data, embedding,
        "A_init_no_adaptation", "full_window_case_balanced", 0,
        args.label_value, args.prediction_threshold,
    )
    for row in initial_rows:
        enrich_metric_row(row)
        row["pseudo_label_method"] = "none"

    base_network.to("cpu")
    world_predictor.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    all_loss_rows: list[dict[str, Any]] = []
    all_pseudo_rows: list[dict[str, Any]] = []
    all_full_rows: list[dict[str, Any]] = list(initial_rows)
    parameter_stats: list[dict[str, Any]] = []
    target_names: list[str] | None = None
    for variant in VARIANTS:
        losses, pseudo_rows, full_rows, stats = train_variant(
            variant, base_network, world_predictor, train_cases, eval_cases,
            full_data, interface, embedding, text_delta, actions,
            calibration_scale, output_dir, args, device, target_names,
        )
        target_names = stats["target_modules"] if target_names is None else target_names
        all_loss_rows.extend(losses)
        all_pseudo_rows.extend(pseudo_rows)
        all_full_rows.extend(full_rows)
        parameter_stats.append(stats)

    add_baseline_deltas(all_full_rows, initial_rows)
    metrics_rows = aggregate_metrics(all_full_rows, all_pseudo_rows)
    write_rows(output_dir / "training_loss.csv", all_loss_rows)
    write_rows(output_dir / "pseudo_label_stats.csv", all_pseudo_rows)
    write_rows(output_dir / "per_case.csv", all_full_rows)
    write_rows(output_dir / "metrics.csv", metrics_rows)
    write_json(output_dir / "parameter_stats.json", {"variants": parameter_stats})

    summary = {
        "stage": "V1 full-window uncertainty-gated pseudo-label fusion",
        "status": "smoke_complete" if args.smoke_only else "complete",
        "smoke_only": bool(args.smoke_only),
        "output_dir": str(output_dir),
        "source_voxtell_modified": False,
        "world_predictor_updated_during_adaptation": False,
        "blur_used": False,
        "prompt_actions_used": False,
        "train_cases": train_names,
        "evaluation_cases": eval_names,
        "case_overlap": overlap,
        "world_checkpoint": checkpoint_metadata,
        "wp_actions": [action.label for action in actions],
        "wp_action_families": sorted({action.family for action in actions}),
        "variants": list(VARIANTS),
        "formula": {
            "prediction_stack": (
                "[source:liver + imagined predictions for every configured "
                "gamma/Gaussian-noise/bias-field action]"
            ),
            "raw_uncertainty": "U = population variance(prediction_stack)",
            "global_calibration": (
                f"scale=max(train_global_p{args.calibration_percentile:g},"
                f"{args.minimum_calibration_scale:g}); "
                f"U_gate=clip(U/scale,0,1)^{args.uncertainty_power:g}"
            ),
            "source_pseudo": f"Y_source = 1[source_probability>{args.prediction_threshold:g}]",
            "ensemble_probability": "P_ensemble = mean(prediction_stack)",
            "uncertainty_weighted_pseudo": "Y_soft=(1-U_gate)*Y_source+U_gate*P_ensemble",
            "adaptation_loss": "class-balanced BCE(logits,Y_soft,inverse-overlap Gaussian weight)",
        },
        "patch_protocol": {
            "full_native_sliding_window": args.patch_limit_per_case == 0,
            "patch_limit_per_case": args.patch_limit_per_case,
            "optimizer_step": "one update after gradient accumulation over every tile in one case",
            "case_balanced": True,
            "overlap_weighting": "Gaussian tile importance divided by accumulated case coverage",
            "gt_patch_selection": False,
        },
        "uncertainty_calibration": calibration_stats,
        "training": {
            "epochs": args.training_epochs,
            "base_voxtell_frozen": True,
            "trainable_scope": "LoRA packed QKV only",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_names,
        },
        "gt_usage": "pseudo-label diagnostics and final full-volume evaluation only",
        "final_metrics": metrics_rows,
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "metrics": str(output_dir / "metrics.csv"),
            "per_case": str(output_dir / "per_case.csv"),
            "pseudo_label_stats": str(output_dir / "pseudo_label_stats.csv"),
            "training_loss": str(output_dir / "training_loss.csv"),
            "train_patch_manifest": str(output_dir / "train_patch_manifest.json"),
            "uncertainty_calibration": str(output_dir / "uncertainty_calibration.json"),
            "parameter_stats": str(output_dir / "parameter_stats.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(output_dir),
        "variants": [row["variant"] for row in metrics_rows],
    }, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

'''
只检查每个 action 是否仍接近 identity，不运行 SFDA
python -u vls_2/v1_uncertainty_fusion.py \
  --calibration-only \
  --gpu 0 \
  --world-checkpoint output_2/v1_multi_action_world_predictor/best_multi_action_world_predictor.pt \
  --output-dir output_2/v1_multi_action_calibration
'''