"""Vision-language action-ensemble hard pseudo-label SFDA.

The four variants are deliberately kept protocol-identical after pseudo-label
construction:

* A_init_no_adaptation: frozen source model, no adaptation;
* B_source_hard: source probability thresholded at 0.5;
* C_vision_wp_hard_fusion: mean(source + six frozen vision WP predictions);
* E_vision_language_wp_hard_fusion: mean(source + six vision and three
  frozen language WP predictions).

Only frozen source/WP predictions create pseudo labels. GT is read after that
step for diagnostics and final evaluation only.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls_2 import v1_uncertainty_fusion as v1
from vls_2 import v2_vision_language_uncertainty_fusion as v2
from vls_2.v0_mri_action_screening import preprocess_image


OUTPUT_DIR = Path("output_2/v2_vision_language_action_ensemble")
VARIANTS = (
    "A_init_no_adaptation",
    "B_source_hard",
    "C_vision_wp_hard_fusion",
    "E_vision_language_wp_hard_fusion",
)
WP_ACTION_LABELS = (
    "gamma:-0.30",
    "gamma:+0.30",
    "gaussian_noise:+0.02",
    "gaussian_noise:+0.05",
    "bias_field:+0.05",
    "bias_field:+0.15",
)
LANGUAGE_PROMPTS = tuple(v2.LANGUAGE_PROMPTS)
PREDICTION_MEMBER_NAMES = tuple(v2.PREDICTION_MEMBER_NAMES)
PREDICTION_COUNTS = {"A_init_no_adaptation": 1, "B_source_hard": 1, "C_vision_wp_hard_fusion": 7,
                     "E_vision_language_wp_hard_fusion": 10}


def load_runtime_dependencies() -> None:
    """Use V1's delayed imports so --help remains usable without nnUNet."""
    v1.load_runtime_dependencies()
    from vls.data import read_image

    globals()["read_image"] = read_image


def make_paths(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(
        description="V1 action-ensemble hard pseudo-label upper bound; no blur/prompt action."
    )
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="output_2/v1_multi_action_world_predictor/best_multi_action_world_predictor.pt",
    )
    parser.add_argument(
        "--language-world-checkpoint",
        default="outputs/v10_language_wp_shared_output/language_wp_final.pt",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--evaluation-cases", type=int, default=0)
    parser.add_argument("--patch-limit-per-case", type=int, default=0)
    parser.add_argument("--training-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bias-sigma-fraction", type=float, default=0.12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--wp-actions",
        default=",".join(WP_ACTION_LABELS),
        help="Must remain the six V1 WP actions for a comparable formal run.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Exactly 1 train case, 1 test case, 1 patch, 1 epoch; never formal full run.",
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Validate fusion weights/action protocol without loading VoxTell or any data.",
    )
    return parser.parse_args()


def parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(labels) != 6 or len(set(labels)) != 6:
        raise ValueError("Exactly six distinct actions are required")
    return labels


def parse_wp_actions(value: str) -> tuple[Any, ...]:
    labels = parse_labels(value)
    if set(labels) != set(WP_ACTION_LABELS):
        raise ValueError(f"WP actions must be exactly {sorted(WP_ACTION_LABELS)}")
    return tuple(
        v1.WpAction(family, float(strength))
        for family, strength in (label.split(":", 1) for label in labels)
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.training_epochs <= 0:
        raise AssertionError("--training-epochs must be positive")
    if args.patch_limit_per_case < 0:
        raise AssertionError("--patch-limit-per-case must be non-negative")
    if args.train_cases < 0 or args.evaluation_cases < 0:
        raise AssertionError("case limits must be non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise AssertionError("invalid optimizer settings")
    if not 0.0 < args.prediction_threshold < 1.0:
        raise AssertionError("--prediction-threshold must be in (0,1)")
    if not 0.0 < args.bias_sigma_fraction < 0.5:
        raise AssertionError("--bias-sigma-fraction must be in (0,0.5)")
    if not args.smoke_only and args.patch_limit_per_case != 0:
        raise AssertionError(
            "Formal runs must use every native sliding-window patch; "
            "--patch-limit-per-case is smoke/debug only"
        )


def equal_weight_fusion(
    source_probability: torch.Tensor,
    action_probability_sum: torch.Tensor,
    action_count: int,
) -> torch.Tensor:
    """Average source and every action prediction with identical 1/(N+1) weight."""
    if action_count <= 0:
        raise AssertionError("At least one action prediction is required")
    if source_probability.shape != action_probability_sum.shape:
        raise AssertionError(
            "Source/action-sum prediction shapes differ: "
            f"{tuple(source_probability.shape)} vs {tuple(action_probability_sum.shape)}"
        )
    return (source_probability.float() + action_probability_sum.float()) / float(
        action_count + 1
    )


def assert_equal_weight_fusion_contract(action_count: int = 6) -> None:
    """Catch the previous source-vs-action-mean 50/50 weighting bug."""
    source_only = equal_weight_fusion(
        torch.ones(1), torch.zeros(1), action_count,
    )
    one_action_only = equal_weight_fusion(
        torch.zeros(1), torch.ones(1), action_count,
    )
    all_actions_only = equal_weight_fusion(
        torch.zeros(1), torch.full((1,), float(action_count)), action_count,
    )
    expected_single = torch.full((1,), 1.0 / float(action_count + 1))
    expected_all = torch.full((1,), action_count / float(action_count + 1))
    if not torch.allclose(source_only, expected_single, atol=0.0, rtol=1e-7):
        raise AssertionError("Source prediction does not have weight 1/(N+1)")
    if not torch.allclose(one_action_only, expected_single, atol=0.0, rtol=1e-7):
        raise AssertionError("Each action prediction does not have weight 1/(N+1)")
    if not torch.allclose(all_actions_only, expected_all, atol=0.0, rtol=1e-7):
        raise AssertionError("Action ensemble fusion denominator is incorrect")


def assert_prediction_contract(
    predictions: list[torch.Tensor], member_names: tuple[str, ...],
) -> None:
    if len(predictions) != len(member_names):
        raise AssertionError(
            f"Expected {len(member_names)} predictions {member_names}, got {len(predictions)}"
        )
    reference_shape = tuple(predictions[0].shape)
    if any(tuple(prediction.shape) != reference_shape for prediction in predictions):
        raise AssertionError("All ensemble prediction shapes must match")
    if any(prediction.dtype != torch.float32 for prediction in predictions):
        raise AssertionError("All ensemble predictions must be float32")
    if any(not torch.isfinite(prediction).all() for prediction in predictions):
        raise AssertionError("Ensemble predictions must be finite")


def serialize_slicer(slicer: tuple[slice, ...]) -> list[list[int | None]]:
    return [[item.start, item.stop, item.step] for item in slicer]


def deserialize_slicer(payload: list[list[int | None]]) -> tuple[slice, ...]:
    return tuple(slice(start, stop, step) for start, stop, step in payload)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def state_digest(network: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_binary(tensor: torch.Tensor, name: str) -> None:
    values = torch.unique(tensor.detach().cpu())
    if not torch.all((values == 0) | (values == 1)):
        raise AssertionError(f"{name} is not strict 0/1: {values.tolist()}")


def binary_stats(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, int | float]:
    pred = prediction.detach().cpu().bool().flatten().numpy()
    truth = target.detach().cpu().bool().flatten().numpy()
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    dice = 1.0 if 2 * tp + fp + fn == 0 else 2.0 * tp / (2 * tp + fp + fn)
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "dice": dice, "iou": iou}


def action_metric_accumulator() -> dict[str, float]:
    return {
        "effective_voxel_weight": 0.0,
        "patch_voxel_observations": 0.0,
        "mse_sum": 0.0,
        "mae_sum": 0.0,
        "consistent": 0.0,
    }


def add_action_metric(
    accumulator: dict[str, float],
    source: torch.Tensor,
    action: torch.Tensor,
    overlap_weight: torch.Tensor,
) -> None:
    source = source.detach().float().cpu()
    action = action.detach().float().cpu()
    weight = overlap_weight.detach().float().cpu().expand_as(source)
    difference = source - action
    accumulator["effective_voxel_weight"] += float(weight.sum())
    accumulator["patch_voxel_observations"] += float(source.numel())
    accumulator["mse_sum"] += float((difference.square() * weight).sum())
    accumulator["mae_sum"] += float((difference.abs() * weight).sum())
    accumulator["consistent"] += float(
        (((source > 0.5) == (action > 0.5)).float() * weight).sum()
    )


def finalize_action_metric(
    case: str, action_type: str, action_label: str, accumulator: dict[str, float],
) -> dict[str, Any]:
    count = max(accumulator["effective_voxel_weight"], 1.0)
    return {
        "case": case,
        "action_type": action_type,
        "action": action_label,
        "prediction_mse": accumulator["mse_sum"] / count,
        "prediction_mae": accumulator["mae_sum"] / count,
        "mask_consistency": accumulator["consistent"] / count,
        "effective_voxel_weight": accumulator["effective_voxel_weight"],
        "patch_voxel_observations": int(accumulator["patch_voxel_observations"]),
        "metric_scope": "inverse-overlap-normalized patch predictions",
        "gt_used": False,
    }


def source_probability(result: dict[str, Any]) -> torch.Tensor:
    return torch.sigmoid(result["final_prediction"][:, :1].detach().float())


@torch.inference_mode()
def wp_prediction(
    interface: Any,
    world_predictor: torch.nn.Module,
    source_context: dict[str, Any],
    action: Any,
    text_delta: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    features = v1.level_features(source_context["decoder_audit"]["skips"], v1.LEVEL_NAMES)
    predicted = world_predictor(features, v1.action_vector(action, device), text_delta)
    logits = v1.native_decoder_from_predicted_skips(
        interface, source_context, predicted, v1.LEVEL_NAMES,
    )
    return torch.sigmoid(logits.detach().float())


def action_geometry(
    interface: Any,
    source_image: np.ndarray,
    action_image: np.ndarray,
    source_padded: torch.Tensor,
    source_slicers: list[tuple],
    source_preprocessed_shape: tuple[int, ...],
) -> tuple[torch.Tensor, list[tuple], dict[str, Any]]:
    action_padded, action_slicers, action_preprocessed_shape = preprocess_image(
        interface.predictor, action_image,
    )
    checks = assert_action_geometry_unchanged(
        source_image, action_image, source_preprocessed_shape,
        action_preprocessed_shape, source_padded, action_padded,
        source_slicers, action_slicers,
    )
    return action_padded, action_slicers, checks


def gt_patch(
    label_padded: torch.Tensor, slicer: tuple, probability: torch.Tensor,
    label_value: int, device: torch.device,
) -> torch.Tensor:
    target = (label_padded[slicer][None].to(device) == label_value).float()
    if tuple(target.shape[-3:]) != tuple(probability.shape[-3:]):
        import torch.nn.functional as F

        target = F.interpolate(target, size=probability.shape[-3:], mode="nearest")
    return target


def _accumulate_stitched_patch(
    output: torch.Tensor,
    patch: torch.Tensor,
    slicer: tuple[slice, ...],
    overlap_weight: torch.Tensor,
) -> None:
    spatial_slicer = slicer[1:]
    patch_cpu = patch.detach().float().cpu().squeeze(0).squeeze(0)
    weight_cpu = overlap_weight.detach().float().cpu().squeeze(0).squeeze(0)
    if tuple(patch_cpu.shape) != tuple(weight_cpu.shape):
        raise AssertionError(
            f"Patch/weight shapes differ: {tuple(patch_cpu.shape)} vs {tuple(weight_cpu.shape)}"
        )
    output[spatial_slicer].add_(patch_cpu * weight_cpu)


def pseudo_case_rows(
    case: Any,
    records: list[dict[str, Any]],
    coverage_weights: list[torch.Tensor],
    source_padded_shape: tuple[int, ...],
    label: np.ndarray,
    interface: Any,
    label_value: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Report one Gaussian-stitched diagnostic per physical preprocessed voxel."""
    label_padded = v1.pad_label_like_image(interface, label)
    spatial_shape = tuple(int(value) for value in source_padded_shape[1:])
    coverage = torch.zeros(spatial_shape, dtype=torch.float32)
    source_volume = torch.zeros_like(coverage)
    target_volume = torch.zeros_like(coverage)
    pseudo_volumes = {
        variant: torch.zeros_like(coverage) for variant in VARIANTS[1:]
    }
    for record, overlap_weight in zip(records, coverage_weights, strict=True):
        target = gt_patch(
            label_padded, record["slicer"], record["source_hard"], label_value, device,
        )
        _accumulate_stitched_patch(
            coverage, torch.ones_like(record["source_hard"]),
            record["slicer"], overlap_weight,
        )
        _accumulate_stitched_patch(
            source_volume, record["source_hard"], record["slicer"], overlap_weight,
        )
        _accumulate_stitched_patch(
            target_volume, target, record["slicer"], overlap_weight,
        )
        for variant in VARIANTS[1:]:
            pseudo = record["pseudo"][variant]
            assert_binary(pseudo, f"{case.case}/{variant}")
            _accumulate_stitched_patch(
                pseudo_volumes[variant], pseudo, record["slicer"], overlap_weight,
            )
        del target

    valid = coverage > 0
    if not torch.any(valid):
        raise AssertionError(f"No stitched pseudo-label coverage for {case.case}")
    source_mask = source_volume > 0.5
    target_mask = target_volume > 0.5
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS[1:]:
        pseudo_mask = pseudo_volumes[variant] > 0.5
        stats = binary_stats(pseudo_mask[valid], target_mask[valid])
        tp, fp, fn, tn = (
            int(stats["tp"]), int(stats["fp"]), int(stats["fn"]), int(stats["tn"]),
        )
        source_valid = source_mask[valid]
        pseudo_valid = pseudo_mask[valid]
        rows.append({
            "variant": variant,
            "case": case.case,
            "scope": "preprocessed_full_volume_gaussian_stitched",
            "pseudo_label_dice": float(stats["dice"]),
            "pseudo_label_iou": float(stats["iou"]),
            "pseudo_label_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "pseudo_positive_voxels": int(pseudo_valid.sum()),
            "background_to_foreground": int(((~source_valid) & pseudo_valid).sum()),
            "foreground_to_background": int((source_valid & (~pseudo_valid)).sum()),
            "voxel_count": int(valid.sum()),
            "gt_used": True,
            "gt_usage": "diagnostic only after pseudo construction and cache write",
        })
    return rows


def case_cache_path(cache_dir: Path, case_name: str) -> Path:
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in case_name
    )
    digest = hashlib.sha256(case_name.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{safe_name}.{digest}.pt"


def load_case_cache(cache: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(Path(cache["cache_path"]), map_location="cpu", weights_only=False)
    if payload.get("stage") != "V2 vision-language action ensemble uint8 pseudo cache":
        raise AssertionError(f"Unexpected pseudo cache stage: {payload.get('stage')}")
    if payload.get("case") != cache["case"]:
        raise AssertionError(
            f"Pseudo cache case mismatch: {payload.get('case')} vs {cache['case']}"
        )
    return payload


@torch.inference_mode()
def _legacy_build_case_target_cache(
    case: Any,
    interface: Any,
    base_network: torch.nn.Module,
    world_predictor: torch.nn.Module,
    embedding: torch.Tensor,
    text_delta: torch.Tensor,
    wp_actions: tuple[Any, ...],
    real_actions: tuple[RealAction, ...],
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct all targets without receiving GT; stream real actions."""
    image, _ = read_image(case)
    source_padded, source_slicers, source_preprocessed_shape = preprocess_image(
        interface.predictor, image,
    )
    full_source_slicers = list(source_slicers)
    if args.patch_limit_per_case:
        source_slicers = source_slicers[:args.patch_limit_per_case]
    if not source_slicers:
        raise AssertionError(f"No native sliding-window patches for {case.case}")
    coverage_weights = v1.overlap_weights(interface, source_padded, source_slicers)
    records: list[dict[str, Any]] = []
    wp_accumulators = {action.label: action_metric_accumulator() for action in wp_actions}

    for patch_index, slicer in enumerate(source_slicers):
        patch = torch.clone(source_padded[slicer][None], memory_format=torch.contiguous_format).to(device)
        context = v1.source_context_from_network(interface, base_network, patch, embedding, device)
        source_p = source_probability(context)
        wp_sum = torch.zeros_like(source_p)
        for action in wp_actions:
            prediction = wp_prediction(interface, world_predictor, context, action, text_delta, device)
            wp_sum.add_(prediction)
            add_action_metric(
                wp_accumulators[action.label], source_p, prediction,
                coverage_weights[patch_index],
            )
            del prediction
        records.append({
            "case": case.case,
            "case_index": int(patch_index),
            "patch_index": patch_index,
            "slicer": slicer,
            "source_probability": source_p.detach().cpu(),
            "source_hard": (source_p > args.prediction_threshold).float().detach().cpu(),
            "wp_sum": wp_sum.detach().cpu(),
            "real_sum": torch.zeros_like(source_p.detach().cpu()),
        })
        del patch, context, source_p, wp_sum

    geometry_rows: list[dict[str, Any]] = []
    real_accumulators = {action.label: action_metric_accumulator() for action in real_actions}
    # The complete raw case is actioned, then discarded before the next action.
    for action in real_actions:
        action_image = apply_action(
            image, action.v0_variant, case_name=case.case, seed=args.seed,
            bias_sigma_fraction=args.bias_sigma_fraction,
        )
        action_padded, action_slicers, geometry = action_geometry(
            interface, image, action_image, source_padded, full_source_slicers,
            source_preprocessed_shape,
        )
        geometry_rows.append({
            "case": case.case,
            "action": action.label,
            **geometry,
            "raw_action_generation": "V0 stable case/action/seed",
        })
        for record in records:
            patch = torch.clone(
                action_padded[record["slicer"]][None], memory_format=torch.contiguous_format,
            ).to(device)
            autocast = (
                torch.autocast(device.type, enabled=True)
                if device.type == "cuda" else nullcontext()
            )
            with autocast:
                result = interface._network_forward_with_states(
                    base_network, patch, embedding.to(device).float(),
                )
            prediction = torch.sigmoid(result["final_prediction"][:, :1].detach().float())
            record["real_sum"].add_(prediction.cpu())
            add_action_metric(
                real_accumulators[action.label], record["source_probability"], prediction,
                coverage_weights[record["patch_index"]],
            )
            del patch, result, prediction
        del action_image, action_padded, action_slicers
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    action_rows = [
        finalize_action_metric(case.case, "imagined_wp", action.label, wp_accumulators[action.label])
        for action in wp_actions
    ] + [
        finalize_action_metric(case.case, "real_mri", action.label, real_accumulators[action.label])
        for action in real_actions
    ]
    for record in records:
        source_probability_cpu = record["source_probability"]
        source_hard = record["source_hard"]
        targets = {
            "B_source_hard": source_hard,
            "C_wp_hard_fusion": (
                equal_weight_fusion(
                    source_probability_cpu, record["wp_sum"], len(wp_actions),
                )
                > args.prediction_threshold
            ).float(),
            "D_real_action_hard_fusion": (
                equal_weight_fusion(
                    source_probability_cpu, record["real_sum"], len(real_actions),
                )
                > args.prediction_threshold
            ).float(),
        }
        for variant, pseudo in targets.items():
            assert_binary(pseudo, f"{case.case}/{variant}")
        record["pseudo"] = targets

    source_padded_shape = tuple(int(value) for value in source_padded.shape)
    prediction_shape = list(records[0]["prediction_shape"])
    prediction_dtype = records[0]["prediction_dtype"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = case_cache_path(cache_dir, case.case)
    cache_payload = {
        "stage": "V1 action ensemble uint8 pseudo cache",
        "case": case.case,
        "prediction_threshold": args.prediction_threshold,
        "target_dtype": "uint8",
        "source_preprocessed_shape": [int(value) for value in source_preprocessed_shape],
        "source_padded_shape": list(source_padded_shape),
        "full_slicer_count": len(full_source_slicers),
        "used_slicer_count": len(source_slicers),
        "records": [
            {
                "patch_index": int(record["patch_index"]),
                "slicer": serialize_slicer(record["slicer"]),
                "targets": {
                    variant: record["pseudo"][variant].to(torch.uint8).contiguous()
                    for variant in VARIANTS[1:]
                },
            }
            for record in records
        ],
    }
    # The cache is written before GT is read, making the no-GT construction
    # boundary explicit and auditable.
    torch.save(cache_payload, cache_path)
    _, label, _ = v1.read_image_and_label(case)
    pseudo_rows = pseudo_case_rows(
        case, records, coverage_weights, source_padded_shape, label,
        interface, args.label_value, device,
    )
    manifest_slicers = [serialize_slicer(slicer) for slicer in source_slicers]
    del image, source_padded, label, coverage_weights, cache_payload, records
    gc.collect()
    return {
        "case": case.case,
        "case_index": int(getattr(case, "case_index", 0)),
        "cache_path": str(cache_path),
        "target_dtype": "uint8",
        "full_slicer_count": len(full_source_slicers),
        "used_slicer_count": len(source_slicers),
        "source_preprocessed_shape": [int(value) for value in source_preprocessed_shape],
        "source_padded_shape": list(source_padded_shape),
        "slicers": manifest_slicers,
    }, action_rows, geometry_rows, pseudo_rows


@torch.inference_mode()
def build_case_target_cache(
    case: Any,
    interface: Any,
    base_network: torch.nn.Module,
    world_predictor: torch.nn.Module,
    language_world_predictor: torch.nn.Module,
    embedding: torch.Tensor,
    text_delta: torch.Tensor,
    language_text_deltas: torch.Tensor,
    wp_actions: tuple[Any, ...],
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build hard pseudo labels from frozen predictions, before reading GT."""
    image, _ = read_image(case)
    source_padded, source_slicers, source_preprocessed_shape = preprocess_image(
        interface.predictor, image,
    )
    full_source_slicers = list(source_slicers)
    if args.patch_limit_per_case:
        source_slicers = source_slicers[:args.patch_limit_per_case]
    if not source_slicers:
        raise AssertionError(f"No native sliding-window patches for {case.case}")
    coverage_weights = v1.overlap_weights(interface, source_padded, source_slicers)
    records: list[dict[str, Any]] = []
    vision_accumulators = {action.label: action_metric_accumulator() for action in wp_actions}
    language_accumulators = {prompt: action_metric_accumulator() for prompt in LANGUAGE_PROMPTS}

    for patch_index, slicer in enumerate(source_slicers):
        patch = torch.clone(
            source_padded[slicer][None], memory_format=torch.contiguous_format,
        ).to(device)
        context = v1.source_context_from_network(interface, base_network, patch, embedding, device)
        source_p, vision_outputs, language_outputs = v2.imagined_predictions(
            interface, base_network, world_predictor, language_world_predictor,
            context, list(wp_actions), text_delta, language_text_deltas, device,
        )
        predictions = [
            source_p,
            *(prediction for _, prediction in vision_outputs),
            *(prediction for _, prediction in language_outputs),
        ]
        assert_prediction_contract(predictions, PREDICTION_MEMBER_NAMES)
        if len(vision_outputs) != 6 or len(language_outputs) != 3:
            raise AssertionError("Expected six vision and three language predictions")
        vision_sum = torch.zeros_like(source_p, dtype=torch.float32)
        for action, prediction in vision_outputs:
            vision_sum.add_(prediction)
            add_action_metric(
                vision_accumulators[action.label], source_p, prediction,
                coverage_weights[patch_index],
            )
        language_sum = torch.zeros_like(source_p, dtype=torch.float32)
        for prompt, prediction in language_outputs:
            language_sum.add_(prediction)
            add_action_metric(
                language_accumulators[prompt], source_p, prediction,
                coverage_weights[patch_index],
            )
        records.append({
            "case": case.case,
            "case_index": int(patch_index),
            "patch_index": patch_index,
            "slicer": slicer,
            "source_probability": source_p.detach().cpu(),
            "source_hard": (source_p > args.prediction_threshold).float().detach().cpu(),
            "vision_sum": vision_sum.detach().cpu(),
            "language_sum": language_sum.detach().cpu(),
            "prediction_shape": list(source_p.shape),
            "prediction_dtype": str(source_p.dtype),
        })
        del patch, context, source_p, vision_outputs, language_outputs
        del predictions, vision_sum, language_sum

    action_rows = [
        finalize_action_metric(case.case, "vision_wp", action.label, vision_accumulators[action.label])
        for action in wp_actions
    ] + [
        finalize_action_metric(case.case, "language_wp", prompt, language_accumulators[prompt])
        for prompt in LANGUAGE_PROMPTS
    ]
    for record in records:
        source_probability_cpu = record["source_probability"]
        targets = {
            "B_source_hard": record["source_hard"],
            "C_vision_wp_hard_fusion": (
                equal_weight_fusion(source_probability_cpu, record["vision_sum"], 6)
                > args.prediction_threshold
            ).float(),
            "E_vision_language_wp_hard_fusion": (
                equal_weight_fusion(
                    source_probability_cpu,
                    record["vision_sum"] + record["language_sum"],
                    9,
                ) > args.prediction_threshold
            ).float(),
        }
        for variant, pseudo in targets.items():
            assert_binary(pseudo, f"{case.case}/{variant}")
        record["pseudo"] = targets

    source_padded_shape = tuple(int(value) for value in source_padded.shape)
    prediction_shape = list(records[0]["prediction_shape"])
    prediction_dtype = records[0]["prediction_dtype"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = case_cache_path(cache_dir, case.case)
    cache_payload = {
        "stage": "V2 vision-language action ensemble uint8 pseudo cache",
        "case": case.case,
        "prediction_threshold": args.prediction_threshold,
        "prediction_member_names": list(PREDICTION_MEMBER_NAMES),
        "prediction_counts": dict(PREDICTION_COUNTS),
        "prediction_shape": prediction_shape,
        "prediction_dtype": prediction_dtype,
        "target_dtype": "uint8",
        "vision_wp_frozen": True,
        "language_wp_frozen": True,
        "gt_used_for_pseudo_construction": False,
        "source_preprocessed_shape": [int(value) for value in source_preprocessed_shape],
        "source_padded_shape": list(source_padded_shape),
        "full_slicer_count": len(full_source_slicers),
        "used_slicer_count": len(source_slicers),
        "records": [
            {
                "patch_index": int(record["patch_index"]),
                "slicer": serialize_slicer(record["slicer"]),
                "targets": {
                    variant: record["pseudo"][variant].to(torch.uint8).contiguous()
                    for variant in VARIANTS[1:]
                },
            }
            for record in records
        ],
    }
    # Write the target cache before loading GT. GT is diagnostic only.
    torch.save(cache_payload, cache_path)
    _, label, _ = v1.read_image_and_label(case)
    pseudo_rows = pseudo_case_rows(
        case, records, coverage_weights, source_padded_shape, label,
        interface, args.label_value, device,
    )
    manifest_slicers = [serialize_slicer(slicer) for slicer in source_slicers]
    del image, source_padded, label, coverage_weights, cache_payload, records
    gc.collect()
    return {
        "case": case.case,
        "case_index": int(getattr(case, "case_index", 0)),
        "cache_path": str(cache_path),
        "target_dtype": "uint8",
        "prediction_member_names": list(PREDICTION_MEMBER_NAMES),
        "prediction_counts": dict(PREDICTION_COUNTS),
        "prediction_shape": prediction_shape,
        "prediction_dtype": prediction_dtype,
        "full_slicer_count": len(full_source_slicers),
        "used_slicer_count": len(source_slicers),
        "source_preprocessed_shape": [int(value) for value in source_preprocessed_shape],
        "source_padded_shape": list(source_padded_shape),
        "slicers": manifest_slicers,
    }, action_rows, [], pseudo_rows


def gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters if parameter.grad is not None
    ]
    return 0.0 if not terms else float(torch.sqrt(sum(terms)).detach().cpu())


def train_variant(
    variant: str,
    base_network_cpu: torch.nn.Module,
    case_caches: list[dict[str, Any]],
    train_cases: list[Any],
    eval_cases: list[Any],
    full_data: dict[str, tuple[np.ndarray, np.ndarray]],
    interface: Any,
    embedding: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    target_names: list[str] | None,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(args.seed)
    student = copy.deepcopy(base_network_cpu)
    targets = v1.inject_lora_qkv(student, args.lora_rank, args.lora_alpha, args.lora_dropout)
    initial_digest = state_digest(student)
    if target_names is not None and targets != target_names:
        raise AssertionError("LoRA target modules differ across variants")
    student = student.to(device).eval()
    trainable = v1.lora_parameters(student)
    base_trainable = sum(
        parameter.numel() for name, parameter in student.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    if base_trainable:
        raise AssertionError(f"Base VoxTell parameters are trainable: {base_trainable}")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loss_rows: list[dict[str, Any]] = []
    cases_by_name = {case.case: case for case in train_cases}
    if set(cases_by_name) != {cache["case"] for cache in case_caches}:
        raise AssertionError("Train cases and pseudo-cache cases differ")
    rng = np.random.default_rng(args.seed)
    training_sequence: list[str] = []
    global_step = 0
    for epoch in range(1, args.training_epochs + 1):
        epoch_indices = rng.permutation(len(case_caches)).tolist()
        for case_position, case_index in enumerate(epoch_indices, start=1):
            cache = case_caches[case_index]
            case = cases_by_name[cache["case"]]
            payload = load_case_cache(cache)
            image, _ = read_image(case)
            source_padded, source_slicers, source_preprocessed_shape = preprocess_image(
                interface.predictor, image,
            )
            if args.patch_limit_per_case:
                source_slicers = source_slicers[:args.patch_limit_per_case]
            serialized_slicers = [serialize_slicer(slicer) for slicer in source_slicers]
            cached_slicers = [record["slicer"] for record in payload["records"]]
            if serialized_slicers != cached_slicers or serialized_slicers != cache["slicers"]:
                raise AssertionError(f"Sliding-window topology changed for {case.case}")
            if [int(value) for value in source_preprocessed_shape] != cache["source_preprocessed_shape"]:
                raise AssertionError(f"Preprocessed shape changed for {case.case}")
            if list(source_padded.shape) != cache["source_padded_shape"]:
                raise AssertionError(f"Padded shape changed for {case.case}")
            coverage_weights = v1.overlap_weights(interface, source_padded, source_slicers)
            optimizer.zero_grad(set_to_none=True)
            patch_losses: list[float] = []
            for record, slicer, overlap_weight in zip(
                payload["records"], source_slicers, coverage_weights, strict=True,
            ):
                if deserialize_slicer(record["slicer"]) != slicer:
                    raise AssertionError(f"Cached slicer mismatch for {case.case}")
                patch = torch.clone(
                    source_padded[slicer][None], memory_format=torch.contiguous_format,
                ).to(device)
                embedding_batch = embedding.to(device).float()
                pseudo_uint8 = record["targets"][variant]
                if pseudo_uint8.dtype != torch.uint8:
                    raise AssertionError(
                        f"Cached pseudo target is {pseudo_uint8.dtype}, expected uint8"
                    )
                pseudo = pseudo_uint8.to(device=device, dtype=torch.float32)
                weight = overlap_weight.to(
                    device=device, dtype=torch.float32,
                ).expand_as(pseudo)
                assert_binary(pseudo, f"training/{variant}/{cache['case']}")
                autocast = (
                    torch.autocast(device.type, enabled=True)
                    if device.type == "cuda" else nullcontext()
                )
                with autocast:
                    result = interface._network_forward_with_states(
                        student, patch, embedding_batch,
                    )
                    logits = result["final_prediction"][:, :1].float()
                    loss, _ = v1.class_balanced_loss(logits, pseudo, weight)
                    scaled_loss = loss / float(len(source_slicers))
                scaled_loss.backward()
                patch_losses.append(float(loss.detach().cpu()))
                del patch, embedding_batch, pseudo, weight, result, logits, loss, scaled_loss
            grad = gradient_norm(trainable)
            before = [parameter.detach().clone() for parameter in trainable]
            optimizer.step()
            update_delta = float(torch.sqrt(sum(
                (parameter.detach() - old).float().square().sum()
                for parameter, old in zip(trainable, before, strict=True)
            )).detach().cpu())
            global_step += 1
            training_sequence.append(cache["case"])
            loss_rows.append({
                "variant": variant,
                "epoch": epoch,
                "case_position": case_position,
                "training_order_index": (epoch - 1) * len(case_caches) + case_position,
                "case": cache["case"],
                "patch_count": len(source_slicers),
                "mean_patch_loss": float(np.mean(patch_losses, dtype=np.float64)),
                "gradient_norm": grad,
                "update_delta_norm": update_delta,
                "optimizer_updates": 1,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "initial_lora_state_digest": initial_digest,
            })
            print(
                f"[v2_vision_language_action_ensemble] {variant} epoch={epoch}/{args.training_epochs} "
                f"case={case_position}/{len(case_caches)} {case.case} "
                f"patches={len(source_slicers)} loss={loss_rows[-1]['mean_patch_loss']:.6f}",
                flush=True,
            )
            del before, payload, image, source_padded, source_slicers, coverage_weights
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    full_rows = v1.evaluate_full_volume(
        interface, student, eval_cases, full_data, embedding, variant,
        "full_window_case_balanced", global_step, args.label_value,
        args.prediction_threshold,
    )
    stats = {
        "variant": variant,
        "initial_lora_state_digest": initial_digest,
        "target_modules": targets,
        "base_trainable_parameters": base_trainable,
        "lora_trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "optimizer_updates": global_step,
        "training_case_order": training_sequence,
    }
    lora_path = output_dir / f"{variant}_lora.pt"
    torch.save({
        "stage": "V2 vision-language action ensemble LoRA checkpoint",
        "variant": variant,
        "state_dict": {name: value.detach().cpu() for name, value in student.state_dict().items()},
        "target_modules": targets,
        "initial_lora_state_digest": initial_digest,
        "training_epochs": args.training_epochs,
    }, lora_path)
    stats["lora_checkpoint"] = str(lora_path)
    del student, optimizer, trainable
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_rows, full_rows, stats


def add_metric_deltas(rows: list[dict[str, Any]]) -> None:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[row["case"]][row["variant"]] = row

    # Normalize every row before reading any baseline row.  The previous
    # single-pass implementation could encounter a C/D row before the B row
    # had been visited, so b["iou"] did not exist yet.
    for row in rows:
        if "iou" not in row:
            if "foreground_iou" not in row:
                raise KeyError(
                    f"Metric row for {row.get('case')}/{row.get('variant')} "
                    "has neither iou nor foreground_iou"
                )
            row["iou"] = row["foreground_iou"]

    for row in rows:
        a = by_case[row["case"]]["A_init_no_adaptation"]
        b = by_case[row["case"]]["B_source_hard"]
        c = by_case[row["case"]]["C_vision_wp_hard_fusion"]
        row["delta_dice_vs_A"] = row["dice"] - a["dice"]
        row["delta_iou_vs_A"] = row["iou"] - a["iou"]
        row["delta_precision_vs_A"] = row["precision"] - a["precision"]
        row["delta_recall_vs_A"] = row["recall"] - a["recall"]
        row["delta_dice_vs_B"] = row["dice"] - b["dice"]
        row["delta_iou_vs_B"] = row["iou"] - b["iou"]
        row["delta_precision_vs_B"] = row["precision"] - b["precision"]
        row["delta_recall_vs_B"] = row["recall"] - b["recall"]
        row["delta_dice_vs_C"] = row["dice"] - c["dice"]
        row["delta_iou_vs_C"] = row["iou"] - c["iou"]
        row["delta_precision_vs_C"] = row["precision"] - c["precision"]
        row["delta_recall_vs_C"] = row["recall"] - c["recall"]


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    output: list[dict[str, Any]] = []
    fields = (
        "dice", "iou", "precision", "recall", "delta_dice_vs_A", "delta_iou_vs_A",
        "delta_precision_vs_A", "delta_recall_vs_A", "delta_dice_vs_B", "delta_iou_vs_B",
        "delta_precision_vs_B", "delta_recall_vs_B",
        "delta_dice_vs_C", "delta_iou_vs_C", "delta_precision_vs_C", "delta_recall_vs_C",
    )
    for variant in VARIANTS:
        group = grouped.get(variant, [])
        row: dict[str, Any] = {"variant": variant, "case_count": len(group)}
        for field in fields:
            row[f"mean_{field}"] = float(np.mean([float(item[field]) for item in group])) if group else None
            row[f"std_{field}"] = float(np.std([float(item[field]) for item in group])) if group else None
        output.append(row)
    return output


def ensemble_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return case-macro E deltas and win rates against A, B, and C."""
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[row["case"]][row["variant"]] = row
    output: list[dict[str, Any]] = []
    metrics = ("dice", "iou", "precision", "recall")
    for baseline_variant in (
        "A_init_no_adaptation", "B_source_hard", "C_vision_wp_hard_fusion",
    ):
        e_rows = []
        for case, variants in by_case.items():
            if "E_vision_language_wp_hard_fusion" not in variants:
                continue
            e = variants["E_vision_language_wp_hard_fusion"]
            baseline = variants[baseline_variant]
            row: dict[str, Any] = {"comparison": "E_vs_" + baseline_variant, "case_count": 1, "case": case}
            for metric in metrics:
                row[f"delta_{metric}"] = float(e[metric] - baseline[metric])
                row[f"win_{metric}"] = bool(e[metric] > baseline[metric])
            e_rows.append(row)
        summary: dict[str, Any] = {
            "comparison": "E_vs_" + baseline_variant,
            "case_count": len(e_rows),
        }
        for metric in metrics:
            deltas = [float(row[f"delta_{metric}"]) for row in e_rows]
            wins = [bool(row[f"win_{metric}"]) for row in e_rows]
            summary[f"mean_delta_{metric}"] = float(np.mean(deltas)) if deltas else None
            summary[f"win_rate_{metric}"] = float(np.mean(wins)) if wins else None
        output.append(summary)
        output.extend(e_rows)
    return output


def run(args: argparse.Namespace) -> None:
    if args.smoke_only:
        args.train_cases = 1
        args.evaluation_cases = 1
        args.patch_limit_per_case = 1
        args.training_epochs = 1
    validate_args(args)
    wp_actions = parse_wp_actions(args.wp_actions)
    assert_equal_weight_fusion_contract(len(wp_actions))
    assert_equal_weight_fusion_contract(len(wp_actions) + len(LANGUAGE_PROMPTS))
    if len(wp_actions) != 6:
        raise AssertionError("The formal ensemble requires exactly six vision actions")
    if args.self_test_only:
        print(json.dumps({
            "status": "self_test_passed",
            "prediction_count": 10,
            "prediction_member_names": list(PREDICTION_MEMBER_NAMES),
            "weight_per_prediction": 0.1,
            "wp_actions": [action.label for action in wp_actions],
            "language_prompts": list(LANGUAGE_PROMPTS),
        }, indent=2))
        return
    load_runtime_dependencies()
    v2.load_runtime_dependencies()
    if v1.SOURCE_PROMPT != "liver":
        raise AssertionError(f"Unexpected source prompt: {v1.SOURCE_PROMPT!r}")
    if LANGUAGE_PROMPTS != ("the liver", "human liver", "hepatic organ"):
        raise AssertionError(f"Unexpected language prompts: {LANGUAGE_PROMPTS}")
    v1.set_seed(args.seed)
    device = v1.resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"Requested CUDA but resolved {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "pseudo_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = make_paths(args)
    train_cases = v1.iter_cases(paths, split="train", limit=args.train_cases)
    eval_cases = v1.iter_cases(paths, split="test", limit=args.evaluation_cases)
    train_names = [case.case for case in train_cases]
    eval_names = [case.case for case in eval_cases]
    overlap = sorted(set(train_names) & set(eval_names))
    if overlap:
        raise AssertionError(f"Train/evaluation overlap: {overlap}")
    if not train_cases or not eval_cases:
        raise AssertionError("Both train and test cases are required")
    if not args.smoke_only and len(eval_cases) != 8:
        raise AssertionError(
            f"Formal evaluation must use the complete 8-case held-out split, got {len(eval_cases)}"
        )

    interface = v1.VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    # Reuse the source network already owned by the interface.  Keeping a
    # deep-copied second VoxTell on GPU needlessly doubles model memory.
    base_network = interface.network.to(device).eval()
    for parameter in base_network.parameters():
        parameter.requires_grad_(False)
    embedding = interface.embed_text_prompts([v1.SOURCE_PROMPT]).detach().cpu()
    text_delta = v1.zero_text_delta(interface, device)
    language_embeddings = interface.embed_text_prompts(
        [v1.SOURCE_PROMPT, *LANGUAGE_PROMPTS],
    ).detach().cpu()
    source_flat = v2.flatten_prompt_embedding(language_embeddings, 0).float()
    language_text_deltas = torch.stack([
        v2.flatten_prompt_embedding(language_embeddings, index).float() - source_flat
        for index in range(1, 1 + len(LANGUAGE_PROMPTS))
    ])
    world_predictor, checkpoint_metadata = v1.load_multi_action_world_predictor(
        Path(args.world_checkpoint), list(wp_actions), device, allow_smoke=args.smoke_only,
    )
    language_world_predictor, language_checkpoint_metadata = v2.load_language_world_predictor(
        Path(args.language_world_checkpoint), device,
    )
    for name, predictor in (("vision", world_predictor), ("language", language_world_predictor)):
        if any(parameter.requires_grad for parameter in predictor.parameters()):
            raise AssertionError(f"{name} World Predictor must be frozen")
        predictor.eval()

    all_cache: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    for case in train_cases:
        cache, actions, geometry, pseudo = build_case_target_cache(
            case, interface, base_network, world_predictor, language_world_predictor,
            embedding, text_delta, language_text_deltas, wp_actions, cache_dir, args, device,
        )
        all_cache.append(cache)
        action_rows.extend(actions)
        geometry_rows.extend(geometry)
        pseudo_rows.extend(pseudo)
        print(
            f"[v2_vision_language_action_ensemble] cached {case.case} "
            f"patches={cache['used_slicer_count']} dtype={cache['target_dtype']}",
            flush=True,
        )

    full_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in eval_cases:
        image, label, _ = v1.read_image_and_label(case)
        full_data[case.case] = (image, label)
    base_network.eval()
    initial_rows = v1.evaluate_full_volume(
        interface, base_network, eval_cases, full_data, embedding,
        "A_init_no_adaptation", "full_window_case_balanced", 0,
        args.label_value, args.prediction_threshold,
    )
    torch.save({
        "stage": "V2 vision-language action ensemble LoRA checkpoint",
        "variant": "A_init_no_adaptation",
        "state_dict": {name: value.detach().cpu() for name, value in base_network.state_dict().items()},
        "adapted": False,
    }, output_dir / "A_init_no_adaptation_lora.pt")
    base_network_cpu = base_network.to("cpu").eval()
    world_predictor.to("cpu")
    language_world_predictor.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_full_rows = list(initial_rows)
    loss_rows: list[dict[str, Any]] = []
    parameter_stats: list[dict[str, Any]] = []
    target_names: list[str] | None = None
    for variant in VARIANTS[1:]:
        losses, full_rows, stats = train_variant(
            variant, base_network_cpu, all_cache, train_cases, eval_cases, full_data,
            interface, embedding, args, device, target_names, output_dir,
        )
        target_names = stats["target_modules"] if target_names is None else target_names
        loss_rows.extend(losses)
        all_full_rows.extend(full_rows)
        parameter_stats.append(stats)

    add_metric_deltas(all_full_rows)
    metrics_rows = aggregate_metrics(all_full_rows)
    comparison_rows = ensemble_comparison(all_full_rows)
    write_rows(output_dir / "training_loss.csv", loss_rows)
    write_rows(output_dir / "action_metrics.csv", action_rows)
    write_rows(output_dir / "action_geometry.csv", geometry_rows)
    write_rows(output_dir / "pseudo_label_metrics.csv", pseudo_rows)
    write_rows(output_dir / "per_case.csv", all_full_rows)
    write_rows(output_dir / "metrics.csv", metrics_rows)
    write_rows(output_dir / "E_comparisons.csv", comparison_rows)
    write_json(output_dir / "parameter_stats.json", {"variants": parameter_stats})
    write_json(output_dir / "train_patch_manifest.json", {
        "case_order": train_names,
        "patch_protocol": "native sliding windows; smoke truncates to first patch",
        "records": [
            {
                "case": cache["case"],
                "full_slicer_count": cache["full_slicer_count"],
                "used_slicer_count": cache["used_slicer_count"],
                "source_preprocessed_shape": cache["source_preprocessed_shape"],
                "source_padded_shape": cache["source_padded_shape"],
                "cache_path": cache["cache_path"],
                "target_dtype": cache["target_dtype"],
                "prediction_count": cache["prediction_counts"],
                "prediction_member_names": cache["prediction_member_names"],
                "prediction_shape": cache["prediction_shape"],
                "prediction_dtype": cache["prediction_dtype"],
                "slicers": cache["slicers"],
            }
            for cache in all_cache
        ],
    })

    lora_digests = [item["initial_lora_state_digest"] for item in parameter_stats]
    if len(set(lora_digests)) != 1:
        raise AssertionError(f"LoRA initialization differs across B/C/E: {lora_digests}")
    training_orders = [item["training_case_order"] for item in parameter_stats]
    expected_rng = np.random.default_rng(args.seed)
    expected_training_order: list[str] = []
    for _ in range(args.training_epochs):
        expected_training_order.extend(
            train_names[index]
            for index in expected_rng.permutation(len(train_names)).tolist()
        )
    if any(order != expected_training_order for order in training_orders):
        raise AssertionError("Training case order differs across B/C/E")
    smoke_binary = {variant: True for variant in VARIANTS[1:]}
    if args.smoke_only:
        for cache in all_cache:
            payload = load_case_cache(cache)
            for record in payload["records"]:
                for variant in VARIANTS[1:]:
                    target = record["targets"][variant]
                    assert_binary(target, f"smoke/{variant}")
                    smoke_binary[variant] = smoke_binary[variant] and target.dtype == torch.uint8
            del payload

    summary = {
        "stage": "V2 vision-language action ensemble hard pseudo-label SFDA",
        "status": "smoke_complete" if args.smoke_only else "complete",
        "smoke_only": bool(args.smoke_only),
        "output_dir": str(output_dir),
        "variants": list(VARIANTS),
        "train_cases": train_names,
        "evaluation_cases": eval_names,
        "case_overlap": overlap,
        "wp_actions": [action.label for action in wp_actions],
        "language_prompts": list(LANGUAGE_PROMPTS),
        "prediction_count": 10,
        "prediction_member_names": list(PREDICTION_MEMBER_NAMES),
        "prediction_counts_by_variant": dict(PREDICTION_COUNTS),
        "prediction_shape": all_cache[0]["prediction_shape"] if all_cache else None,
        "prediction_dtype": all_cache[0]["prediction_dtype"] if all_cache else None,
        "blur_used": False,
        "prompt_action_used": False,
        "world_predictor_updated_during_adaptation": False,
        "vision_wp_frozen": True,
        "language_wp_frozen": True,
        "source_voxtell_base_frozen": True,
        "gt_used_for_pseudo_construction": False,
        "gt_usage": "pseudo-label diagnostics and final evaluation only",
        "world_checkpoint": checkpoint_metadata,
        "language_world_checkpoint": language_checkpoint_metadata,
        "formulas": {
            "B_source_hard": "Y_B=1[P_source>0.5]",
            "C_vision_wp_hard_fusion": "Y_C=1[(P_source+sum(P_vision_i))/7>0.5]",
            "E_vision_language_wp_hard_fusion": "Y_E=1[(P_source+sum(P_vision_i)+sum(P_language_i))/10>0.5]",
            "all_targets": "strict binary 0/1; no soft target",
        },
        "training_protocol": {
            "native_sliding_windows": True,
            "one_optimizer_update_per_case": True,
            "split_case_order": train_names,
            "training_case_order_per_epoch": expected_training_order,
            "case_order_rule": "NumPy default_rng(seed) permutation, reset identically per variant",
            "patch_order": "native slicer order",
            "overlap_weight": "inverse-overlap Gaussian weight",
            "loss": "class_balanced_loss",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "training_epochs": args.training_epochs,
            "lora_initialization_digests_BCE": lora_digests,
            "lora_initialization_consistent_BCE": len(set(lora_digests)) == 1,
            "training_order_consistent_BCE": all(
                order == expected_training_order for order in training_orders
            ),
        },
        "cache_protocol": {
            "directory": str(cache_dir),
            "target_dtype": "uint8",
            "scope": "one case in memory at a time; source patches and weights regenerated during training",
            "all_case_float_patch_cache_retained": False,
        },
        "language_action_protocol": {
            "prompts": list(LANGUAGE_PROMPTS),
            "wrong_prompts_used": False,
            "native_decoder_path": True,
            "shared_output_before_projection": True,
        },
        "smoke_validation": {
            "one_train_case": args.smoke_only and len(train_cases) == 1,
            "one_test_case": args.smoke_only and len(eval_cases) == 1,
            "one_patch": args.smoke_only and all(cache["used_slicer_count"] == 1 for cache in all_cache),
            "equal_weight_fusion_contract": True,
            "C_pseudo_strict_binary_uint8": smoke_binary["C_vision_wp_hard_fusion"],
            "E_pseudo_strict_binary_uint8": smoke_binary["E_vision_language_wp_hard_fusion"],
            "gt_not_used_to_construct_pseudo": True,
            "prediction_contract": True,
            "wp_frozen": True,
        },
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "metrics": str(output_dir / "metrics.csv"),
            "per_case": str(output_dir / "per_case.csv"),
            "action_metrics": str(output_dir / "action_metrics.csv"),
            "action_geometry": str(output_dir / "action_geometry.csv"),
            "pseudo_label_metrics": str(output_dir / "pseudo_label_metrics.csv"),
            "training_loss": str(output_dir / "training_loss.csv"),
            "E_comparisons": str(output_dir / "E_comparisons.csv"),
            "lora_checkpoints": [str(output_dir / f"{variant}_lora.pt") for variant in VARIANTS],
        },
        "final_metrics": metrics_rows,
        "E_comparisons": comparison_rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
