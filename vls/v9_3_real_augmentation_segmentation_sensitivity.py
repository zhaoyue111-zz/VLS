"""V9.3 real-augmentation segmentation prediction sensitivity diagnostic.

This is an independent, inference-only diagnostic.  It runs the unchanged
VoxTell encoder and native decoder on the same selected patch under three
inputs: original, real gamma(+0.30), and real blur(sigma=1.5).  No World
Predictor is loaded, trained, or called, and no VoxTell module is modified.

The per-patch CSV makes the three pairwise prediction disagreements explicit;
summary.json aggregates them by V9.3 patch kind and, when labels are present,
checks whether larger disagreement accompanies a drop in GT segmentation
performance.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Permit both ``python -m vls...`` and the direct invocation used by the
# project scripts (``python vls/...py``).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_image_cases, read_image, read_image_and_label
from vls.v2_experiment import (
    padded_image_and_slicers,
    padded_visual_action_and_slicers,
    resolve_device,
    select_patch_slicers,
)
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT
from vls.v7_0d_protocol_sanity import set_seed
from vls.voxtell_states import VoxTellStateInterface
from vls.v9_0_world_state_selection_audit import pad_selected_patches


OUTPUT_DIR = Path("outputs/v9_3_real_augmentation_segmentation_sensitivity")
GAMMA_STRENGTH = 0.30
BLUR_SIGMA = 1.5
ACTION_PROTOCOL = (("gamma", GAMMA_STRENGTH), ("blur", BLUR_SIGMA))
PATCH_KINDS = ("context", "foreground", "context_fill")
MANIFEST_PATCH_KIND_ALIASES = {"repeat_fill": "context_fill"}

CSV_FIELDS = (
    "case",
    "patch_index",
    "patch_kind",
    "has_gt",
    "original_vs_gamma_probability_mse",
    "original_vs_gamma_probability_mae",
    "original_vs_gamma_mask_dice",
    "original_vs_gamma_mask_iou",
    "original_vs_blur_probability_mse",
    "original_vs_blur_probability_mae",
    "original_vs_blur_mask_dice",
    "original_vs_blur_mask_iou",
    "gamma_vs_blur_probability_mse",
    "gamma_vs_blur_probability_mae",
    "gamma_vs_blur_mask_dice",
    "gamma_vs_blur_mask_iou",
    "original_gt_dice",
    "original_gt_iou",
    "gamma_gt_dice",
    "gamma_gt_iou",
    "blur_gt_dice",
    "blur_gt_iou",
    "gamma_gt_dice_drop_vs_original",
    "gamma_gt_iou_drop_vs_original",
    "blur_gt_dice_drop_vs_original",
    "blur_gt_iou_drop_vs_original",
)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(
        description="V9.3 real augmentation segmentation prediction sensitivity diagnostic"
    )
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0, help="debug limit; 0 uses all V9.3 train cases")
    parser.add_argument("--patch-limit", type=int, default=0, help="debug limit; 0 uses all selected patches")
    parser.add_argument(
        "--patch-manifest",
        default="outputs/v9_3_hierarchical_residual_world_predictor/train_patch_manifest.json",
        help="existing V9.3 patch manifest to reuse; set to an empty string to rescore patches",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="forward one real original/gamma/blur patch and verify output shapes only",
    )
    return parser.parse_args()


def pad_label_like_image(interface: VoxTellStateInterface, label: np.ndarray) -> torch.Tensor:
    """Pad labels with the same spatial padding used by VoxTell input patches."""
    from acvl_utils.cropping_and_padding.padding import pad_nd_image

    padded, _ = pad_nd_image(
        torch.from_numpy(label[None].astype(np.float32, copy=False)),
        interface.predictor.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    return padded


def binary_overlap(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    dice = 1.0 if tp + fp + fn == 0 else 2.0 * tp / max(2 * tp + fp + fn, 1)
    iou = 1.0 if tp + fp + fn == 0 else tp / max(tp + fp + fn, 1)
    return float(dice), float(iou)


def pairwise_prediction_metrics(
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    left_probability = np.asarray(left_probability, dtype=np.float32)
    right_probability = np.asarray(right_probability, dtype=np.float32)
    if left_probability.shape != right_probability.shape:
        raise AssertionError(
            f"Probability shape mismatch: {left_probability.shape} vs {right_probability.shape}"
        )
    dice, iou = binary_overlap(
        left_probability > threshold,
        right_probability > threshold,
    )
    delta = left_probability - right_probability
    return {
        "probability_mse": float(np.mean(np.square(delta), dtype=np.float64)),
        "probability_mae": float(np.mean(np.abs(delta), dtype=np.float64)),
        "mask_dice": dice,
        "mask_iou": iou,
    }


def gt_prediction_metrics(
    probability: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    dice, iou = binary_overlap(probability > threshold, gt_mask)
    return {"dice": dice, "iou": iou}


@torch.inference_mode()
def forward_probability(
    interface: VoxTellStateInterface,
    patch: torch.Tensor,
    prompt_embedding: torch.Tensor,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Run the frozen native encoder+decoder and return one prompt probability."""
    result = interface.forward_with_states(patch, prompt_embedding)
    logits = result["final_prediction"]
    if logits.ndim != 5 or logits.shape[0] != 1 or logits.shape[1] != 1:
        raise AssertionError(f"Expected [1, 1, D, H, W] final logits, got {tuple(logits.shape)}")
    logits = logits[:, 0].detach().float()
    probability = torch.sigmoid(logits)
    return probability[0].cpu().numpy(), tuple(logits.shape)


def forward_triplet(
    interface: VoxTellStateInterface,
    original_patch: torch.Tensor,
    gamma_patch: torch.Tensor,
    blur_patch: torch.Tensor,
    prompt_embedding: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    probabilities: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for name, patch in (("original", original_patch), ("gamma", gamma_patch), ("blur", blur_patch)):
        probabilities[name], shapes[name] = forward_probability(interface, patch, prompt_embedding)
    if len(set(shapes.values())) != 1:
        raise AssertionError(f"Original/gamma/blur output shapes differ: {shapes}")
    return probabilities, shapes


def make_patch_row(
    case_name: str,
    patch_index: int,
    patch_kind: str,
    probabilities: dict[str, np.ndarray],
    gt_mask: np.ndarray | None,
    prediction_threshold: float,
) -> dict[str, Any]:
    pairs = {
        "original_vs_gamma": pairwise_prediction_metrics(
            probabilities["original"], probabilities["gamma"], prediction_threshold,
        ),
        "original_vs_blur": pairwise_prediction_metrics(
            probabilities["original"], probabilities["blur"], prediction_threshold,
        ),
        "gamma_vs_blur": pairwise_prediction_metrics(
            probabilities["gamma"], probabilities["blur"], prediction_threshold,
        ),
    }
    row: dict[str, Any] = {
        "case": case_name,
        "patch_index": patch_index,
        "patch_kind": patch_kind,
        "has_gt": gt_mask is not None,
    }
    for pair_name, metrics in pairs.items():
        for metric_name, value in metrics.items():
            row[f"{pair_name}_{metric_name}"] = value

    for prediction_name in ("original", "gamma", "blur"):
        prefix = f"{prediction_name}_gt"
        if gt_mask is None:
            row[f"{prefix}_dice"] = None
            row[f"{prefix}_iou"] = None
        else:
            metrics = gt_prediction_metrics(
                probabilities[prediction_name], gt_mask, prediction_threshold,
            )
            row[f"{prefix}_dice"] = metrics["dice"]
            row[f"{prefix}_iou"] = metrics["iou"]

    if gt_mask is None:
        for field in (
            "gamma_gt_dice_drop_vs_original",
            "gamma_gt_iou_drop_vs_original",
            "blur_gt_dice_drop_vs_original",
            "blur_gt_iou_drop_vs_original",
        ):
            row[field] = None
    else:
        row["gamma_gt_dice_drop_vs_original"] = row["original_gt_dice"] - row["gamma_gt_dice"]
        row["gamma_gt_iou_drop_vs_original"] = row["original_gt_iou"] - row["gamma_gt_iou"]
        row["blur_gt_dice_drop_vs_original"] = row["original_gt_dice"] - row["blur_gt_dice"]
        row["blur_gt_iou_drop_vs_original"] = row["original_gt_iou"] - row["blur_gt_iou"]
    return row


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def correlation(x_values: list[Any], y_values: list[Any]) -> dict[str, Any]:
    paired = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values, strict=True)
        if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y))
    ]
    result: dict[str, Any] = {"n": len(paired), "pearson": None, "spearman": None}
    if len(paired) < 2:
        return result
    x = np.asarray([item[0] for item in paired], dtype=np.float64)
    y = np.asarray([item[1] for item in paired], dtype=np.float64)
    if np.std(x) > 0 and np.std(y) > 0:
        result["pearson"] = float(np.corrcoef(x, y)[0, 1])
        try:
            from scipy.stats import spearmanr

            spearman = spearmanr(x, y).statistic
            result["spearman"] = None if not np.isfinite(spearman) else float(spearman)
        except ImportError:
            pass
    return result


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_summary = {}
    for pair_name in ("original_vs_gamma", "original_vs_blur", "gamma_vs_blur"):
        pair_summary[pair_name] = {
            metric: mean_field(rows, f"{pair_name}_{metric}")
            for metric in ("probability_mse", "probability_mae", "mask_dice", "mask_iou")
        }
    gt_summary = {
        prediction_name: {
            "dice": mean_field(rows, f"{prediction_name}_gt_dice"),
            "iou": mean_field(rows, f"{prediction_name}_gt_iou"),
        }
        for prediction_name in ("original", "gamma", "blur")
    }
    relation = {}
    for action, pair_name in (("gamma", "original_vs_gamma"), ("blur", "original_vs_blur")):
        relation[action] = {
            "probability_mse_vs_gt_dice_drop": correlation(
                [row.get(f"{pair_name}_probability_mse") for row in rows],
                [row.get(f"{action}_gt_dice_drop_vs_original") for row in rows],
            ),
            "probability_mae_vs_gt_dice_drop": correlation(
                [row.get(f"{pair_name}_probability_mae") for row in rows],
                [row.get(f"{action}_gt_dice_drop_vs_original") for row in rows],
            ),
            "probability_mse_vs_gt_iou_drop": correlation(
                [row.get(f"{pair_name}_probability_mse") for row in rows],
                [row.get(f"{action}_gt_iou_drop_vs_original") for row in rows],
            ),
            "probability_mae_vs_gt_iou_drop": correlation(
                [row.get(f"{pair_name}_probability_mae") for row in rows],
                [row.get(f"{action}_gt_iou_drop_vs_original") for row in rows],
            ),
        }
    return {
        "patch_count": len(rows),
        "gt_patch_count": sum(bool(row["has_gt"]) for row in rows),
        "prediction_disagreement": pair_summary,
        "gt_performance": gt_summary,
        "disagreement_performance_relationship": relation,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_protocol(args: argparse.Namespace) -> None:
    if ACTION_PROTOCOL != (("gamma", 0.30), ("blur", 1.5)):
        raise AssertionError("Real augmentation protocol must remain gamma=0.30 and blur=1.5")
    if (
        args.patches_per_case != 4
        or args.foreground_patches_per_case != 2
        or args.foreground_candidate_patches != 16
    ):
        raise AssertionError("Patch protocol must match V9.3: patches=4, foreground=2, candidates=16")
    if args.foreground_threshold != 0.5 or args.prediction_threshold != 0.5:
        raise AssertionError("V9.3 thresholds must remain 0.5")
    if args.case_limit < 0 or args.patch_limit < 0:
        raise AssertionError("Debug limits must be non-negative")


def make_paths(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )


def load_train_cases(paths: ProjectPaths, args: argparse.Namespace) -> list[Any]:
    cases = iter_image_cases(paths, split="train")
    if len(cases) != 30:
        raise AssertionError(f"V9.3 requires the complete 30-case train split, got {len(cases)}")
    if args.case_limit:
        cases = cases[: args.case_limit]
    return cases


def deserialize_slicer(value: list[list[int | None]]) -> tuple[slice, ...]:
    return tuple(slice(start, stop, step) for start, stop, step in value)


def load_patch_manifest(path: Path, cases: list[Any], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise AssertionError(f"V9.3 patch manifest must contain a list: {path}")
    expected_cases = {case.case for case in cases}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        case_name = str(record.get("case", ""))
        if case_name in expected_cases:
            grouped.setdefault(case_name, []).append(record)
    missing = sorted(expected_cases - set(grouped))
    if missing:
        raise AssertionError(f"Patch manifest is missing requested cases: {missing[:3]}")
    for case_name, case_records in grouped.items():
        if len(case_records) < args.patches_per_case:
            raise AssertionError(
                f"Patch manifest has {len(case_records)} patches for {case_name}; "
                f"expected at least {args.patches_per_case}"
            )
        case_records.sort(key=lambda record: int(record.get("patch_index", 0)))
        for record in case_records[: args.patches_per_case]:
            if record.get("patch_kind") not in (*PATCH_KINDS, *MANIFEST_PATCH_KIND_ALIASES):
                raise AssertionError(f"Unknown patch kind in manifest: {record.get('patch_kind')}")
            if "slicer" not in record:
                raise AssertionError(f"Manifest record has no slicer: {record}")
    return grouped


def build_selected_inputs(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    prompt_embedding: torch.Tensor,
    args: argparse.Namespace,
    manifest_records: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[tuple], list[str]]:
    if manifest_records is None:
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface,
            image,
            prompt_embedding,
            args.patches_per_case,
            args.foreground_patches_per_case,
            args.foreground_candidate_patches,
            args.foreground_threshold,
        )
        slicers, patch_kinds = pad_selected_patches(
            slicers, patch_kinds, args.patches_per_case,
        )
    else:
        original_padded, _ = padded_image_and_slicers(interface.predictor, image)
        records = manifest_records[: args.patches_per_case]
        slicers = [deserialize_slicer(record["slicer"]) for record in records]
        patch_kinds = [str(record["patch_kind"]) for record in records]
    if args.patch_limit:
        slicers = slicers[: args.patch_limit]
        patch_kinds = patch_kinds[: args.patch_limit]
    patch_kinds = [MANIFEST_PATCH_KIND_ALIASES.get(kind, kind) for kind in patch_kinds]
    action_padded = {
        action: padded_visual_action_and_slicers(interface.predictor, image, action, strength)[0]
        for action, strength in ACTION_PROTOCOL
    }
    for action, padded in action_padded.items():
        if tuple(padded.shape) != tuple(original_padded.shape):
            raise AssertionError(f"{action} padded shape differs from original: {padded.shape} vs {original_padded.shape}")
    return original_padded, action_padded, slicers, patch_kinds


def run_sanity(args: argparse.Namespace) -> None:
    paths = make_paths(args)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V9.3 requires CUDA, resolved {device}")
    cases = load_train_cases(paths, args)
    if not cases:
        raise AssertionError("No train cases available for sanity check")
    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    manifest = None
    if args.patch_manifest:
        manifest_path = Path(args.patch_manifest)
        if manifest_path.exists():
            manifest = load_patch_manifest(manifest_path, cases, args)
    image, _ = read_image(cases[0])
    original_padded, action_padded, slicers, patch_kinds = build_selected_inputs(
        interface, image, prompt_embedding, args,
        None if manifest is None else manifest[cases[0].case],
    )
    if not slicers:
        raise AssertionError("Patch selection returned no patch")
    slicer = slicers[0]
    patches = {
        "original": torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format),
        "gamma": torch.clone(action_padded["gamma"][slicer][None], memory_format=torch.contiguous_format),
        "blur": torch.clone(action_padded["blur"][slicer][None], memory_format=torch.contiguous_format),
    }
    probabilities, output_shapes = forward_triplet(
        interface, patches["original"], patches["gamma"], patches["blur"], prompt_embedding,
    )
    print(json.dumps({
        "sanity": "passed",
        "case": cases[0].case,
        "patch_kind": patch_kinds[0],
        "input_shapes": {name: list(value.shape) for name, value in patches.items()},
        "output_shapes": {name: list(shape) for name, shape in output_shapes.items()},
        "output_shapes_equal": len(set(output_shapes.values())) == 1,
        "probability_ranges": {
            name: [float(np.min(value)), float(np.max(value))]
            for name, value in probabilities.items()
        },
        "world_predictor_loaded": False,
        "decoder_modified": False,
    }, indent=2), flush=True)


def run(args: argparse.Namespace) -> None:
    validate_protocol(args)
    set_seed(args.seed)
    if args.sanity_only:
        run_sanity(args)
        return

    paths = make_paths(args)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V9.3 requires CUDA, resolved {device}")
    cases = load_train_cases(paths, args)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "augmentation_prediction_sensitivity.csv"
    summary_path = output_dir / "summary.json"

    print("[V9.3 diagnostic] loading frozen VoxTell encoder+native decoder; no WP", flush=True)
    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    manifest = None
    if args.patch_manifest:
        manifest_path = Path(args.patch_manifest)
        if manifest_path.exists():
            manifest = load_patch_manifest(manifest_path, cases, args)
    text_backbone = getattr(interface.predictor, "text_backbone", None)
    interface.predictor.text_backbone = None
    interface.predictor.tokenizer = None
    interface.predictor._text_embedding_cache.clear()
    del text_backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    gt_case_count = 0
    for case_index, case in enumerate(cases, start=1):
        print(f"[V9.3 diagnostic] case {case_index}/{len(cases)} start {case.case}", flush=True)
        image, _ = read_image(case)
        label: np.ndarray | None = None
        if case.label_path.exists():
            _, label, _ = read_image_and_label(case)
            gt_case_count += 1
        original_padded, action_padded, slicers, patch_kinds = build_selected_inputs(
            interface, image, prompt_embedding, args,
            None if manifest is None else manifest[case.case],
        )
        label_padded = pad_label_like_image(interface, label) if label is not None else None
        for patch_index, (slicer, patch_kind) in enumerate(zip(slicers, patch_kinds, strict=True)):
            patches = {
                "original": torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format),
                "gamma": torch.clone(action_padded["gamma"][slicer][None], memory_format=torch.contiguous_format),
                "blur": torch.clone(action_padded["blur"][slicer][None], memory_format=torch.contiguous_format),
            }
            probabilities, _ = forward_triplet(
                interface, patches["original"], patches["gamma"], patches["blur"], prompt_embedding,
            )
            gt_mask = None
            if label_padded is not None:
                gt_patch = label_padded[slicer][0].detach().cpu().numpy()
                gt_mask = gt_patch == args.label_value
                if gt_mask.shape != probabilities["original"].shape:
                    raise AssertionError(
                        f"GT/prediction shape mismatch for {case.case} patch={patch_index}: "
                        f"{gt_mask.shape} vs {probabilities['original'].shape}"
                    )
            rows.append(make_patch_row(
                case.case, patch_index, patch_kind, probabilities, gt_mask, args.prediction_threshold,
            ))
            print(
                f"[V9.3 diagnostic] case {case_index}/{len(cases)} "
                f"patch {patch_index + 1}/{len(slicers)} kind={patch_kind} complete",
                flush=True,
            )
            del patches, probabilities
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del action_padded, original_padded, image, label, label_padded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(csv_path, rows)
    by_patch_kind = {
        patch_kind: summarize_group([row for row in rows if row["patch_kind"] == patch_kind])
        for patch_kind in PATCH_KINDS
    }
    summary = {
        "stage": "V9.3 real augmentation segmentation prediction sensitivity diagnostic",
        "split": "train",
        "train_case_count": len(cases),
        "patch_count": len(rows),
        "gt_available_case_count": gt_case_count,
        "gt_available_patch_count": sum(bool(row["has_gt"]) for row in rows),
        "models_trained": False,
        "world_predictor_loaded": False,
        "decoder_modified": False,
        "segmentation_head_modified": False,
        "action_protocol": {"gamma_strength": GAMMA_STRENGTH, "blur_sigma": BLUR_SIGMA},
        "patch_protocol": {
            "patches_per_case": args.patches_per_case,
            "foreground_patches_per_case": args.foreground_patches_per_case,
            "foreground_candidate_patches": args.foreground_candidate_patches,
            "foreground_threshold": args.foreground_threshold,
            "selection_uses_gt": False,
            "short_volume_fill": "deterministic repeat_fill",
            "patch_manifest": args.patch_manifest or None,
        },
        "prediction_definition": "sigmoid of native VoxTell final_prediction logits for the single liver prompt",
        "patch_kind_summary": by_patch_kind,
        "overall_summary": summarize_group(rows),
        "interpretation": {
            "prediction_changed": "Use nonzero original_vs_gamma/original_vs_blur probability MAE or MSE and mask Dice/IoU below 1 as evidence of prediction change.",
            "performance_decline": "GT Dice/IoU drop is original performance minus the corresponding gamma or blur performance; positive means degradation.",
            "relationship": "Positive Pearson/Spearman disagreement-versus-drop values indicate that stronger prediction disagreement tends to accompany larger GT performance decline; correlation is descriptive, not causal.",
        },
        "outputs": {
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), "patch_count": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

'''
python -u vls/v9_3_real_augmentation_segmentation_sensitivity.py --patch-manifest outputs/v9_3_hierarchical_residual_world_predictor/train_patch_manifest.json
'''