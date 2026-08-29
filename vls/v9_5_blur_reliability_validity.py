"""V9.5 blur-reliability validity diagnostic.

Evaluation only: the frozen V9.3 hierarchical residual World Predictor is
loaded from its checkpoint, and its correct-action blur prediction is decoded
through the unchanged VoxTell native decoder.  No model is trained, no loss
or segmentation head is added, and no SFDA path is run.

Reliability follows the existing V8.3 convention: higher scores should mark
source pseudo-label voxels that are correct against GT.  Confidence is the
source prediction confidence rank; real_blur and imagined_blur are one minus
the tie-aware percentile rank of the source-vs-view probability disagreement.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image, read_image_and_label
from vls.v2_experiment import padded_image_and_slicers, padded_visual_action_and_slicers, resolve_device, visual_action
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, pad_label_like_image
from vls.v7_0d_protocol_sanity import set_seed
from vls.v8_3_real_aug_oracle_reliability import rank_metrics, tie_aware_percentile_rank
from vls.voxtell_states import VoxTellStateInterface
from vls.v9_3_hierarchical_residual_world_predictor import LEVEL_NAMES, level_features, native_decoder_from_predicted_skips
from vls.v9_4_hierarchical_wp_matched_diagnostic import (
    BLUR_SIGMA,
    build_text_delta,
    canonical_patch_kind,
    deserialize_slicer,
    load_test_manifest,
    load_v93_world_predictor,
)


OUTPUT_DIR = Path("outputs/v9_5_blur_reliability_validity")
METHODS = ("confidence", "real_blur", "imagined_blur")
SCOPES = ("overall", "foreground")
REGIONS = ("TP", "FP", "FN", "TN")
PATCH_KINDS = ("context", "foreground", "context_fill")

PATCH_FIELDS = (
    "case",
    "patch_index",
    "patch_kind",
    "method",
    "scope",
    "voxel_count",
    "pseudo_accuracy",
    "reliability_mean",
    "auroc",
    "auprc",
    "spearman",
    "disagreement_probability_mse",
    "disagreement_probability_mae",
    "source_gt_dice",
    "real_blur_gt_dice",
    "real_blur_gt_dice_drop_vs_source",
)
REGION_FIELDS = (
    "case",
    "patch_index",
    "patch_kind",
    "method",
    "scope",
    "region",
    "voxel_count",
    "reliability_mean",
)
METRIC_FIELDS = (
    "method",
    "scope",
    "patch_count",
    "valid_auroc_count",
    "pseudo_accuracy",
    "reliability_mean",
    "auroc",
    "auprc",
    "spearman",
    "aggregation",
)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V9.5 blur reliability validity diagnostic")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/v9_3_hierarchical_residual_world_predictor/best_hierarchical_world_predictor.pt",
    )
    parser.add_argument(
        "--patch-manifest",
        default="outputs/v9_3_hierarchical_residual_world_predictor/test_patch_manifest.json",
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0, help="debug smoke-test limit; 0 uses all 8 test cases")
    parser.add_argument("--patch-limit", type=int, default=0, help="debug smoke-test limit per case; 0 uses all 4 manifest patches")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="evaluate fixed manifest patches without writing formal output files",
    )
    return parser.parse_args()


def make_paths(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    return 1.0 if tp + fp + fn == 0 else float(2.0 * tp / max(2 * tp + fp + fn, 1))


def disagreement_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    return {
        "disagreement_probability_mse": float(np.mean(delta.astype(np.float64) ** 2)),
        "disagreement_probability_mae": float(np.mean(np.abs(delta), dtype=np.float64)),
    }


def reliability_maps(
    source_probability: np.ndarray,
    real_blur_probability: np.ndarray,
    imagined_blur_probability: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    source_probability = np.asarray(source_probability, dtype=np.float32)
    real_blur_probability = np.asarray(real_blur_probability, dtype=np.float32)
    imagined_blur_probability = np.asarray(imagined_blur_probability, dtype=np.float32)
    if source_probability.shape != real_blur_probability.shape or source_probability.shape != imagined_blur_probability.shape:
        raise AssertionError(
            f"Probability shape mismatch: source={source_probability.shape}, "
            f"real_blur={real_blur_probability.shape}, imagined_blur={imagined_blur_probability.shape}"
        )
    confidence = np.maximum(source_probability, 1.0 - source_probability)
    real_disagreement = np.abs(source_probability - real_blur_probability)
    imagined_disagreement = np.abs(source_probability - imagined_blur_probability)
    maps = {
        "confidence": tie_aware_percentile_rank(confidence).reshape(confidence.shape),
        "real_blur": 1.0 - tie_aware_percentile_rank(real_disagreement).reshape(real_disagreement.shape),
        "imagined_blur": 1.0 - tie_aware_percentile_rank(imagined_disagreement).reshape(imagined_disagreement.shape),
    }
    raw = {
        "confidence": {"disagreement_probability_mse": None, "disagreement_probability_mae": None},
        "real_blur": disagreement_metrics(source_probability, real_blur_probability),
        "imagined_blur": disagreement_metrics(source_probability, imagined_blur_probability),
    }
    return maps, raw


def correlation(x_values: list[Any], y_values: list[Any]) -> dict[str, Any]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values, strict=True)
        if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y))
    ]
    result: dict[str, Any] = {"n": len(pairs), "pearson": None, "spearman": None}
    if len(pairs) < 2:
        return result
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return result
    result["pearson"] = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy.stats import spearmanr

        value = spearmanr(x, y).statistic
        result["spearman"] = None if not np.isfinite(value) else float(value)
    except ImportError:
        pass
    return result


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def region_mean(rows: list[dict[str, Any]], method: str, scope: str, regions: set[str]) -> float | None:
    selected = [
        row for row in rows
        if row["method"] == method and row["scope"] == scope and row["region"] in regions and int(row["voxel_count"]) > 0
    ]
    total = sum(int(row["voxel_count"]) for row in selected)
    if total == 0:
        return None
    return float(sum(float(row["reliability_mean"]) * int(row["voxel_count"]) for row in selected) / total)


def aggregate_metric_rows(patch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for scope in SCOPES:
            selected = [row for row in patch_rows if row["method"] == method and row["scope"] == scope]
            rows.append({
                "method": method,
                "scope": scope,
                "patch_count": len(selected),
                "valid_auroc_count": sum(row["auroc"] is not None for row in selected),
                **{
                    field: (
                        None if not [row[field] for row in selected if row[field] is not None]
                        else float(np.mean([row[field] for row in selected if row[field] is not None], dtype=np.float64))
                    )
                    for field in ("pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman")
                },
                "aggregation": "patch_macro",
            })
    return rows


def metric_lookup(rows: list[dict[str, Any]], method: str, scope: str) -> dict[str, float | None]:
    row = next(row for row in rows if row["method"] == method and row["scope"] == scope)
    return {field: row[field] for field in ("auroc", "auprc", "spearman", "reliability_mean")}


def method_comparison(metric_rows: list[dict[str, Any]], method: str, reference: str, scope: str) -> dict[str, Any]:
    left = metric_lookup(metric_rows, method, scope)
    right = metric_lookup(metric_rows, reference, scope)
    fields = ("auroc", "auprc", "spearman")
    available = [field for field in fields if left[field] is not None and right[field] is not None]
    return {
        "scope": scope,
        "method": method,
        "reference": reference,
        "method_metrics": left,
        "reference_metrics": right,
        "deltas_method_minus_reference": {
            field: None if field not in available else float(left[field]) - float(right[field])
            for field in fields
        },
        "better_on": [field for field in available if float(left[field]) > float(right[field])],
        "all_available_metrics_strictly_better": bool(available) and all(
            float(left[field]) > float(right[field]) for field in available
        ),
    }


def build_summary(
    args: argparse.Namespace,
    cases: list[Any],
    checkpoint_metadata: dict[str, Any],
    patch_rows: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    direction: dict[str, dict[str, Any]] = {}
    region_summary: dict[str, dict[str, dict[str, float | None]]] = {}
    for method in METHODS:
        region_summary[method] = {}
        for scope in SCOPES:
            region_summary[method][scope] = {
                region: region_mean(region_rows, method, scope, {region}) for region in REGIONS
            }
            correct = region_mean(region_rows, method, scope, {"TP", "TN"})
            incorrect = region_mean(region_rows, method, scope, {"FP", "FN"})
            metric = metric_lookup(metric_rows, method, scope)
            direction.setdefault(method, {})[scope] = {
                "correct_region_reliability_mean": correct,
                "incorrect_region_reliability_mean": incorrect,
                "correct_minus_incorrect": None if correct is None or incorrect is None else correct - incorrect,
                "auroc": metric["auroc"],
                "spearman": metric["spearman"],
                "direction_correct": (
                    correct is not None and incorrect is not None and correct > incorrect
                    and metric["auroc"] is not None and float(metric["auroc"]) >= 0.5
                    and metric["spearman"] is not None and float(metric["spearman"]) >= 0.0
                ),
            }

    correlation_summary: dict[str, dict[str, Any]] = {}
    for scope_name, predicate in [
        ("overall", lambda row: True),
        ("context", lambda row: row["patch_kind"] in {"context", "context_fill"}),
        ("foreground", lambda row: row["patch_kind"] == "foreground"),
        ("context_fill", lambda row: row["patch_kind"] == "context_fill"),
    ]:
        selected = [row for row in patch_rows if row["scope"] == "overall" and predicate(row)]
        drop = [row["real_blur_gt_dice_drop_vs_source"] for row in selected]
        correlation_summary[scope_name] = {}
        for method in ("real_blur", "imagined_blur"):
            correlation_summary[scope_name][method] = {
                "probability_mse_vs_gt_dice_drop": correlation(
                    [row["disagreement_probability_mse"] for row in selected if row["method"] == method],
                    [row["real_blur_gt_dice_drop_vs_source"] for row in selected if row["method"] == method],
                ),
                "probability_mae_vs_gt_dice_drop": correlation(
                    [row["disagreement_probability_mae"] for row in selected if row["method"] == method],
                    [row["real_blur_gt_dice_drop_vs_source"] for row in selected if row["method"] == method],
                ),
            }

    oracle_gap = {}
    for scope in SCOPES:
        imagined = metric_lookup(metric_rows, "imagined_blur", scope)
        real = metric_lookup(metric_rows, "real_blur", scope)
        oracle_gap[scope] = {
            field: None if imagined[field] is None or real[field] is None else float(imagined[field]) - float(real[field])
            for field in ("auroc", "auprc", "spearman")
        }
    return {
        "stage": "V9.5 blur reliability validity diagnostic",
        "evaluation_only": True,
        "models_trained": False,
        "sfda_run": False,
        "voxtell_modified": False,
        "v93_wp_modified": False,
        "loss_modified": False,
        "segmentation_head_added": False,
        "test_case_count": len(cases),
        "test_cases": [case.case for case in cases],
        "fixed_manifest_patch_count": len({(row["case"], row["patch_index"]) for row in patch_rows}),
        "action_protocol": {"blur_sigma": BLUR_SIGMA},
        "checkpoint": checkpoint_metadata,
        "patch_manifest": str(Path(args.patch_manifest)),
        "gt_usage": "GT is used only to label source pseudo-label voxel correctness and compute source/real-blur GT Dice drop.",
        "reliability_definition": {
            "confidence": "tie-aware percentile rank of max(source_probability, 1-source_probability)",
            "real_blur": "1 - tie-aware percentile rank(abs(source_probability-real_blur_probability))",
            "imagined_blur": "1 - tie-aware percentile rank(abs(source_probability-correct_action_wp_blur_probability))",
            "higher_is_more_reliable": True,
            "metric_aggregation": "patch macro; TP/FP/FN/TN means are voxel-count weighted",
        },
        "scopes": {
            "overall": "all fixed manifest voxels",
            "foreground": "GT foreground voxels only",
        },
        "reliability_metrics": metric_rows,
        "region_reliability_means": region_summary,
        "direction_judgment": direction,
        "imagined_blur_vs_confidence": {
            scope: method_comparison(metric_rows, "imagined_blur", "confidence", scope) for scope in SCOPES
        },
        "imagined_blur_vs_real_blur_oracle_gap": oracle_gap,
        "patch_disagreement_vs_real_blur_gt_dice_drop": correlation_summary,
        "outputs": {
            "patch_reliability_metrics": str(OUTPUT_DIR / "patch_reliability_metrics.csv"),
            "voxel_reliability_regions": str(OUTPUT_DIR / "voxel_reliability_regions.csv"),
            "reliability_metrics": str(OUTPUT_DIR / "reliability_metrics.csv"),
            "summary": str(OUTPUT_DIR / "summary.json"),
        },
    }


def run(args: argparse.Namespace) -> None:
    if BLUR_SIGMA != 1.5:
        raise AssertionError("V9.5 requires blur sigma=1.5")
    if args.case_limit < 0 or args.patch_limit < 0:
        raise AssertionError("Debug limits must be non-negative")
    if args.prediction_threshold != 0.5:
        raise AssertionError("V9.3 prediction threshold must remain 0.5")
    set_seed(args.seed)
    paths = make_paths(args)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V9.5 requires CUDA, resolved {device}")
    cases = iter_cases(paths, split="test")
    if len(cases) != 8:
        raise AssertionError(f"V9.5 requires the original 8-case V9.3 test split, got {len(cases)}")
    if args.case_limit:
        cases = cases[: args.case_limit]
    manifest = load_test_manifest(Path(args.patch_manifest), cases, args.patch_limit)

    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    world_predictor, checkpoint_metadata = load_v93_world_predictor(Path(args.world_checkpoint), device)
    prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    text_delta = build_text_delta(interface, device)
    text_backbone = getattr(interface.predictor, "text_backbone", None)
    interface.predictor.text_backbone = None
    interface.predictor.tokenizer = None
    interface.predictor._text_embedding_cache.clear()
    del text_backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    patch_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        print(f"[V9.5] case {case_index}/{len(cases)} start {case.case}", flush=True)
        image, label, _ = read_image_and_label(case)
        source_padded, _ = padded_image_and_slicers(interface.predictor, image)
        blur_padded, _ = padded_visual_action_and_slicers(interface.predictor, image, "blur", BLUR_SIGMA)
        if tuple(source_padded.shape) != tuple(blur_padded.shape):
            raise AssertionError(f"Source/blur padded shape mismatch for {case.case}")
        label_padded = pad_label_like_image(interface, label)
        for record in manifest[case.case]:
            patch_index = int(record["patch_index"])
            patch_kind = canonical_patch_kind(str(record["patch_kind"]))
            slicer = deserialize_slicer(record["slicer"])
            source_patch = torch.clone(source_padded[slicer][None], memory_format=torch.contiguous_format)
            blur_patch = torch.clone(blur_padded[slicer][None], memory_format=torch.contiguous_format)
            source_context = interface.forward_with_audit_context(source_patch, prompt_embedding)
            blur_context = interface.forward_with_audit_context(blur_patch, prompt_embedding)
            source_features = level_features(source_context["decoder_audit"]["skips"], LEVEL_NAMES)
            correct_features = world_predictor(
                source_features,
                visual_action("blur", BLUR_SIGMA, device),
                text_delta,
            )
            imagined_logits = native_decoder_from_predicted_skips(
                interface, source_context, correct_features, LEVEL_NAMES,
            )
            source_probability = torch.sigmoid(source_context["final_prediction"][:, :1]).detach().float().cpu().numpy()
            real_blur_probability = torch.sigmoid(blur_context["final_prediction"][:, :1]).detach().float().cpu().numpy()
            imagined_blur_probability = torch.sigmoid(imagined_logits).detach().float().cpu().numpy()
            gt_np = (label_padded[slicer][None].detach().cpu().numpy() == args.label_value).astype(bool)
            if gt_np.shape != source_probability.shape:
                raise AssertionError(f"GT/source shape mismatch for {case.case}: {gt_np.shape} vs {source_probability.shape}")
            source_pseudo = source_probability > args.prediction_threshold
            source_correct = source_pseudo == gt_np
            source_gt_dice = binary_dice(source_pseudo, gt_np)
            real_blur_gt_dice = binary_dice(real_blur_probability > args.prediction_threshold, gt_np)
            dice_drop = source_gt_dice - real_blur_gt_dice
            maps, raw_disagreement = reliability_maps(
                source_probability, real_blur_probability, imagined_blur_probability,
            )
            flat_gt = gt_np.reshape(-1)
            flat_pseudo = source_pseudo.reshape(-1)
            flat_correct = source_correct.reshape(-1)
            region_masks = {
                "TP": flat_pseudo & flat_gt,
                "FP": flat_pseudo & ~flat_gt,
                "FN": ~flat_pseudo & flat_gt,
                "TN": ~flat_pseudo & ~flat_gt,
            }
            scope_masks = {
                "overall": np.ones(flat_gt.size, dtype=bool),
                "foreground": flat_gt,
            }
            for method in METHODS:
                flat_reliability = maps[method].reshape(-1)
                for scope, scope_mask in scope_masks.items():
                    metrics = rank_metrics(flat_reliability, flat_correct, scope_mask)
                    patch_rows.append({
                        "case": case.case,
                        "patch_index": patch_index,
                        "patch_kind": patch_kind,
                        "method": method,
                        "scope": scope,
                        "voxel_count": int(scope_mask.sum()),
                        **metrics,
                        **raw_disagreement[method],
                        "source_gt_dice": source_gt_dice,
                        "real_blur_gt_dice": real_blur_gt_dice,
                        "real_blur_gt_dice_drop_vs_source": dice_drop,
                    })
                    for region, region_mask in region_masks.items():
                        selected = flat_reliability[region_mask & scope_mask]
                        region_rows.append({
                            "case": case.case,
                            "patch_index": patch_index,
                            "patch_kind": patch_kind,
                            "method": method,
                            "scope": scope,
                            "region": region,
                            "voxel_count": int(selected.size),
                            "reliability_mean": None if selected.size == 0 else float(np.mean(selected, dtype=np.float64)),
                        })
            print(f"[V9.5] case {case_index}/{len(cases)} patch {patch_index + 1}/{len(manifest[case.case])} complete", flush=True)
            del source_patch, blur_patch, source_context, blur_context, source_features, correct_features, imagined_logits
            del source_probability, real_blur_probability, imagined_blur_probability, gt_np, source_pseudo, source_correct
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del image, label, label_padded, source_padded, blur_padded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.smoke_only:
        metric_rows = aggregate_metric_rows(patch_rows)
        print(json.dumps({
            "smoke": "passed",
            "test_cases_evaluated": len(cases),
            "patches_evaluated": len({(row["case"], row["patch_index"]) for row in patch_rows}),
            "patch_rows": len(patch_rows),
            "region_rows": len(region_rows),
            "methods": list(METHODS),
            "overall_metrics": [row for row in metric_rows if row["scope"] == "overall"],
            "checkpoint": checkpoint_metadata["path"],
            "training_run": False,
            "sfda_run": False,
        }, indent=2), flush=True)
        return

    metric_rows = aggregate_metric_rows(patch_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "patch_reliability_metrics.csv", PATCH_FIELDS, patch_rows)
    write_csv(OUTPUT_DIR / "voxel_reliability_regions.csv", REGION_FIELDS, region_rows)
    write_csv(OUTPUT_DIR / "reliability_metrics.csv", METRIC_FIELDS, metric_rows)
    summary = build_summary(args, cases, checkpoint_metadata, patch_rows, region_rows, metric_rows)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"complete": True, "outputs": summary["outputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
