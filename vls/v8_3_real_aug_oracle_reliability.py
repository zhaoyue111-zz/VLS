"""V8.3 real-augmentation oracle reliability diagnostic.

This is diagnostic-only.  It does not load or modify the World Predictor and
does not train any model.  It replaces imagined action states with real
gamma/blur VoxTell predictions, while preserving the V7/V8 confidence-rank,
percentile-rank stability, and joint-product definitions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import (
    padded_visual_action_and_slicers,
    resolve_device,
    select_patch_slicers,
)
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, pad_label_like_image
from vls.v7_0d_protocol_sanity import binary_metrics, set_seed
from vls.voxtell_states import VoxTellStateInterface


ACTION_PROTOCOL = (("gamma", 0.30), ("blur", 1.5))
RELIABILITY_METHODS = (
    "confidence_rank",
    "real_gamma_world_stability",
    "real_blur_world_stability",
    "real_pairwise_world_stability",
    "real_gamma_joint_product",
    "real_blur_joint_product",
    "real_pairwise_joint_product",
)
SCOPES = ("overall", "foreground", "background")
REGIONS = ("TP", "FP", "FN", "TN")
METRIC_NAMES = ("pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman")

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover - exercised only in minimal environments
    average_precision_score = None
    roc_auc_score = None
try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover - scipy is already a project dependency
    spearmanr = None


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V8.3 real-augmentation oracle reliability diagnostic")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--output-dir", default="outputs/v8_3_real_aug_oracle_reliability")
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


class CsvSink:
    def __init__(self, path: Path, fields: tuple[str, ...]) -> None:
        self.handle = path.open("w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(fields), extrasaction="ignore")
        self.writer.writeheader()
        self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: Any) -> None:
        if value is None:
            return
        value = float(value)
        if np.isfinite(value):
            self.total += value
            self.count += 1

    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class PatchMetricAccumulator:
    patch_count: int = 0
    metrics: dict[str, RunningMean] = field(default_factory=dict)

    def update(self, row: dict[str, Any]) -> None:
        self.patch_count += 1
        for name in METRIC_NAMES:
            self.metrics.setdefault(name, RunningMean()).update(row.get(name))

    def row(self, method: str, case: str, scope: str, aggregation: str) -> dict[str, Any]:
        return {
            "method": method,
            "case": case,
            "scope": scope,
            "patch_count": self.patch_count,
            "valid_auroc_count": self.metrics.get("auroc", RunningMean()).count,
            "aggregation": aggregation,
            **{name: self.metrics.get(name, RunningMean()).mean() for name in METRIC_NAMES},
        }


@dataclass
class RegionAccumulator:
    count: int = 0
    reliability_sum: float = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size:
            self.count += int(values.size)
            self.reliability_sum += float(values.sum())

    def mean(self) -> float | None:
        return self.reliability_sum / self.count if self.count else None


METRIC_FIELDS = (
    "method", "case", "scope", "patch_count", "valid_auroc_count",
    "pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman", "aggregation",
)
REGION_FIELDS = ("method", "case", "region", "voxel_count", "reliability_mean", "aggregation")
PATCH_FIELDS = (
    "case", "patch_index", "patch_kind", "method", "scope", "voxel_count",
    "pseudo_accuracy", "reliability_mean", "auroc", "auprc", "spearman",
)


def memory_status(device: torch.device) -> dict[str, float]:
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    result = {"cpu_rss_mb": float(rss_mb)}
    if device.type == "cuda" and torch.cuda.is_available():
        result.update({
            "gpu_allocated_mb": float(torch.cuda.memory_allocated(device) / 1024**2),
            "gpu_reserved_mb": float(torch.cuda.memory_reserved(device) / 1024**2),
        })
    else:
        result.update({"gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0})
    return result


def write_progress(path: Path, completed_cases: list[str], rows: int, device: torch.device, stage: str) -> None:
    path.write_text(json.dumps({
        "stage": stage,
        "completed_cases": completed_cases,
        "completed_case_count": len(completed_cases),
        "completed_patch_rows": rows,
        "memory": memory_status(device),
    }, indent=2))


def rank_metrics(reliability: np.ndarray, correct: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    scores = np.asarray(reliability, dtype=np.float64).reshape(-1)[mask.reshape(-1)]
    labels = np.asarray(correct, dtype=np.int8).reshape(-1)[mask.reshape(-1)]
    if scores.size == 0:
        return {name: None for name in METRIC_NAMES}
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive and n_negative:
        if roc_auc_score is not None:
            auroc = float(roc_auc_score(labels, scores))
        else:
            ranks = rankdata(scores)
            auroc = float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))
    else:
        auroc = None
    if n_positive:
        if average_precision_score is not None:
            auprc = float(average_precision_score(labels, scores))
        else:
            order = np.argsort(-scores, kind="mergesort")
            sorted_labels = labels[order]
            cumulative = np.cumsum(sorted_labels == 1)
            precision = cumulative / np.arange(1, labels.size + 1)
            auprc = float(precision[sorted_labels == 1].sum() / n_positive)
    else:
        auprc = None
    if spearmanr is not None and np.ptp(scores) > 0.0 and np.ptp(labels) > 0.0:
        correlation = spearmanr(scores, labels).statistic
        spearman = None if not np.isfinite(correlation) else float(correlation)
    elif np.ptp(scores) == 0.0 or np.ptp(labels) == 0.0:
        spearman = None
    else:
        score_rank = rankdata(scores)
        label_rank = rankdata(labels)
        score_rank -= score_rank.mean()
        label_rank -= label_rank.mean()
        denominator = float(np.linalg.norm(score_rank) * np.linalg.norm(label_rank))
        spearman = None if denominator == 0.0 else float(np.dot(score_rank, label_rank) / denominator)
    return {
        "pseudo_accuracy": float(labels.mean()),
        "reliability_mean": float(scores.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "spearman": spearman,
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, values.size]
    midranks = 0.5 * (starts + 1 + ends)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat(midranks, ends - starts)
    return ranks


def tie_aware_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the V7/V8 tie-aware percentile rank."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, values.size]
    midranks = 0.5 * (starts + 1 + ends) / max(values.size, 1)
    ranks = np.empty(values.size, dtype=np.float32)
    ranks[order] = np.repeat(midranks.astype(np.float32), ends - starts)
    return ranks


def binary_metric_iou(prediction: np.ndarray, target: np.ndarray) -> dict[str, int | float]:
    metrics = binary_metrics(prediction, target)
    tp, fp, fn = int(metrics["tp"]), int(metrics["fp"]), int(metrics["fn"])
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(metrics["tn"]),
        "iou": 1.0 if tp + fp + fn == 0 else tp / max(tp + fp + fn, 1),
    }


def real_augmentation_reliabilities(
    source_probability: np.ndarray,
    gamma_probability: np.ndarray,
    blur_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    source_probability = np.asarray(source_probability, dtype=np.float32)
    gamma_probability = np.asarray(gamma_probability, dtype=np.float32)
    blur_probability = np.asarray(blur_probability, dtype=np.float32)
    confidence = np.maximum(source_probability, 1.0 - source_probability)
    confidence_rank = tie_aware_percentile_rank(confidence).reshape(confidence.shape)
    gamma_delta = np.abs(source_probability - gamma_probability)
    blur_delta = np.abs(source_probability - blur_probability)
    pairwise_delta = np.mean(np.stack([
        gamma_delta,
        blur_delta,
        np.abs(gamma_probability - blur_probability),
    ], axis=0), axis=0)
    gamma_stability = 1.0 - tie_aware_percentile_rank(gamma_delta).reshape(gamma_delta.shape)
    blur_stability = 1.0 - tie_aware_percentile_rank(blur_delta).reshape(blur_delta.shape)
    pairwise_stability = 1.0 - tie_aware_percentile_rank(pairwise_delta).reshape(pairwise_delta.shape)
    return {
        "confidence_rank": confidence_rank,
        "real_gamma_world_stability": gamma_stability,
        "real_blur_world_stability": blur_stability,
        "real_pairwise_world_stability": pairwise_stability,
        "real_gamma_joint_product": (confidence_rank * gamma_stability).astype(np.float32),
        "real_blur_joint_product": (confidence_rank * blur_stability).astype(np.float32),
        "real_pairwise_joint_product": (confidence_rank * pairwise_stability).astype(np.float32),
    }


@torch.inference_mode()
def predict_probability(
    interface: VoxTellStateInterface,
    patch: torch.Tensor,
    prompt_embedding: torch.Tensor,
) -> np.ndarray:
    result = interface.forward_with_states(patch, prompt_embedding)
    probability = torch.sigmoid(result["final_prediction"][:, 0:1]).detach().float().cpu().numpy()
    del result, patch
    return probability


def run(args: argparse.Namespace) -> None:
    if ACTION_PROTOCOL != (("gamma", 0.30), ("blur", 1.5)):
        raise AssertionError("V8.3 action protocol is not gamma=0.30 and blur=1.5")
    if args.patches_per_case <= 0 or args.foreground_patches_per_case < 0:
        raise AssertionError("Invalid patch protocol")
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V8.3 requires CUDA, resolved {device}")
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    test_cases = iter_cases(paths, split="test")
    if len(test_cases) != 8:
        raise AssertionError(f"V8.3 requires exactly 8 test cases, got {len(test_cases)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_sink = CsvSink(output_dir / "reliability_metrics.csv", METRIC_FIELDS)
    region_sink = CsvSink(output_dir / "reliability_regions.csv", REGION_FIELDS)
    patch_sink = CsvSink(output_dir / "patch_reliability_metrics.csv", PATCH_FIELDS)
    progress_path = output_dir / "progress.json"
    completed_cases: list[str] = []
    completed_patch_rows = 0
    write_progress(progress_path, completed_cases, completed_patch_rows, device, "initializing")

    try:
        print("[V8.3] loading frozen VoxTell/base model", flush=True)
        interface = VoxTellStateInterface.from_model_dir(
            paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
        )
        prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
        text_backbone = getattr(interface.predictor, "text_backbone", None)
        interface.predictor.text_backbone = None
        interface.predictor.tokenizer = None
        interface.predictor._text_embedding_cache.clear()
        del text_backbone
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        global_metrics: dict[tuple[str, str], dict[str, RunningMean]] = {}
        global_regions: dict[tuple[str, str], RunningMean] = {}

        for case_index, case in enumerate(test_cases, start=1):
            print(f"[V8.3] case {case_index}/8 start {case.case}", flush=True)
            image, label, _ = read_image_and_label(case)
            label_padded = pad_label_like_image(interface, label)
            original_padded, slicers, patch_kinds = select_patch_slicers(
                interface,
                image,
                prompt_embedding,
                args.patches_per_case,
                args.foreground_patches_per_case,
                args.foreground_candidate_patches,
                args.foreground_threshold,
            )
            if len(slicers) != args.patches_per_case:
                raise AssertionError(f"Patch selector returned too few patches for {case.case}")

            case_metrics: dict[tuple[str, str], PatchMetricAccumulator] = {}
            case_regions: dict[tuple[str, str], RegionAccumulator] = {}
            for patch_index, slicer in enumerate(slicers, start=1):
                print(
                    f"[V8.3] case {case_index}/8 patch {patch_index}/{len(slicers)} source forward",
                    flush=True,
                )
                source_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
                source_probability = predict_probability(interface, source_patch, prompt_embedding)
                gt_np = (label_padded[slicer][None].detach().cpu().numpy() == args.label_value).astype(bool)
                source_pseudo = source_probability > args.prediction_threshold
                correct_np = source_pseudo == gt_np
                transformed_probabilities: dict[str, np.ndarray] = {}
                for action, strength in ACTION_PROTOCOL:
                    print(
                        f"[V8.3] case {case_index}/8 patch {patch_index}/{len(slicers)} action={action} forward",
                        flush=True,
                    )
                    transformed_padded, _ = padded_visual_action_and_slicers(
                        interface.predictor, image, action, strength,
                    )
                    if tuple(transformed_padded.shape) != tuple(original_padded.shape):
                        raise AssertionError(f"Action padded shape mismatch for {case.case} {action}")
                    action_patch = torch.clone(transformed_padded[slicer][None], memory_format=torch.contiguous_format)
                    transformed_probabilities[action] = predict_probability(interface, action_patch, prompt_embedding)
                    del transformed_padded, action_patch

                reliability_maps = real_augmentation_reliabilities(
                    source_probability,
                    transformed_probabilities["gamma"],
                    transformed_probabilities["blur"],
                )
                flat_gt = gt_np.reshape(-1)
                flat_pseudo = source_pseudo.reshape(-1)
                flat_correct = correct_np.reshape(-1)
                for method, reliability in reliability_maps.items():
                    flat_reliability = reliability.reshape(-1)
                    regions = {
                        "TP": flat_pseudo & flat_gt,
                        "FP": flat_pseudo & ~flat_gt,
                        "FN": ~flat_pseudo & flat_gt,
                        "TN": ~flat_pseudo & ~flat_gt,
                    }
                    scopes = {
                        "overall": np.ones(flat_gt.size, dtype=bool),
                        "foreground": flat_gt,
                        "background": ~flat_gt,
                    }
                    for scope, mask in scopes.items():
                        patch_metrics = rank_metrics(flat_reliability, flat_correct, mask)
                        patch_sink.write({
                            "case": case.case,
                            "patch_index": patch_index - 1,
                            "patch_kind": patch_kinds[patch_index - 1],
                            "method": method,
                            "scope": scope,
                            "voxel_count": int(mask.sum()),
                            **patch_metrics,
                        })
                        case_metrics.setdefault((method, scope), PatchMetricAccumulator()).update(patch_metrics)
                    for region, mask in regions.items():
                        case_regions.setdefault((method, region), RegionAccumulator()).update(
                            flat_reliability[mask],
                        )

                del reliability_maps, transformed_probabilities, source_probability
                del gt_np, source_pseudo, correct_np
                gc.collect()
                print(
                    f"[V8.3] case {case_index}/8 patch {patch_index}/{len(slicers)} reliability done",
                    flush=True,
                )

            for (method, scope), accumulator in sorted(case_metrics.items()):
                row = accumulator.row(method, case.case, scope, "case_patch_macro")
                metric_sink.write(row)
                target = global_metrics.setdefault((method, scope), {})
                for name in METRIC_NAMES:
                    target.setdefault(name, RunningMean()).update(row.get(name))
            for (method, region), accumulator in sorted(case_regions.items()):
                region_sink.write({
                    "method": method,
                    "case": case.case,
                    "region": region,
                    "voxel_count": accumulator.count,
                    "reliability_mean": accumulator.mean(),
                    "aggregation": "case_voxel_weighted",
                })
                global_regions.setdefault((method, region), RunningMean()).update(accumulator.mean())

            completed_cases.append(case.case)
            completed_patch_rows += len(slicers)
            write_progress(progress_path, completed_cases, completed_patch_rows, device, "running")
            print(
                f"[V8.3] case {case_index}/8 complete memory={memory_status(device)}",
                flush=True,
            )
            del case_metrics, case_regions, label_padded, original_padded, image, label
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        foreground_rows: dict[str, dict[str, Any]] = {}
        for (method, scope), metrics in sorted(global_metrics.items()):
            row = {
                "method": method,
                "case": "__all__",
                "scope": scope,
                "patch_count": None,
                "valid_auroc_count": metrics.get("auroc", RunningMean()).count,
                "aggregation": "8_case_macro",
                **{name: metrics.get(name, RunningMean()).mean() for name in METRIC_NAMES},
            }
            metric_sink.write(row)
            if scope == "foreground":
                foreground_rows[method] = row
        region_summary: dict[str, dict[str, float | None]] = defaultdict(dict)
        for (method, region), statistic in sorted(global_regions.items()):
            mean = statistic.mean()
            region_sink.write({
                "method": method,
                "case": "__all__",
                "region": region,
                "voxel_count": None,
                "reliability_mean": mean,
                "aggregation": "8_case_macro",
            })
            region_summary[method][region] = mean

        def method_supported(method: str) -> bool | None:
            row = foreground_rows.get(method)
            if row is None or row.get("auroc") is None or row.get("spearman") is None:
                return None
            return float(row["auroc"]) >= 0.5 and float(row["spearman"]) >= 0.0

        def method_inversion(method: str) -> bool | None:
            row = foreground_rows.get(method)
            if row is None or row.get("auroc") is None or row.get("spearman") is None:
                return None
            return float(row["auroc"]) < 0.5 or float(row["spearman"]) < 0.0

        world_methods = {
            "gamma_only": "real_gamma_world_stability",
            "blur_only": "real_blur_world_stability",
            "gamma_blur_pairwise": "real_pairwise_world_stability",
        }
        method_status = {name: method_supported(method) for name, method in world_methods.items()}
        inversion = {name: method_inversion(method) for name, method in world_methods.items()}
        if any(value is True for value in inversion.values()):
            real_supported: bool | None = False
        elif all(value is True for value in method_status.values()):
            real_supported = True
        else:
            real_supported = None
        summary = {
            "stage": "V8.3 Real-Aug Oracle Reliability Diagnostic",
            "test_cases": [case.case for case in test_cases],
            "test_case_count": len(test_cases),
            "completed_case_count": len(completed_cases),
            "action_protocol": {"gamma": 0.30, "blur": 1.5},
            "patch_protocol": {
                "patches_per_case": args.patches_per_case,
                "foreground_patches_per_case": args.foreground_patches_per_case,
            },
            "world_predictor_loaded": False,
            "models_trained": False,
            "gt_used_only_for_reliability_diagnosis": True,
            "reliability_formula_modified": False,
            "reliability_construction": {
                "source": "real VoxTell source probability",
                "gamma": "real gamma0.3 VoxTell probability",
                "blur": "real blur1.5 VoxTell probability",
                "percentile_rank": "reused V7/V8 tie-aware percentile rank",
                "world_stability": "1 - percentile_rank(abs probability difference)",
                "pairwise_stability": "1 - percentile_rank(mean absolute difference over source/gamma/blur pairs)",
                "joint_product": "confidence_rank * real augmentation world stability",
            },
            "real_augmentation_reliability_supported": real_supported,
            "gamma_only_effective": method_status["gamma_only"],
            "blur_only_effective": method_status["blur_only"],
            "gamma_blur_pairwise_effective": method_status["gamma_blur_pairwise"],
            "foreground_reliability_inversion": inversion,
            "foreground_metrics": {
                name: {
                    "method": method,
                    "auroc": foreground_rows.get(method, {}).get("auroc"),
                    "auprc": foreground_rows.get(method, {}).get("auprc"),
                    "spearman": foreground_rows.get(method, {}).get("spearman"),
                }
                for name, method in world_methods.items()
            },
            "foreground_region_reliability_means": {
                name: {region: region_summary.get(method, {}).get(region) for region in REGIONS}
                for name, method in world_methods.items()
            },
            "confidence_and_joint_foreground_metrics": {
                method: {
                    "auroc": foreground_rows.get(method, {}).get("auroc"),
                    "auprc": foreground_rows.get(method, {}).get("auprc"),
                    "spearman": foreground_rows.get(method, {}).get("spearman"),
                }
                for method in RELIABILITY_METHODS
                if method in foreground_rows
            },
            "imagined_reliability_comparison": {
                "available": False,
                "note": "V8.3 intentionally does not load or evaluate the World Predictor; compare against V8.2 summary separately",
            },
            "interpretation": {
                "real_reliability_formula_issue_if_inversion": "If real gamma/blur/pairwise foreground AUROC < 0.5 or Spearman < 0, the reliability definition itself is implicated independently of World Predictor accuracy",
                "next_step_if_real_supported_but_imagined_invalid": "Improve World Predictor task-relevant transition supervision",
            },
            "outputs": {
                "reliability_metrics": str(output_dir / "reliability_metrics.csv"),
                "reliability_regions": str(output_dir / "reliability_regions.csv"),
                "patch_reliability_metrics": str(output_dir / "patch_reliability_metrics.csv"),
                "progress": str(progress_path),
                "summary": str(output_dir / "summary.json"),
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_progress(progress_path, completed_cases, completed_patch_rows, device, "complete")
        print("[V8.3] complete", flush=True)
    finally:
        for sink in (metric_sink, region_sink, patch_sink):
            sink.close()


if __name__ == "__main__":
    run(parse_args())
