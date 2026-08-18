"""V7.2 fresh held-out confirmation using the V7.1d protocol unchanged.

This runner reuses the V7.1d implementation and changes only the evaluation
case manifest plus the reporting/bootstrap layer.  It intentionally does not
introduce action-specific World reliability or any new training component.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vls.config import ProjectPaths
from vls.data import CaseRecord, iter_image_cases, read_image_and_label
from vls.v2_experiment import prepare_functional_seg_head, resolve_device
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, load_world_model
from vls.v7_0d_protocol_sanity import VARIANTS, build_evaluation_cache, set_seed
from vls.v7_1a_lora_qkv_smoke import lora_parameters
from vls.v7_1b_protocol_consolidation import (
    EVAL_STEPS,
    evaluate_full_volume,
    pool_full_volume,
    write_csv,
)
from vls.v7_1c_class_balanced_loss_sanity import (
    build_cache_from_manifest,
    train_variant,
    validate_manifest,
)
from vls.voxtell_states import VoxTellStateInterface


FRESH_CASES = [
    "HEM_PA81.nii.gz",
    "HEM_PA89.nii.gz",
    "HEM_PA101.nii.gz",
    "HCC_10020116_20180118.nii.gz",
    "MET_PA13.nii.gz",
    "HCC_10074592_20130816.nii.gz",
    "CYST_PA19.nii.gz",
    "HEM_PA13.nii.gz",
]

V3_2E_CASES = [
    "HEM_PA99.nii.gz",
    "HCC_10140276_20170721.nii.gz",
    "HEM_PA97.nii.gz",
    "CYST_PA69.nii.gz",
    "HCC_10076307_20130925.nii.gz",
    "CYST_PA20.nii.gz",
    "CYST_PA56.nii.gz",
    "HCC_10136895_20170421.nii.gz",
    "CYST_PA62.nii.gz",
    "HEM_PA29.nii.gz",
    "HEP_PA28.nii.gz",
    "HEM_PA107.nii.gz",
]

V6_CASES = [
    "CYST_PA62.nii.gz",
    "HEM_PA29.nii.gz",
    "HEP_PA28.nii.gz",
    "HEM_PA107.nii.gz",
]

V7_CASES = [
    "HEM_PA99.nii.gz",
    "HCC_10140276_20170721.nii.gz",
    "HEM_PA97.nii.gz",
    "CYST_PA69.nii.gz",
    "CYST_PA62.nii.gz",
    "HEM_PA29.nii.gz",
    "HEP_PA28.nii.gz",
    "HEM_PA107.nii.gz",
]

HISTORICAL_SOURCES = {
    "v3_2e_world_predictor_training_or_validation": V3_2E_CASES,
    "v6_reliability_experiments": V6_CASES,
    "v7_reliability_experiments": V7_CASES,
    "checkpoint_threshold_step_selection": V3_2E_CASES,
}

METRIC_NAMES = ("dice", "foreground_iou", "precision", "recall")
COMPARISONS = {
    "confidence_uniform": ("A0_confidence_rank", "A_uniform_balanced"),
    "world_uniform": ("A1_world", "A_uniform_balanced"),
    "joint_uniform": ("A2_joint_product", "A_uniform_balanced"),
    "joint_confidence": ("A2_joint_product", "A0_confidence_rank"),
}


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V7.2 fresh held-out V7.1d confirmation")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--train-manifest", default="outputs/v7_1b_protocol_consolidation/train_patch_manifest.json")
    parser.add_argument("--fresh-manifest", default="outputs/v7_2_fresh_heldout_confirmation/fresh_eval_manifest.json")
    parser.add_argument("--output-dir", default="outputs/v7_2_fresh_heldout_confirmation")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--train-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--training-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def fresh_manifest_payload() -> dict[str, Any]:
    excluded = sorted({case for cases in HISTORICAL_SOURCES.values() for case in cases})
    return {
        "stage": "V7.2 fresh held-out reliability contribution confirmation",
        "manifest_version": 1,
        "selection_policy": "fixed before evaluation from cases absent from all audited historical experiment sets; no GT-based case selection",
        "fresh_cases": FRESH_CASES,
        "historical_excluded_cases": excluded,
        "historical_exclusion_sources": HISTORICAL_SOURCES,
        "explicitly_excluded_cases": [
            "CYST_PA62.nii.gz",
            "HEM_PA29.nii.gz",
            "HEP_PA28.nii.gz",
            "HEM_PA107.nii.gz",
        ],
        "source_audit": {
            "v3_2e_summary": "outputs/v3_2e_frozen_input_projection/v3_2e_summary.json",
            "v6_outputs": [
                "outputs/v6_0_imagined_world_reliability",
                "outputs/v6_0b_statistical_correction",
                "outputs/v6_0c_final_statistical_validation",
                "outputs/v6_1_unified_reliability_fusion",
            ],
            "v7_outputs": [
                "outputs/v7_0d_protocol_sanity",
                "outputs/v7_1b_protocol_consolidation",
                "outputs/v7_1c_class_balanced_loss_sanity",
                "outputs/v7_1d_reliability_contribution_isolation",
            ],
        },
    }


def load_or_create_fresh_manifest(path: Path) -> dict[str, Any]:
    expected = fresh_manifest_payload()
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("fresh_cases") != expected["fresh_cases"]:
            raise AssertionError("fresh_eval_manifest.json already exists with different cases")
        if manifest.get("historical_excluded_cases") != expected["historical_excluded_cases"]:
            raise AssertionError("fresh_eval_manifest.json already exists with different exclusions")
        return manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, indent=2))
    return expected


def make_case_records(paths: ProjectPaths, names: list[str]) -> list[CaseRecord]:
    cases = []
    for name in names:
        image_path = paths.image_dir / name
        label_path = paths.label_dir / name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")
        cases.append(CaseRecord(name, image_path, label_path))
    return cases


def validate_case_manifest(manifest: dict[str, Any], train_cases: list[CaseRecord], fresh_cases: list[CaseRecord]) -> dict[str, Any]:
    fresh_names = [case.case for case in fresh_cases]
    train_names = [case.case for case in train_cases]
    excluded = set(manifest["historical_excluded_cases"])
    duplicate_fresh = sorted(name for name in set(fresh_names) if fresh_names.count(name) > 1)
    if duplicate_fresh:
        raise AssertionError(f"fresh manifest has duplicate cases: {duplicate_fresh}")
    overlap_train = sorted(set(train_names) & set(fresh_names))
    overlap_history = sorted(excluded & set(fresh_names))
    if overlap_train or overlap_history:
        raise AssertionError(f"case overlap: train={overlap_train}, history={overlap_history}")
    if set(fresh_names) != set(manifest["fresh_cases"]):
        raise AssertionError("fresh case records differ from fixed fresh_eval_manifest.json")
    return {
        "fresh_cases_unique": len(fresh_names) == len(set(fresh_names)),
        "fresh_vs_adaptation_overlap": overlap_train,
        "fresh_vs_historical_overlap": overlap_history,
        "historical_excluded_cases": sorted(excluded),
        "fresh_case_count": len(fresh_names),
    }


def bootstrap_summary(values: np.ndarray, seed: int, replicates: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap requires a non-empty one-dimensional case delta array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    return {
        "n_cases": int(values.size),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "ci95_low": float(np.percentile(bootstrap_means, 2.5)),
        "ci95_high": float(np.percentile(bootstrap_means, 97.5)),
        "positive_case_fraction": float(np.mean(values > 0)),
        "positive_case_count": int(np.count_nonzero(values > 0)),
    }


def contribution_rows(
    full_rows: list[dict[str, Any]],
    comparisons: dict[str, tuple[str, str]] = COMPARISONS,
) -> list[dict[str, Any]]:
    final = {
        (row["case"], row["order"], row["variant"]): row
        for row in full_rows
        if int(row["step"]) == 20
    }
    rows = []
    case_orders = sorted({(case, order) for case, order, _ in final})
    for case, order in case_orders:
        if (case, order, "A_uniform_balanced") not in final:
            continue
        row: dict[str, Any] = {"case": case, "order": order, "step": 20}
        for name, (left, right) in comparisons.items():
            left_row = final[(case, order, left)]
            right_row = final[(case, order, right)]
            row[f"{name}_left"] = left
            row[f"{name}_right"] = right
            for metric in METRIC_NAMES:
                row[f"{name}_{metric}_delta"] = float(left_row[metric] - right_row[metric])
            row[f"{name}_dice_win"] = bool(left_row["dice"] > right_row["dice"])
        rows.append(row)
    return rows


def paired_contribution(
    rows: list[dict[str, Any]],
    bootstrap_replicates: int,
    seed: int,
    comparisons: dict[str, tuple[str, str]] = COMPARISONS,
) -> dict[str, Any]:
    by_order = {}
    for order in ("forward", "reverse"):
        order_rows = [row for row in rows if row["order"] == order]
        by_order[order] = {}
        for name in comparisons:
            deltas = np.asarray([row[f"{name}_dice_delta"] for row in order_rows], dtype=np.float64)
            by_order[order][name] = {
                "case_count": len(order_rows),
                "mean_dice_delta": float(deltas.mean()),
                "positive_case_fraction": float(np.mean(deltas > 0)),
                "positive_case_count": int(np.count_nonzero(deltas > 0)),
            }

    by_case = {}
    for name in comparisons:
        case_values = {}
        for case in sorted({row["case"] for row in rows}):
            values = [row[f"{name}_dice_delta"] for row in rows if row["case"] == case]
            case_values[case] = float(np.mean(values))
        deltas = np.asarray(list(case_values.values()), dtype=np.float64)
        by_case[name] = {
            "comparison": name,
            "left": comparisons[name][0],
            "right": comparisons[name][1],
            "case_deltas": case_values,
            "bootstrap": bootstrap_summary(deltas, seed, bootstrap_replicates),
        }
    return {
        "definition": "case-level Dice delta; forward/reverse deltas are averaged within each case for the primary paired bootstrap",
        "comparisons": comparisons,
        "by_order": by_order,
        "primary_case_averaged": by_case,
    }


def build_summary_metrics(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in curve if int(row["step"]) == 20]


def run(args: argparse.Namespace) -> None:
    if args.training_rounds != 5 or args.lora_rank != 4 or args.lora_alpha != 8.0 or args.lora_dropout != 0.0:
        raise AssertionError("V7.1d fixed LoRA/training protocol was changed")
    if args.learning_rate != 1e-4 or args.bootstrap_replicates != 10000:
        raise AssertionError("V7.2 fixed learning rate or bootstrap count was changed")
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V7.2 requires CUDA, resolved {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_fresh_manifest(Path(args.fresh_manifest))
    if not 8 <= len(manifest.get("fresh_cases", [])) <= 12:
        raise AssertionError("fresh evaluation manifest must contain 8-12 cases")
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    adaptation_cases = iter_image_cases(paths, "train", args.train_cases)
    fresh_cases = make_case_records(paths, list(manifest["fresh_cases"]))
    overlap_check = validate_case_manifest(manifest, adaptation_cases, fresh_cases)

    teacher = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    prepare_functional_seg_head(teacher, args.selected_stage)
    prompt_embedding = teacher.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(
        checkpoint_path,
        int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
        device,
        args.hidden_channels,
    )

    train_manifest_path = Path(args.train_manifest)
    train_manifest = json.loads(train_manifest_path.read_text())
    train_cache = build_cache_from_manifest(
        teacher, world_model, adaptation_cases, prompt_embedding,
        train_manifest, args, device,
    )
    eval_cache = build_evaluation_cache(
        teacher, world_model, fresh_cases, prompt_embedding, args, device,
    )
    for sample in train_cache:
        if sample["case"] not in [case.case for case in adaptation_cases]:
            raise AssertionError("train manifest contains a non-adaptation case")
    train_samples = train_cache

    full_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in fresh_cases:
        image, label, _ = read_image_and_label(case)
        full_data[case.case] = (image, label)

    world_model.to("cpu")
    teacher.network.to("cpu")
    teacher.functional_seg_head.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    base_network = copy.deepcopy(teacher.network).cpu().eval()
    base_total = sum(parameter.numel() for parameter in base_network.parameters())
    for parameter in base_network.parameters():
        parameter.requires_grad = False
    base_network.to(device)
    initial_full = evaluate_full_volume(
        teacher, base_network, fresh_cases, full_data, prompt_embedding,
        "A_init_no_adaptation", "forward", 0, args.label_value,
        args.prediction_threshold,
    )
    base_network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    variant_map = {"A_uniform_balanced": "uniform_balanced", **VARIANTS}
    reference = None
    for sample in train_samples:
        reference = next(iter(sample["weights"].values()))
        break
    if reference is None:
        raise RuntimeError("fixed train manifest produced no training samples")
    for sample in train_samples:
        sample["weights"]["uniform_balanced"] = np.ones_like(reference, dtype=np.float32)

    losses: list[dict[str, Any]] = []
    pseudo_stats: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = list(initial_full)
    parameter_stats: list[dict[str, Any]] = []
    target_names = None
    for variant, source in variant_map.items():
        print(f"[V7.2] forward {variant}", flush=True)
        l, p, f, _debug, stats = train_variant(
            variant, source, "forward", base_network, train_samples, fresh_cases,
            full_data, eval_cache, teacher, prompt_embedding, args, device,
            base_total, target_names, True,
        )
        target_names = stats["target_modules"] if target_names is None else target_names
        losses.extend(l); pseudo_stats.extend(p); full_rows.extend(f)
        parameter_stats.append({"variant": variant, "order": "forward", **stats})
        print(f"[V7.2] reverse {variant}", flush=True)
        l, p, f, _debug, stats = train_variant(
            variant, source, "reverse", base_network, train_samples, fresh_cases,
            full_data, eval_cache, teacher, prompt_embedding, args, device,
            base_total, target_names, False,
        )
        losses.extend(l); pseudo_stats.extend(p); full_rows.extend(f)
        parameter_stats.append({"variant": variant, "order": "reverse", **stats})

    curve = pool_full_volume(full_rows)
    contribution_rows_data = contribution_rows(full_rows)
    contribution = paired_contribution(
        contribution_rows_data, args.bootstrap_replicates, args.seed,
    )
    final_metrics = build_summary_metrics(curve)

    write_csv(output_dir / "training_loss.csv", losses)
    write_csv(output_dir / "pseudo_label_stats.csv", pseudo_stats)
    write_csv(output_dir / "full_volume_results.csv", full_rows)
    write_csv(output_dir / "full_volume_curve.csv", curve)
    write_csv(output_dir / "paired_contribution.csv", contribution_rows_data)
    (output_dir / "paired_contribution.json").write_text(json.dumps(contribution, indent=2))
    (output_dir / "bootstrap.json").write_text(json.dumps({
        name: payload["bootstrap"]
        for name, payload in contribution["primary_case_averaged"].items()
    }, indent=2))
    (output_dir / "parameter_stats.json").write_text(json.dumps(parameter_stats, indent=2))
    (output_dir / "train_patch_manifest.json").write_text(json.dumps(train_manifest, indent=2))

    summary = {
        "stage": "V7.2 fresh held-out V7.1d reliability contribution confirmation",
        "fresh_eval_manifest": str(Path(args.fresh_manifest)),
        "fresh_cases": manifest["fresh_cases"],
        "historical_excluded_cases": manifest["historical_excluded_cases"],
        "historical_exclusion_sources": manifest["historical_exclusion_sources"],
        "case_overlap_check": overlap_check,
        "adaptation_cases": [case.case for case in adaptation_cases],
        "train_manifest": str(train_manifest_path),
        "world_checkpoint": str(checkpoint_path),
        "selected_stage": args.selected_stage,
        "seed": args.seed,
        "resolved_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "protocol": {
            "base_protocol": "V7.1d",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "rounds": args.training_rounds,
            "updates_per_round": len(train_samples),
            "total_updates": len(train_samples) * args.training_rounds,
            "loss": "class-balanced pseudo-label BCE with existing reliability weighting",
            "strong_augmentation": "fixed gamma(+0.30)/blur(1.5) from V7.1d train manifest",
            "full_volume_inference": True,
            "early_stopping": False,
            "hyperparameter_tuning": False,
            "gt_used_for_case_selection": False,
            "action_specific_world_reliability": False,
            "world_predictor_updated": False,
            "forward_reverse_order": True,
            "formal_result_step": 20,
        },
        "variants": variant_map,
        "final_step20_full_volume": final_metrics,
        "paired_contribution": contribution,
        "bootstrap": {
            "unit": "case",
            "paired": True,
            "seed": args.seed,
            "replicates": args.bootstrap_replicates,
            "comparisons": ["joint_confidence", "world_uniform"],
            "primary_values": {
                name: payload["bootstrap"]
                for name, payload in contribution["primary_case_averaged"].items()
                if name in {"joint_confidence", "world_uniform"}
            },
        },
        "outputs": {
            name: str(output_dir / name)
            for name in (
                "fresh_eval_manifest.json", "train_patch_manifest.json",
                "full_volume_results.csv", "full_volume_curve.csv",
                "training_loss.csv", "pseudo_label_stats.csv",
                "paired_contribution.csv", "paired_contribution.json",
                "bootstrap.json", "parameter_stats.json", "summary.json",
            )
        },
        "status": "complete; fixed V7.1d protocol on fresh held-out cases",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
