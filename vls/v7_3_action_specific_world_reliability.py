"""V7.3 action-specific World reliability on the fixed V7.2 dev cases.

The training protocol is inherited from V7.1d/V7.2.  The only method change
is the World reliability map used for adaptation samples:

* world_pairwise: source/gamma/blur pairwise disagreement (legacy method)
* world_actual_action: disagreement between source and the manifest's actual
  strong action for that adaptation sample

The final confirmation cases are fixed and written to a manifest, but are
deliberately never loaded or evaluated by this development run.
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
from vls.data import CaseRecord, iter_image_cases, load_split, read_image, read_image_and_label
from vls.v2_experiment import (
    padded_image_and_slicers,
    prepare_functional_seg_head,
    resolve_device,
    state_to_intermediate_prediction,
    visual_action,
)
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, load_world_model
from vls.v6_1_unified_reliability_fusion import percentile_rank
from vls.v7_0d_protocol_sanity import (
    VARIANTS,
    build_evaluation_cache,
    resize_logits,
    set_seed,
    strong_padded_image,
)
from vls.v7_1b_protocol_consolidation import (
    evaluate_full_volume,
    pool_full_volume,
    write_csv,
)
from vls.v7_1c_class_balanced_loss_sanity import (
    train_variant,
    validate_manifest,
)
from vls.voxtell_states import VoxTellStateInterface


ACTION_VARIANTS = {
    "A_uniform_balanced": "uniform_balanced",
    "A0_confidence_rank": "confidence_rank",
    "A1_world_pairwise": "world_pairwise",
    "A2_joint_pairwise": "joint_pairwise",
    "A3_world_actual": "world_actual_action",
    "A4_joint_actual": "joint_actual",
}

METRIC_NAMES = ("dice", "foreground_iou", "precision", "recall")
COMPARISONS = {
    "world_actual_pairwise": ("A3_world_actual", "A1_world_pairwise"),
    "joint_actual_pairwise": ("A4_joint_actual", "A2_joint_pairwise"),
    "world_actual_uniform": ("A3_world_actual", "A_uniform_balanced"),
    "joint_actual_confidence": ("A4_joint_actual", "A0_confidence_rank"),
}

HISTORICAL_CASES = [
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


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V7.3 action-specific World reliability")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--train-manifest", default="outputs/v7_1b_protocol_consolidation/train_patch_manifest.json")
    parser.add_argument("--dev-manifest", default="outputs/v7_2_fresh_heldout_confirmation/fresh_eval_manifest.json")
    parser.add_argument("--final-manifest", default="outputs/v7_3_action_specific_world_reliability/final_confirmation_manifest.json")
    parser.add_argument("--output-dir", default="outputs/v7_3_action_specific_world_reliability")
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


def make_final_manifest(split_json: Path, dev_manifest_path: Path) -> dict[str, Any]:
    split = load_split(split_json)
    all_cases = sorted(set(split["train_cases"]) | set(split["test_cases"]))
    dev_manifest = json.loads(dev_manifest_path.read_text())
    dev_cases = list(dev_manifest["fresh_cases"])
    excluded = sorted(set(HISTORICAL_CASES) | set(dev_cases))
    final_cases = sorted(set(all_cases) - set(excluded))
    return {
        "stage": "V7.3 final confirmation case manifest",
        "manifest_version": 1,
        "selection_rule": "lexicographically sorted all cases in split_json after removing the audited V3.2e/V6/V7 cases and the fixed V7.2 development cases",
        "selection_uses_gt_or_metrics": False,
        "all_split_cases_count": len(all_cases),
        "historical_excluded_cases": sorted(HISTORICAL_CASES),
        "v7_2_development_cases": dev_cases,
        "excluded_cases": excluded,
        "final_confirmation_cases": final_cases,
        "final_confirmation_case_count": len(final_cases),
        "run_in_v7_3": False,
        "reason_not_run": "reserved until the action-specific World reliability method is completely frozen",
        "source_split_json": str(split_json),
        "source_v7_2_manifest": str(dev_manifest_path),
    }


def load_or_create_final_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("final_confirmation_cases") != expected["final_confirmation_cases"]:
            raise AssertionError("final_confirmation_manifest.json has changed final cases")
        if existing.get("run_in_v7_3") is not False:
            raise AssertionError("final confirmation cases must not be run in V7.3")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, indent=2))
    return expected


def make_case_records(paths: ProjectPaths, names: list[str]) -> list[CaseRecord]:
    cases = []
    for name in names:
        image_path = paths.image_dir / name
        label_path = paths.label_dir / name
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing image/label for development case {name}")
        cases.append(CaseRecord(name, image_path, label_path))
    return cases


def parse_manifest_action(value: str) -> tuple[str, float]:
    family, strength_text = value.split(":", 1)
    strength = float(strength_text)
    expected = {"gamma": 0.30, "blur": 1.5}
    if family not in expected or not np.isclose(strength, expected[family]):
        raise AssertionError(f"Unsupported fixed V7.1d train action: {value}")
    return family, strength


@torch.inference_mode()
def action_specific_reliability(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    source_state: torch.Tensor,
    source_probability: torch.Tensor,
    selected_stage: str,
    final_shape: tuple[int, int, int],
    action_family: str,
    action_strength: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    gamma_state = world_model(source_state, action=visual_action("gamma", 0.30, device))
    blur_state = world_model(source_state, action=visual_action("blur", 1.5, device))
    probabilities = [
        torch.sigmoid(resize_logits(
            state_to_intermediate_prediction(interface, selected_stage, state),
            final_shape,
        ))
        for state in (source_state, gamma_state, blur_state)
    ]
    stack = torch.cat(probabilities, dim=0)
    pairwise = torch.stack([
        (stack[0] - stack[1]).abs(),
        (stack[0] - stack[2]).abs(),
        (stack[1] - stack[2]).abs(),
    ], dim=0).mean(dim=0)
    actual_index = 1 if action_family == "gamma" else 2
    actual = (stack[0] - stack[actual_index]).abs()
    confidence = torch.maximum(source_probability, 1.0 - source_probability)
    confidence_rank = percentile_rank(confidence.flatten().cpu().numpy()).astype(np.float32)
    pairwise_rank = percentile_rank(pairwise.flatten().cpu().numpy()).astype(np.float32)
    actual_rank = percentile_rank(actual.flatten().cpu().numpy()).astype(np.float32)
    world_pairwise = 1.0 - pairwise_rank
    world_actual = 1.0 - actual_rank
    weights = {
        "confidence_rank": confidence_rank,
        "world_pairwise": world_pairwise,
        "joint_pairwise": (confidence_rank * world_pairwise).astype(np.float32),
        "world_actual_action": world_actual,
        "joint_actual": (confidence_rank * world_actual).astype(np.float32),
    }
    diagnostics = {
        "action_family": action_family,
        "action_strength": float(action_strength),
        "pairwise_disagreement_mean": float(pairwise.mean().cpu()),
        "actual_disagreement_mean": float(actual.mean().cpu()),
        "pairwise_world_mean": float(world_pairwise.mean()),
        "actual_world_mean": float(world_actual.mean()),
        "mean_abs_world_map_delta": float(np.mean(np.abs(world_actual - world_pairwise))),
        "mean_abs_joint_map_delta": float(np.mean(np.abs(weights["joint_actual"] - weights["joint_pairwise"]))),
        "actual_disagreement_voxels": int(actual.numel()),
    }
    return weights, diagnostics


def build_action_train_cache(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    cases: list[CaseRecord],
    prompt_embedding: torch.Tensor,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    validate_manifest(manifest, cases, args, interface.predictor)
    cache = []
    for case_index, case in enumerate(cases):
        record = manifest["records"][case_index]
        action_family, action_strength = parse_manifest_action(record["augmentation"])
        image, _ = read_image(case)
        original_padded, _ = padded_image_and_slicers(interface.predictor, image)
        spatial_slicer = (
            slice(None),
            *(slice(int(start), int(stop), None)
              for start, stop in zip(record["slicer_start"], record["slicer_stop"], strict=True)),
        )
        strong_padded = strong_padded_image(
            interface, image, action_family, action_strength, original_padded,
        )
        patch = torch.clone(original_padded[spatial_slicer][None], memory_format=torch.contiguous_format)
        strong_patch = torch.clone(strong_padded[spatial_slicer][None], memory_format=torch.contiguous_format)
        result = interface.forward_with_states(patch, prompt_embedding)
        source_state = result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
        source_probability = torch.sigmoid(result["final_prediction"][:, 0:1].detach().float().to(device))
        final_shape = tuple(int(size) for size in source_probability.shape[-3:])
        weights, diagnostics = action_specific_reliability(
            interface, world_model, source_state, source_probability,
            args.selected_stage, final_shape, action_family, action_strength, device,
        )
        pseudo = (source_probability > args.prediction_threshold).float()
        cache.append({
            "case": case.case,
            "patch_index": int(record["patch_index"]),
            "patch_kind": record["patch_kind"],
            "slicer": spatial_slicer,
            "image": strong_patch.detach().cpu(),
            "embedding": prompt_embedding.detach().cpu(),
            "pseudo": pseudo.detach().cpu(),
            "weights": weights,
            "has_foreground": bool(torch.count_nonzero(pseudo)),
            "augmentation": record["augmentation"],
            "action_family": action_family,
            "action_strength": action_strength,
            "action_diagnostics": diagnostics,
            "case_index": case_index,
        })
    return cache


def array_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q05": float(np.percentile(values, 5)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "q75": float(np.percentile(values, 75)),
        "q95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def action_reliability_tables(train_cache: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sources = (
        "confidence_rank", "world_pairwise", "joint_pairwise",
        "world_actual_action", "joint_actual",
    )
    by_sample = []
    by_action_values: dict[tuple[str, str], list[np.ndarray]] = {}
    diagnostics = []
    for sample in train_cache:
        action = sample["action_family"]
        for source in sources:
            stats = array_stats(sample["weights"][source])
            by_sample.append({
                "case": sample["case"],
                "augmentation": sample["augmentation"],
                "action_family": action,
                "reliability_source": source,
                **stats,
            })
            by_action_values.setdefault((action, source), []).append(sample["weights"][source])
        diagnostics.append({"case": sample["case"], **sample["action_diagnostics"]})

    by_action = []
    for (action, source), arrays in sorted(by_action_values.items()):
        by_action.append({
            "action_family": action,
            "reliability_source": source,
            **array_stats(np.concatenate([array.reshape(-1) for array in arrays])),
        })
    action_names = sorted({row["action_family"] for row in diagnostics})
    diagnostic_summary = {}
    for action in action_names:
        rows = [row for row in diagnostics if row["action_family"] == action]
        diagnostic_summary[action] = {
            "sample_count": len(rows),
            "mean_abs_world_map_delta": float(np.mean([row["mean_abs_world_map_delta"] for row in rows])),
            "mean_abs_joint_map_delta": float(np.mean([row["mean_abs_joint_map_delta"] for row in rows])),
            "pairwise_world_mean": float(np.mean([row["pairwise_world_mean"] for row in rows])),
            "actual_world_mean": float(np.mean([row["actual_world_mean"] for row in rows])),
        }
    diagnostic_summary["all_actual_maps_differ_from_pairwise"] = all(
        row["mean_abs_world_map_delta"] > 0.0 for row in diagnostics
    )
    return by_sample, by_action, {"per_sample": diagnostics, "by_action": diagnostic_summary}


def bootstrap_summary(values: np.ndarray, seed: int, replicates: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    return {
        "n_cases": int(values.size),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "positive_case_fraction": float(np.mean(values > 0)),
        "positive_case_count": int(np.count_nonzero(values > 0)),
    }


def contribution_rows(full_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final = {
        (row["case"], row["order"], row["variant"]): row
        for row in full_rows if int(row["step"]) == 20
    }
    rows = []
    for case, order in sorted({(key[0], key[1]) for key in final}):
        row = {"case": case, "order": order, "step": 20}
        for name, (left, right) in COMPARISONS.items():
            left_row = final[(case, order, left)]
            right_row = final[(case, order, right)]
            row[f"{name}_left"] = left
            row[f"{name}_right"] = right
            for metric in METRIC_NAMES:
                row[f"{name}_{metric}_delta"] = float(left_row[metric] - right_row[metric])
            row[f"{name}_dice_win"] = bool(left_row["dice"] > right_row["dice"])
        rows.append(row)
    return rows


def paired_contribution(rows: list[dict[str, Any]], seed: int, replicates: int) -> dict[str, Any]:
    by_order = {}
    for order in ("forward", "reverse"):
        order_rows = [row for row in rows if row["order"] == order]
        by_order[order] = {}
        for name in COMPARISONS:
            values = np.asarray([row[f"{name}_dice_delta"] for row in order_rows])
            by_order[order][name] = {
                "case_count": len(order_rows),
                "mean_delta": float(values.mean()),
                "positive_case_fraction": float(np.mean(values > 0)),
                "positive_case_count": int(np.count_nonzero(values > 0)),
            }
    primary = {}
    for name, (left, right) in COMPARISONS.items():
        case_deltas = {}
        for case in sorted({row["case"] for row in rows}):
            values = [row[f"{name}_dice_delta"] for row in rows if row["case"] == case]
            case_deltas[case] = float(np.mean(values))
        values = np.asarray(list(case_deltas.values()), dtype=np.float64)
        primary[name] = {
            "left": left,
            "right": right,
            "case_deltas": case_deltas,
            "bootstrap": bootstrap_summary(values, seed, replicates),
        }
    return {"comparisons": COMPARISONS, "by_order": by_order, "primary_case_averaged": primary}


def run(args: argparse.Namespace) -> None:
    if (
        args.training_rounds != 5 or args.lora_rank != 4
        or args.lora_alpha != 8.0 or args.lora_dropout != 0.0
        or args.learning_rate != 1e-4 or args.bootstrap_replicates != 10000
    ):
        raise AssertionError("V7.3 fixed V7.1d/V7.2 training protocol was changed")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_manifest_path = Path(args.dev_manifest)
    final_manifest = load_or_create_final_manifest(
        Path(args.final_manifest), make_final_manifest(Path(args.split_json), dev_manifest_path),
    )
    if final_manifest.get("run_in_v7_3") is not False:
        raise AssertionError("V7.3 must not run final confirmation cases")

    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V7.3 requires CUDA, resolved {device}")
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    dev_manifest = json.loads(dev_manifest_path.read_text())
    adaptation_cases = iter_image_cases(paths, "train", args.train_cases)
    development_cases = make_case_records(paths, list(dev_manifest["fresh_cases"]))
    adaptation_names = [case.case for case in adaptation_cases]
    development_names = [case.case for case in development_cases]
    if set(adaptation_names) & set(development_names):
        raise AssertionError("adaptation/development case overlap")
    if set(final_manifest["final_confirmation_cases"]) & (set(adaptation_names) | set(development_names)):
        raise AssertionError("final confirmation case overlaps a V7.3 used case")

    teacher = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    prepare_functional_seg_head(teacher, args.selected_stage)
    prompt_embedding = teacher.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(
        checkpoint_path, int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
        device, args.hidden_channels,
    )

    train_manifest_path = Path(args.train_manifest)
    train_manifest = json.loads(train_manifest_path.read_text())
    train_cache = build_action_train_cache(
        teacher, world_model, adaptation_cases, prompt_embedding,
        train_manifest, args, device,
    )
    eval_cache = build_evaluation_cache(
        teacher, world_model, development_cases, prompt_embedding, args, device,
    )
    full_data = {}
    for case in development_cases:
        image, label, _ = read_image_and_label(case)
        full_data[case.case] = (image, label)

    by_sample, by_action, action_diagnostics = action_reliability_tables(train_cache)
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
        teacher, base_network, development_cases, full_data, prompt_embedding,
        "A_init_no_adaptation", "forward", 0, args.label_value, args.prediction_threshold,
    )
    base_network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reference = train_cache[0]["weights"]["confidence_rank"]
    for sample in train_cache:
        sample["weights"]["uniform_balanced"] = np.ones_like(reference, dtype=np.float32)

    losses = []
    pseudo_stats = []
    full_rows = list(initial_full)
    parameter_stats = []
    target_names = None
    for variant, source in ACTION_VARIANTS.items():
        print(f"[V7.3] forward {variant}", flush=True)
        l, p, f, _debug, stats = train_variant(
            variant, source, "forward", base_network, train_cache, development_cases,
            full_data, eval_cache, teacher, prompt_embedding, args, device,
            base_total, target_names, True,
        )
        target_names = stats["target_modules"] if target_names is None else target_names
        losses.extend(l); pseudo_stats.extend(p); full_rows.extend(f)
        parameter_stats.append({"variant": variant, "order": "forward", **stats})
        print(f"[V7.3] reverse {variant}", flush=True)
        l, p, f, _debug, stats = train_variant(
            variant, source, "reverse", base_network, train_cache, development_cases,
            full_data, eval_cache, teacher, prompt_embedding, args, device,
            base_total, target_names, False,
        )
        losses.extend(l); pseudo_stats.extend(p); full_rows.extend(f)
        parameter_stats.append({"variant": variant, "order": "reverse", **stats})

    curve = pool_full_volume(full_rows)
    contribution_rows_data = contribution_rows(full_rows)
    contribution = paired_contribution(contribution_rows_data, args.seed, args.bootstrap_replicates)
    write_csv(output_dir / "training_loss.csv", losses)
    write_csv(output_dir / "pseudo_label_stats.csv", pseudo_stats)
    write_csv(output_dir / "full_volume_results.csv", full_rows)
    write_csv(output_dir / "full_volume_curve.csv", curve)
    write_csv(output_dir / "paired_contribution.csv", contribution_rows_data)
    write_csv(output_dir / "action_reliability_by_sample.csv", by_sample)
    write_csv(output_dir / "action_reliability_distribution.csv", by_action)
    write_csv(output_dir / "action_reliability_diagnostics.csv", action_diagnostics["per_sample"])
    (output_dir / "paired_contribution.json").write_text(json.dumps(contribution, indent=2))
    (output_dir / "bootstrap.json").write_text(json.dumps({
        name: value["bootstrap"] for name, value in contribution["primary_case_averaged"].items()
    }, indent=2))
    (output_dir / "parameter_stats.json").write_text(json.dumps(parameter_stats, indent=2))
    (output_dir / "final_confirmation_manifest.json").write_text(json.dumps(final_manifest, indent=2))

    summary = {
        "stage": "V7.3 action-specific World reliability",
        "development_cases": development_names,
        "adaptation_cases": adaptation_names,
        "final_confirmation_cases": final_manifest["final_confirmation_cases"],
        "final_confirmation_cases_used_this_run": [],
        "final_confirmation_manifest": str(Path(args.final_manifest)),
        "train_manifest": str(train_manifest_path),
        "world_checkpoint": str(checkpoint_path),
        "selected_stage": args.selected_stage,
        "case_overlap_check": {
            "adaptation_vs_development": sorted(set(adaptation_names) & set(development_names)),
            "final_vs_adaptation_or_development": sorted(set(final_manifest["final_confirmation_cases"]) & (set(adaptation_names) | set(development_names))),
            "historical_excluded_vs_development": sorted(set(HISTORICAL_CASES) & set(development_names)),
        },
        "seed": args.seed,
        "resolved_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "protocol": {
            "base_protocol": "V7.2/V7.1d",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "rounds": args.training_rounds,
            "updates_per_round": len(train_cache),
            "total_updates": len(train_cache) * args.training_rounds,
            "pseudo_label": "unchanged fixed teacher pseudo-label",
            "loss": "unchanged class-balanced pseudo-label BCE",
            "augmentation": "unchanged fixed gamma(+0.30)/blur(1.5) train manifest",
            "forward_reverse_order": True,
            "formal_result_step": 20,
            "observation_curve_steps": [4, 8, 12, 16, 20],
            "early_stopping": False,
            "hyperparameter_tuning": False,
            "gt_used_for_case_selection": False,
            "world_predictor_updated": False,
            "only_method_change": "legacy pairwise world stability -> actual-action world stability",
            "action_specific_world_reliability": True,
        },
        "variants": ACTION_VARIANTS,
        "final_step20_full_volume": [row for row in curve if int(row["step"]) == 20],
        "paired_contribution": contribution,
        "action_reliability_diagnostics": action_diagnostics,
        "bootstrap": {
            "unit": "case",
            "paired": True,
            "seed": args.seed,
            "replicates": args.bootstrap_replicates,
            "comparisons": list(COMPARISONS),
            "results": {
                name: value["bootstrap"]
                for name, value in contribution["primary_case_averaged"].items()
            },
        },
        "outputs": {
            name: str(output_dir / name)
            for name in (
                "final_confirmation_manifest.json", "full_volume_results.csv",
                "full_volume_curve.csv", "training_loss.csv", "pseudo_label_stats.csv",
                "paired_contribution.csv", "paired_contribution.json", "bootstrap.json",
                "action_reliability_by_sample.csv", "action_reliability_distribution.csv",
                "action_reliability_diagnostics.csv", "parameter_stats.json", "summary.json",
            )
        },
        "status": "complete; V7.3 actual-action World reliability on V7.2 development cases only",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
