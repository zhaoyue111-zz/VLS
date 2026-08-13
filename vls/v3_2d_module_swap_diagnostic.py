from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases
from vls.v2_experiment import (
    build_dataset as build_visual_dataset,
    prepare_functional_seg_head,
    resolve_device,
    v2_pass_summary,
)
from vls.v3_2a_unified_experiment import (
    build_language_dataset,
    collect_language_eval,
    set_seed,
)
from vls.v3_2b_joint_experiment import (
    collect_visual_eval,
    language_joint_pass_summary,
    make_model,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D


SWAP_GROUPS = {
    "action_mlp": ("action_mlp.",),
    "input_projection": ("input_projection.",),
    "blocks": ("blocks.",),
    "output_projection": ("output_projection.",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3.2d no-training module-swap diagnostic.")
    paths = ProjectPaths()
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--base-checkpoint", default="outputs/v3_2c_lr5e-5/unified_world_predictor_step200.pt")
    parser.add_argument("--v2-checkpoint", default="outputs/v2_final/world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v3_2d_module_swap")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--dev-cases", type=int, default=8)
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gamma-strengths", nargs="+", type=float, default=[-0.3, -0.15, 0.15, 0.3])
    parser.add_argument("--blur-sigmas", nargs="+", type=float, default=[0.5, 1.5])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_unified_model(
    base_checkpoint: Path,
    in_channels: int,
    text_delta_dim: int,
    hidden_channels: int,
    device: torch.device,
) -> VisualWorldPredictor3D:
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    model = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        action_dim=3,
        num_blocks=2,
        use_action=True,
        text_delta_dim=text_delta_dim,
        use_language=True,
        allow_unconditioned=True,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected base checkpoint load: missing={missing}, unexpected={unexpected}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def apply_single_swap(
    base_model: nn.Module,
    v2_checkpoint: Path,
    swap_name: str,
) -> tuple[nn.Module, list[str]]:
    model = deepcopy(base_model)
    v2 = torch.load(v2_checkpoint, map_location="cpu", weights_only=False)
    base_state = model.state_dict()
    v2_state = v2["conditioned_state_dict"]
    prefixes = SWAP_GROUPS[swap_name]
    replaced = []
    for key in base_state:
        if key.startswith(prefixes):
            if key not in v2_state:
                raise KeyError(f"V2 checkpoint is missing swap key {key}")
            base_state[key] = v2_state[key].detach().clone()
            replaced.append(key)
    if not replaced:
        raise RuntimeError(f"No parameters matched swap group {swap_name}")
    model.load_state_dict(base_state, strict=True)
    model.eval()
    return model, replaced


def rows_at_step(rows: list[dict[str, Any]], step: int = 0) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["step"]) == step]


def compact_group_metrics(rows: list[dict[str, Any]], groups: list[str]) -> dict[str, Any]:
    result = {}
    for group in groups:
        result[group] = {}
        for row in rows:
            if row["split"] == "val" and row["group"] == group:
                result[group].setdefault(str(row["group_value"]), {})[str(row["model"])] = {
                    "state_normalized_mse": float(row["state_normalized_mse"]),
                    "mask_logit_normalized_mse": float(row["mask_logit_normalized_mse"]),
                }
    return result


def weak_strength_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weak = {}
    for strength in ["+0.15", "+0.30", "+0.50"]:
        values = {
            str(row["model"]): {
                "state_normalized_mse": float(row["state_normalized_mse"]),
                "mask_logit_normalized_mse": float(row["mask_logit_normalized_mse"]),
            }
            for row in rows
            if row["split"] == "val" and row["group"] == "strengths" and row["group_value"] == strength
        }
        weak[strength] = values
    return weak


def evaluate_variant(
    name: str,
    model: nn.Module,
    language_val: dict[str, Any],
    visual_val: dict[str, Any],
    v2_agnostic: nn.Module,
    interface: VoxTellStateInterface,
    selected_stage: str,
    batch_size: int,
    replaced_keys: list[str],
) -> dict[str, Any]:
    language_rows, language_groups = collect_language_eval(
        0, language_val, language_val, model, interface, selected_stage, batch_size,
    )
    visual_rows, visual_groups = collect_visual_eval(
        0, "val", visual_val, model, v2_agnostic, interface, selected_stage, batch_size,
    )
    language_rows = rows_at_step(language_rows)
    language_groups = rows_at_step(language_groups)
    visual_rows = rows_at_step(visual_rows)
    visual_groups = rows_at_step(visual_groups)
    language_pass = language_joint_pass_summary(language_rows, language_groups)
    visual_pass = v2_pass_summary(visual_rows, visual_groups)
    return {
        "name": name,
        "replaced_keys": replaced_keys,
        "language_pass": language_pass,
        "visual_pass": visual_pass,
        "joint_passed": bool(language_pass["passed"] and visual_pass["passed"]),
        "language_overall": compact_group_metrics(language_rows, ["overall_micro", "overall_macro"]),
        "language_directions": compact_group_metrics(language_groups, ["directions"]),
        "language_cases": compact_group_metrics(language_groups, ["case_names"]),
        "visual_overall": compact_group_metrics(visual_rows, ["overall_micro", "overall_macro"]),
        "visual_families": compact_group_metrics(visual_groups, ["action_families"]),
        "visual_strengths": compact_group_metrics(visual_groups, ["strengths"]),
        "visual_cases": compact_group_metrics(visual_groups, ["case_names"]),
        "weak_strengths": weak_strength_metrics(visual_groups),
        "language_rows": language_rows,
        "language_group_rows": language_groups,
        "visual_rows": visual_rows,
        "visual_group_rows": visual_groups,
    }


def language_metric_map(result: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    metrics = {}
    for row in result["language_rows"] + result["language_group_rows"]:
        metrics[(str(row["group"]), str(row["group_value"]), str(row["model"]))] = float(row["state_normalized_mse"])
    return metrics


def language_diff(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    a = language_metric_map(base)
    b = language_metric_map(other)
    diffs = [abs(b[key] - a[key]) for key in a.keys() & b.keys()]
    return {
        "max_abs_state_metric_difference": max(diffs) if diffs else None,
        "mean_abs_state_metric_difference": sum(diffs) / len(diffs) if diffs else None,
    }


def write_variant_csv(output_dir: Path, result: dict[str, Any]) -> None:
    fields = ["step", "split", "group", "group_value", "model", "state_normalized_mse", "mask_logit_normalized_mse"]
    name = result["name"]
    for key in ["language_rows", "language_group_rows", "visual_rows", "visual_group_rows"]:
        path = output_dir / f"{name}_{key}.csv"
        rows = result[key]
        with path.open("w", newline="") as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


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
    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    train_cases = iter_cases(paths, split="train", limit=args.dev_cases)
    val_cases = iter_cases(paths, split="test", limit=args.val_cases)
    language_val = build_language_dataset(
        interface, val_cases, args.selected_stage, args.patches_per_case,
        args.foreground_patches_per_case, args.foreground_candidate_patches, args.foreground_threshold,
    )
    visual_values = {"gamma": list(args.gamma_strengths), "blur": list(args.blur_sigmas)}
    visual_val = build_visual_dataset(
        interface, val_cases, DEFAULT_PROMPTS, ["gamma", "blur"], visual_values,
        args.selected_stage, args.patches_per_case, args.foreground_patches_per_case,
        args.foreground_candidate_patches, args.foreground_threshold,
    )
    prepare_functional_seg_head(interface, args.selected_stage)
    base_checkpoint = Path(args.base_checkpoint)
    v2_checkpoint = Path(args.v2_checkpoint)
    base_meta = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    base_model = load_unified_model(
        base_checkpoint, int(language_val["states"].shape[1]), int(base_meta["text_delta_dim"]),
        args.hidden_channels, device,
    )
    v2_agnostic = make_model(
        v2_checkpoint, int(language_val["states"].shape[1]), int(base_meta["text_delta_dim"]),
        args.hidden_channels, device, state_key="agnostic_state_dict",
    )

    base_result = evaluate_variant(
        "base_v3_2c_step200", base_model, language_val, visual_val, v2_agnostic,
        interface, args.selected_stage, args.batch_size, [],
    )
    results = {base_result["name"]: base_result}
    action_model, action_keys = apply_single_swap(base_model, v2_checkpoint, "action_mlp")
    action_result = evaluate_variant(
        "swap_action_mlp", action_model, language_val, visual_val, v2_agnostic,
        interface, args.selected_stage, args.batch_size, action_keys,
    )
    action_result["language_diff_vs_base"] = language_diff(base_result, action_result)
    results[action_result["name"]] = action_result

    action_recovered = (
        action_result["visual_pass"]["strength_win_count"] > 3
        and action_result["joint_passed"]
    )
    additional_swaps = [] if action_recovered else ["input_projection", "blocks", "output_projection"]
    for swap_name in additional_swaps:
        model, replaced_keys = apply_single_swap(base_model, v2_checkpoint, swap_name)
        result = evaluate_variant(
            f"swap_{swap_name}", model, language_val, visual_val, v2_agnostic,
            interface, args.selected_stage, args.batch_size, replaced_keys,
        )
        result["language_diff_vs_base"] = language_diff(base_result, result)
        results[result["name"]] = result

    for result in results.values():
        write_variant_csv(output_dir, result)
    summary = {
        "args": vars(args),
        "train_cases": [case.case for case in train_cases],
        "val_cases": [case.case for case in val_cases],
        "base_checkpoint": str(base_checkpoint),
        "v2_checkpoint": str(v2_checkpoint),
        "training_performed": False,
        "action_mlp_swap_recovered_joint": action_recovered,
        "variants_evaluated": list(results),
        "results": results,
        "scope_note": "V3.2d diagnostic only: no optimizer/backward; each swap independently replaces one module from V2 conditioned checkpoint while all other V3.2c step200 parameters remain unchanged.",
    }
    (output_dir / "v3_2d_summary.json").write_text(json.dumps(summary, indent=2))
    compact = {
        name: {
            "joint_passed": result["joint_passed"],
            "language_passed": result["language_pass"]["passed"],
            "visual_passed": result["visual_pass"]["passed"],
            "visual_strength_win_count": result["visual_pass"]["strength_win_count"],
            "visual_case_win_count": result["visual_pass"]["case_win_count"],
            "language_diff_vs_base": result.get("language_diff_vs_base"),
        }
        for name, result in results.items()
    }
    print(json.dumps({"summary_path": str(output_dir / "v3_2d_summary.json"), "variants": compact}, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
