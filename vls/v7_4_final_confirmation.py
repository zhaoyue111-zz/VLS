"""V7.4 final confirmation on the immutable V7.3 final-case manifest.

This runner changes no training or reliability implementation.  It reuses the
V7.3 action-specific cache only for the frozen pairwise maps and evaluates the
four pre-registered variants on the 18 cases listed in the final manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vls.config import ProjectPaths
from vls.data import CaseRecord, iter_image_cases, read_image_and_label
from vls.v2_experiment import prepare_functional_seg_head, resolve_device
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, load_world_model
from vls.v7_0d_protocol_sanity import build_evaluation_cache, set_seed
from vls.v7_1b_protocol_consolidation import evaluate_full_volume, pool_full_volume, write_csv
from vls.v7_1c_class_balanced_loss_sanity import train_variant, validate_manifest
from vls.v7_2_fresh_heldout_confirmation import (
    contribution_rows,
    paired_contribution,
)
from vls.v7_3_action_specific_world_reliability import build_action_train_cache
from vls.voxtell_states import VoxTellStateInterface


VARIANTS = {
    "A_uniform_balanced": "uniform_balanced",
    "A0_confidence_rank": "confidence_rank",
    "A1_world_pairwise": "world_pairwise",
    "A2_joint_pairwise": "joint_pairwise",
}
COMPARISONS = {
    "confidence_uniform": ("A0_confidence_rank", "A_uniform_balanced"),
    "world_uniform": ("A1_world_pairwise", "A_uniform_balanced"),
    "joint_uniform": ("A2_joint_pairwise", "A_uniform_balanced"),
    "joint_confidence": ("A2_joint_pairwise", "A0_confidence_rank"),
}
FINAL_CASE_COUNT = 18
FORMAL_STEP = 20
OBSERVATION_STEPS = (4, 8, 12, 16, 20)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V7.4 final confirmation")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt",
    )
    parser.add_argument(
        "--train-manifest",
        default="outputs/v7_1b_protocol_consolidation/train_patch_manifest.json",
    )
    parser.add_argument(
        "--final-manifest",
        default="outputs/v7_3_action_specific_world_reliability/final_confirmation_manifest.json",
    )
    parser.add_argument(
        "--output-dir", default="outputs/v7_4_final_confirmation",
    )
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
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_final_cases(paths: ProjectPaths, manifest_path: Path) -> tuple[dict[str, Any], list[CaseRecord]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing immutable final confirmation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    names = list(manifest.get("final_confirmation_cases", []))
    if manifest.get("final_confirmation_case_count") != FINAL_CASE_COUNT:
        raise AssertionError("final confirmation manifest must contain exactly 18 cases")
    if len(names) != FINAL_CASE_COUNT or len(set(names)) != FINAL_CASE_COUNT:
        raise AssertionError("final confirmation manifest has invalid case count or duplicates")
    if manifest.get("selection_uses_gt_or_metrics") is not False:
        raise AssertionError("final manifest is not marked as independent of GT/metrics")
    if manifest.get("run_in_v7_3") is not False:
        raise AssertionError("final cases must be unused by V7.3")

    cases: list[CaseRecord] = []
    for name in names:
        image_path = paths.image_dir / name
        label_path = paths.label_dir / name
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing final confirmation image/label for {name}")
        cases.append(CaseRecord(name, image_path, label_path))
    return manifest, cases


def assert_fixed_protocol(args: argparse.Namespace) -> None:
    if (
        args.training_rounds != 5
        or args.lora_rank != 4
        or args.lora_alpha != 8.0
        or args.lora_dropout != 0.0
        or args.learning_rate != 1e-4
        or args.bootstrap_replicates != 10000
    ):
        raise AssertionError("V7.4 protocol is frozen and cannot be changed")


def run(args: argparse.Namespace) -> None:
    assert_fixed_protocol(args)
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V7.4 requires CUDA, resolved {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    final_manifest_path = Path(args.final_manifest)
    final_manifest, final_cases = load_final_cases(paths, final_manifest_path)
    adaptation_cases = iter_image_cases(paths, "train", args.train_cases)
    adaptation_names = [case.case for case in adaptation_cases]
    final_names = [case.case for case in final_cases]
    overlap = sorted(set(adaptation_names) & set(final_names))
    if overlap:
        raise AssertionError(f"adaptation/final case overlap: {overlap}")

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
    train_cache = build_action_train_cache(
        teacher, world_model, adaptation_cases, prompt_embedding,
        train_manifest, args, device,
    )
    for sample in train_cache:
        if sample["case"] not in adaptation_names:
            raise AssertionError("train manifest contains a non-adaptation case")
    if len(train_cache) != len(adaptation_cases):
        raise AssertionError("expected one fixed foreground train patch per adaptation case")

    reference = train_cache[0]["weights"]["confidence_rank"]
    for sample in train_cache:
        sample["weights"]["uniform_balanced"] = np.ones_like(reference, dtype=np.float32)

    eval_cache = build_evaluation_cache(
        teacher, world_model, final_cases, prompt_embedding, args, device,
    )
    full_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in final_cases:
        image, label, _ = read_image_and_label(case)
        full_data[case.case] = (image, label)

    world_model.to("cpu")
    teacher.network.to("cpu")
    teacher.functional_seg_head.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    base_network = copy.deepcopy(teacher.network).cpu().eval()
    base_total = sum(parameter.numel() for parameter in base_network.parameters())
    for parameter in base_network.parameters():
        parameter.requires_grad = False
    base_network.to(device)

    initial_full = evaluate_full_volume(
        teacher, base_network, final_cases, full_data, prompt_embedding,
        "A_init_no_adaptation", "forward", 0, args.label_value,
        args.prediction_threshold,
    )
    base_network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    losses: list[dict[str, Any]] = []
    pseudo_stats: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = list(initial_full)
    parameter_stats: list[dict[str, Any]] = []
    target_names: list[str] | None = None
    for variant, source in VARIANTS.items():
        print(f"[V7.4] forward {variant}", flush=True)
        l, p, f, _debug, stats = train_variant(
            variant, source, "forward", base_network, train_cache, final_cases,
            full_data, eval_cache, teacher, prompt_embedding, args, device,
            base_total, target_names, True,
        )
        target_names = stats["target_modules"] if target_names is None else target_names
        losses.extend(l)
        pseudo_stats.extend(p)
        full_rows.extend(f)
        parameter_stats.append({"variant": variant, "order": "forward", **stats})

        print(f"[V7.4] reverse {variant}", flush=True)
        l, p, f, _debug, stats = train_variant(
            variant, source, "reverse", base_network, train_cache, final_cases,
            full_data, eval_cache, teacher, prompt_embedding, args, device,
            base_total, target_names, False,
        )
        losses.extend(l)
        pseudo_stats.extend(p)
        full_rows.extend(f)
        parameter_stats.append({"variant": variant, "order": "reverse", **stats})

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = float(torch.cuda.max_memory_allocated(device) / 1024**2)
        peak_reserved = float(torch.cuda.max_memory_reserved(device) / 1024**2)
    else:
        peak_allocated = peak_reserved = 0.0

    curve = pool_full_volume(full_rows)
    contribution_data = contribution_rows(full_rows, comparisons=COMPARISONS)
    contribution = paired_contribution(
        contribution_data, args.bootstrap_replicates, args.seed,
        comparisons=COMPARISONS,
    )

    write_csv(output_dir / "full_volume_results.csv", full_rows)
    write_csv(output_dir / "full_volume_curve.csv", curve)
    write_csv(output_dir / "paired_contribution.csv", contribution_data)
    write_csv(output_dir / "training_loss.csv", losses)
    write_csv(output_dir / "pseudo_label_stats.csv", pseudo_stats)
    (output_dir / "paired_contribution.json").write_text(json.dumps(contribution, indent=2))
    (output_dir / "bootstrap.json").write_text(json.dumps({
        name: payload["bootstrap"]
        for name, payload in contribution["primary_case_averaged"].items()
    }, indent=2))
    (output_dir / "parameter_stats.json").write_text(json.dumps(parameter_stats, indent=2))
    (output_dir / "final_confirmation_manifest.json").write_text(
        json.dumps(final_manifest, indent=2),
    )

    final_step20 = [row for row in curve if int(row["step"]) == FORMAL_STEP]
    summary = {
        "stage": "V7.4 final confirmation",
        "final_confirmation_manifest_source": str(final_manifest_path),
        "final_confirmation_cases": final_names,
        "final_confirmation_case_count": len(final_names),
        "final_confirmation_cases_used_this_run": final_names,
        "adaptation_cases": adaptation_names,
        "case_overlap": overlap,
        "case_overlap_count": len(overlap),
        "world_checkpoint": str(checkpoint_path),
        "train_manifest": str(train_manifest_path),
        "selected_stage": args.selected_stage,
        "seed": args.seed,
        "resolved_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_allocated_mb": peak_allocated,
        "peak_cuda_reserved_mb": peak_reserved,
        "protocol": {
            "frozen_from": "V7.3/V7.2",
            "same_world_checkpoint": True,
            "same_adaptation_cases": True,
            "same_train_manifest": True,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "rounds": args.training_rounds,
            "updates_per_round": len(train_cache),
            "total_updates": len(train_cache) * args.training_rounds,
            "loss": "class-balanced pseudo-label BCE, unchanged",
            "augmentation": "fixed gamma(+0.30)/blur(1.5), unchanged",
            "prediction_threshold": args.prediction_threshold,
            "full_volume_inference": True,
            "formal_result_step": FORMAL_STEP,
            "observation_curve_steps": list(OBSERVATION_STEPS),
            "forward_reverse_order": True,
            "early_stopping": False,
            "hyperparameter_tuning": False,
            "gt_based_case_selection": False,
            "reliability_modified": False,
            "actual_action_variants_run": False,
        },
        "variants": VARIANTS,
        "comparisons": COMPARISONS,
        "final_step20_full_volume": final_step20,
        "paired_contribution": contribution,
        "bootstrap": {
            "unit": "case, averaging forward/reverse within case",
            "seed": args.seed,
            "replicates": args.bootstrap_replicates,
            "results": {
                name: payload["bootstrap"]
                for name, payload in contribution["primary_case_averaged"].items()
            },
        },
        "outputs": {
            name: str(output_dir / name)
            for name in (
                "final_confirmation_manifest.json", "full_volume_results.csv",
                "full_volume_curve.csv", "paired_contribution.csv",
                "paired_contribution.json", "bootstrap.json", "training_loss.csv",
                "pseudo_label_stats.csv", "parameter_stats.json", "summary.json",
            )
        },
        "status": "complete; fixed step20 final confirmation; no method changes",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
