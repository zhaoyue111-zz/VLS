"""V8.3b stage-matched real-augmentation oracle reliability diagnostic.

This diagnostic isolates World Predictor transition fidelity.  Real gamma and
blur images are passed through VoxTell, and their selected decoder states are
converted with the same intermediate functional segmentation head used by
V8.2.  No World Predictor is loaded and no model is trained.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import (
    padded_visual_action_and_slicers,
    prepare_functional_seg_head,
    resolve_device,
    select_patch_slicers,
    state_to_intermediate_prediction,
)
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, pad_label_like_image
from vls.v7_0d_protocol_sanity import set_seed
from vls.voxtell_states import VoxTellStateInterface
from vls.v8_3_real_aug_oracle_reliability import (
    ACTION_PROTOCOL,
    CsvSink,
    METRIC_FIELDS,
    METRIC_NAMES,
    PATCH_FIELDS,
    REGION_FIELDS,
    REGIONS,
    RELIABILITY_METHODS,
    SCOPES,
    PatchMetricAccumulator,
    RegionAccumulator,
    RunningMean,
    memory_status,
    rank_metrics,
    real_augmentation_reliabilities,
    write_progress,
)


OUTPUT_DIR = "outputs/v8_3b_stage_matched_oracle"
SELECTED_STAGE = "decoder_stage_1_low_to_high"
FIXED_V82_IMAGINED_PAIRWISE = {
    "auroc": 0.2964244171792574,
    "spearman": -0.322833140573606,
}
FIXED_V83_FULL_FINAL_PAIRWISE = {
    "auroc": 0.5795445063841145,
    "spearman": 0.08528437710498045,
}


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V8.3b stage-matched real-augmentation oracle diagnostic")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--selected-stage", default=SELECTED_STAGE)
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


def stage_probability_and_source_final_pseudo(
    interface: VoxTellStateInterface,
    patch: torch.Tensor,
    prompt_embedding: torch.Tensor,
    selected_stage: str,
    device: torch.device,
    prediction_threshold: float,
    need_final_pseudo: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    with torch.inference_mode():
        result = interface.forward_with_states(patch, prompt_embedding)
        state = result["decoder_states"][selected_stage][:, 0].detach().float().to(device)
        stage_logits = state_to_intermediate_prediction(interface, selected_stage, state)
        stage_probability = torch.sigmoid(stage_logits).detach().float().cpu().numpy()
        final_pseudo = None
        if need_final_pseudo:
            final_probability = torch.sigmoid(
                result["final_prediction"][:, 0:1].detach().float(),
            ).cpu().numpy()
            final_pseudo = final_probability > prediction_threshold
            del final_probability
    del result, state, stage_logits, patch
    return stage_probability, final_pseudo


def align_binary_to_stage(binary: np.ndarray, stage_shape: tuple[int, int, int]) -> np.ndarray:
    tensor = torch.from_numpy(binary.astype(np.float32, copy=False))
    if tuple(tensor.shape[-3:]) != stage_shape:
        tensor = F.interpolate(tensor, size=stage_shape, mode="nearest")
    return tensor.numpy().astype(bool, copy=False)


def run(args: argparse.Namespace) -> None:
    if args.selected_stage != SELECTED_STAGE:
        raise AssertionError("V8.3b must use the V8.2 selected stage")
    if ACTION_PROTOCOL != (("gamma", 0.30), ("blur", 1.5)):
        raise AssertionError("V8.3b action protocol differs from V8.2")
    if args.patches_per_case != 4 or args.foreground_patches_per_case != 2:
        raise AssertionError("V8.3b patch protocol must be 4 patches and 2 foreground patches per case")
    if args.foreground_candidate_patches != 16 or args.prediction_threshold != 0.5:
        raise AssertionError("V8.3b thresholds/candidate protocol differs from V8.2")

    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V8.3b requires CUDA, resolved {device}")
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    test_cases = iter_cases(paths, split="test")
    if len(test_cases) != 8:
        raise AssertionError(f"V8.3b requires exactly 8 test cases, got {len(test_cases)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_sink = CsvSink(output_dir / "stage_matched_reliability_metrics.csv", METRIC_FIELDS)
    region_sink = CsvSink(output_dir / "stage_matched_reliability_regions.csv", REGION_FIELDS)
    patch_sink = CsvSink(output_dir / "stage_matched_patch_metrics.csv", PATCH_FIELDS)
    progress_path = output_dir / "progress.json"
    completed_cases: list[str] = []
    completed_patch_rows = 0
    write_progress(progress_path, completed_cases, completed_patch_rows, device, "initializing")

    try:
        print("[V8.3b] loading frozen VoxTell/base model", flush=True)
        interface = VoxTellStateInterface.from_model_dir(
            paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
        )
        prepare_functional_seg_head(interface, args.selected_stage)
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
            print(f"[V8.3b] case {case_index}/8 start {case.case}", flush=True)
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
                    f"[V8.3b] case {case_index}/8 patch {patch_index}/4 source stage forward",
                    flush=True,
                )
                source_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
                source_stage_probability, source_final_pseudo = stage_probability_and_source_final_pseudo(
                    interface,
                    source_patch,
                    prompt_embedding,
                    args.selected_stage,
                    device,
                    args.prediction_threshold,
                    True,
                )
                transformed_stage_probabilities: dict[str, np.ndarray] = {}
                for action, strength in ACTION_PROTOCOL:
                    print(
                        f"[V8.3b] case {case_index}/8 patch {patch_index}/4 action={action} stage forward",
                        flush=True,
                    )
                    transformed_padded, _ = padded_visual_action_and_slicers(
                        interface.predictor, image, action, strength,
                    )
                    if tuple(transformed_padded.shape) != tuple(original_padded.shape):
                        raise AssertionError(f"Action padded shape mismatch for {case.case} {action}")
                    action_patch = torch.clone(transformed_padded[slicer][None], memory_format=torch.contiguous_format)
                    transformed_stage_probabilities[action], _ = stage_probability_and_source_final_pseudo(
                        interface,
                        action_patch,
                        prompt_embedding,
                        args.selected_stage,
                        device,
                        args.prediction_threshold,
                        False,
                    )
                    del transformed_padded, action_patch

                stage_shape = tuple(int(size) for size in source_stage_probability.shape[-3:])
                gt_stage = align_binary_to_stage(
                    (label_padded[slicer][None].detach().cpu().numpy() == args.label_value),
                    stage_shape,
                )
                source_pseudo_stage = align_binary_to_stage(source_final_pseudo, stage_shape)
                correct = source_pseudo_stage == gt_stage
                raw_reliability_maps = real_augmentation_reliabilities(
                    source_stage_probability,
                    transformed_stage_probabilities["gamma"],
                    transformed_stage_probabilities["blur"],
                )
                reliability_maps = {
                    "confidence_rank": raw_reliability_maps["confidence_rank"],
                    "real_stage_gamma_world_stability": raw_reliability_maps["real_gamma_world_stability"],
                    "real_stage_blur_world_stability": raw_reliability_maps["real_blur_world_stability"],
                    "real_stage_pairwise_world_stability": raw_reliability_maps["real_pairwise_world_stability"],
                    "real_stage_gamma_joint_product": raw_reliability_maps["real_gamma_joint_product"],
                    "real_stage_blur_joint_product": raw_reliability_maps["real_blur_joint_product"],
                    "real_stage_pairwise_joint_product": raw_reliability_maps["real_pairwise_joint_product"],
                }
                flat_gt = gt_stage.reshape(-1)
                flat_pseudo = source_pseudo_stage.reshape(-1)
                flat_correct = correct.reshape(-1)
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

                del raw_reliability_maps, reliability_maps, transformed_stage_probabilities
                del source_stage_probability, source_final_pseudo
                del gt_stage, source_pseudo_stage, correct
                gc.collect()
                print(
                    f"[V8.3b] case {case_index}/8 patch {patch_index}/4 reliability done",
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
            print(f"[V8.3b] case {case_index}/8 complete memory={memory_status(device)}", flush=True)
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

        region_summary: dict[str, dict[str, float | None]] = {}
        for (method, region), statistic in sorted(global_regions.items()):
            region_summary.setdefault(method, {})[region] = statistic.mean()
            region_sink.write({
                "method": method,
                "case": "__all__",
                "region": region,
                "voxel_count": None,
                "reliability_mean": statistic.mean(),
                "aggregation": "8_case_macro",
            })

        stage_world_methods = {
            "gamma": "real_stage_gamma_world_stability",
            "blur": "real_stage_blur_world_stability",
            "pairwise": "real_stage_pairwise_world_stability",
        }

        def foreground_value(method: str, name: str) -> float | None:
            value = foreground_rows.get(method, {}).get(name)
            return None if value is None else float(value)

        stage_metrics = {
            name: {
                "method": method,
                "auroc": foreground_value(method, "auroc"),
                "auprc": foreground_value(method, "auprc"),
                "spearman": foreground_value(method, "spearman"),
            }
            for name, method in stage_world_methods.items()
        }
        pairwise_auroc = stage_metrics["pairwise"]["auroc"]
        pairwise_spearman = stage_metrics["pairwise"]["spearman"]
        stage_matched_supported = (
            None
            if pairwise_auroc is None or pairwise_spearman is None
            else pairwise_auroc > 0.5 and pairwise_spearman > 0.0
        )
        inversion = {
            name: (
                None
                if values["auroc"] is None or values["spearman"] is None
                else values["auroc"] < 0.5 or values["spearman"] < 0.0
            )
            for name, values in stage_metrics.items()
        }
        if stage_matched_supported is True:
            interpretation = "World Predictor transition fidelity/supervision is the leading isolated factor relative to the fixed V8.2 imagined reference"
        elif stage_matched_supported is False and FIXED_V83_FULL_FINAL_PAIRWISE["auroc"] > 0.5:
            interpretation = "selected-stage/intermediate-head observation path remains problematic despite effective full-final real augmentation"
        else:
            interpretation = "stage-matched evidence is inconclusive under the fixed protocol"

        summary = {
            "stage": "V8.3b Stage-Matched Real-Aug Oracle Diagnostic",
            "test_cases": [case.case for case in test_cases],
            "test_case_count": len(test_cases),
            "completed_case_count": len(completed_cases),
            "selected_stage": args.selected_stage,
            "patch_protocol": {
                "patches_per_case": args.patches_per_case,
                "foreground_patches_per_case": args.foreground_patches_per_case,
                "foreground_candidate_patches": args.foreground_candidate_patches,
            },
            "action_protocol": {"gamma": 0.30, "blur": 1.5},
            "world_predictor_loaded": False,
            "models_trained": False,
            "gt_used_only_for_reliability_diagnosis": True,
            "final_prediction_used_for_reliability": False,
            "source_pseudo_label_definition": "V8.2-compatible source final_prediction thresholded at 0.5, nearest-neighbor aligned to selected-stage grid for correctness diagnosis only",
            "reliability_formula_modified": False,
            "stage_matched_oracle_supported": stage_matched_supported,
            "real_stage_gamma_foreground_auroc": stage_metrics["gamma"]["auroc"],
            "real_stage_gamma_foreground_spearman": stage_metrics["gamma"]["spearman"],
            "real_stage_blur_foreground_auroc": stage_metrics["blur"]["auroc"],
            "real_stage_blur_foreground_spearman": stage_metrics["blur"]["spearman"],
            "real_stage_pairwise_foreground_auroc": stage_metrics["pairwise"]["auroc"],
            "real_stage_pairwise_foreground_spearman": stage_metrics["pairwise"]["spearman"],
            "foreground_metrics": stage_metrics,
            "foreground_reliability_inversion": inversion,
            "foreground_region_reliability_means": {
                name: {region: region_summary.get(method, {}).get(region) for region in REGIONS}
                for name, method in stage_world_methods.items()
            },
            "fixed_reference_comparison": {
                "v8_2_imagined_pairwise_foreground": FIXED_V82_IMAGINED_PAIRWISE,
                "v8_3_full_final_real_pairwise_foreground": FIXED_V83_FULL_FINAL_PAIRWISE,
                "note": "Fixed reference values are recorded only; V8.2 and V8.3 were not loaded or rerun by V8.3b",
            },
            "interpretation": interpretation,
            "outputs": {
                "metrics": str(output_dir / "stage_matched_reliability_metrics.csv"),
                "regions": str(output_dir / "stage_matched_reliability_regions.csv"),
                "patch_metrics": str(output_dir / "stage_matched_patch_metrics.csv"),
                "progress": str(progress_path),
                "summary": str(output_dir / "summary.json"),
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_progress(progress_path, completed_cases, completed_patch_rows, device, "complete")
        print("[V8.3b] complete", flush=True)
    finally:
        for sink in (metric_sink, region_sink, patch_sink):
            sink.close()


if __name__ == "__main__":
    run(parse_args())
