from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import CaseRecord, read_image, read_image_and_label
from vls.v2_experiment import padded_image_and_slicers, prepare_functional_seg_head, resolve_device
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, load_world_model
from vls.v7_0d_protocol_sanity import (
    STRONG_ACTIONS,
    VARIANTS,
    build_evaluation_cache,
    evaluate_network,
    iter_cases,
    iter_image_cases,
    reliability_from_source,
    select_patch_slicers,
    set_seed,
    strong_padded_image,
)
from vls.v7_1a_lora_qkv_smoke import inject_lora_qkv, lora_parameters
from vls.v7_1b_protocol_consolidation import (
    EVAL_STEPS,
    evaluate_full_volume,
    full_volume_metric_row,
    make_manifest,
    pool_full_volume,
    write_csv,
)
from vls.voxtell_states import VoxTellStateInterface


def slicer_from_record(record: dict[str, Any]) -> tuple[slice, ...]:
    return (
        slice(None),
        *(
            slice(int(start), int(stop), None)
            for start, stop in zip(record["slicer_start"], record["slicer_stop"], strict=True)
        ),
    )


def validate_manifest(
    manifest: dict[str, Any],
    train_cases: list[CaseRecord],
    args: argparse.Namespace,
    predictor: Any,
) -> None:
    if int(manifest.get("foreground_candidate_patches", -1)) != args.foreground_candidate_patches:
        raise AssertionError("foreground_candidate_patches differs from canonical manifest")
    records = manifest.get("records", [])
    if len(records) != len(train_cases):
        raise AssertionError("canonical manifest does not contain one patch per train case")
    expected_cases = [case.case for case in train_cases]
    if [record["case"] for record in records] != expected_cases:
        raise AssertionError("canonical manifest train case order differs from split_json")
    expected_patch_size = tuple(int(size) for size in predictor.patch_size)
    for index, record in enumerate(records):
        if record.get("patch_kind") != "foreground":
            raise AssertionError(f"manifest record {index} is not a foreground patch")
        shape = tuple(
            int(stop) - int(start)
            for start, stop in zip(record["slicer_start"], record["slicer_stop"], strict=True)
        )
        if shape != expected_patch_size:
            raise AssertionError(f"manifest patch size {shape} != VoxTell patch size {expected_patch_size}")
        family, strength = STRONG_ACTIONS[index % len(STRONG_ACTIONS)]
        expected_aug = f"{family}:{strength}"
        if record.get("augmentation") != expected_aug:
            raise AssertionError(f"manifest augmentation {record.get('augmentation')} != {expected_aug}")
    config = manifest.get("config", {})
    if config:
        checks = {
            "patches_per_case": args.patches_per_case,
            "foreground_patches_per_case": args.foreground_patches_per_case,
            "selected_stage": args.selected_stage,
        }
        for key, expected in checks.items():
            if config.get(key) != expected:
                raise AssertionError(f"manifest config {key} differs from current configuration")


def build_cache_from_manifest(
    interface: VoxTellStateInterface,
    world_model: nn.Module,
    cases: list[CaseRecord],
    prompt_embedding: torch.Tensor,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    validate_manifest(manifest, cases, args, interface.predictor)
    cache: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        image, _ = read_image(case)
        original_padded, _ = padded_image_and_slicers(interface.predictor, image)
        record = manifest["records"][case_index]
        slicer = slicer_from_record(record)
        if any(stop > size for stop, size in zip(record["slicer_stop"], original_padded.shape[-3:], strict=True)):
            raise AssertionError(f"manifest slicer is outside padded image for {case.case}")
        family, strength = STRONG_ACTIONS[case_index % len(STRONG_ACTIONS)]
        strong_padded = strong_padded_image(
            interface, image, family, strength, original_padded,
        )
        patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
        strong_patch = torch.clone(strong_padded[slicer][None], memory_format=torch.contiguous_format)
        result = interface.forward_with_states(patch, prompt_embedding)
        source_state = result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
        source_probability = torch.sigmoid(result["final_prediction"][:, 0:1].detach().float().to(device))
        final_shape = tuple(int(size) for size in source_probability.shape[-3:])
        weights = reliability_from_source(
            interface, world_model, source_state, source_probability,
            args.selected_stage, final_shape, device,
        )
        pseudo = (source_probability > args.prediction_threshold).float()
        cache.append({
            "case": case.case,
            "patch_index": int(record["patch_index"]),
            "patch_kind": record["patch_kind"],
            "slicer": slicer,
            "image": strong_patch.detach().cpu(),
            "embedding": prompt_embedding.detach().cpu(),
            "pseudo": pseudo.detach().cpu(),
            "weights": weights,
            "has_foreground": bool(torch.count_nonzero(pseudo)),
            "augmentation": record["augmentation"],
            "case_index": case_index,
        })
    return cache


def class_balanced_loss(
    logits: torch.Tensor,
    pseudo: torch.Tensor,
    reliability: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    bce = F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
    positive = pseudo > 0.5
    negative = ~positive
    class_losses: list[torch.Tensor] = []
    stats: dict[str, float | int] = {
        "pseudo_positive_voxels": int(positive.sum().detach().cpu()),
        "pseudo_negative_voxels": int(negative.sum().detach().cpu()),
        "positive_fraction": float(positive.float().mean().detach().cpu()),
    }
    for name, mask in (("positive", positive), ("negative", negative)):
        selected_reliability = reliability[mask]
        selected_bce = bce[mask]
        weight_sum = selected_reliability.sum()
        present = bool(mask.any()) and float(weight_sum.detach().cpu()) > 1e-12
        stats[f"{name}_weight_sum"] = float(weight_sum.detach().cpu())
        stats[f"{name}_weight_mean"] = float(selected_reliability.mean().detach().cpu()) if bool(mask.any()) else 0.0
        if present:
            class_loss = (selected_reliability * selected_bce).sum() / weight_sum
            class_losses.append(class_loss)
            stats[f"L_{'pos' if name == 'positive' else 'neg'}"] = float(class_loss.detach().cpu())
        else:
            stats[f"L_{'pos' if name == 'positive' else 'neg'}"] = 0.0
    if not class_losses:
        raise RuntimeError("Both pseudo-label classes have zero usable reliability")
    total = torch.stack(class_losses).mean()
    stats["total_loss"] = float(total.detach().cpu())
    return total, stats


def train_variant(
    variant: str,
    source: str,
    order: str,
    base_network: nn.Module,
    train_samples: list[dict[str, Any]],
    eval_cases: list[CaseRecord],
    full_data: dict[str, tuple[np.ndarray, np.ndarray]],
    eval_cache: list[dict[str, Any]],
    interface: VoxTellStateInterface,
    embedding: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    base_total: int,
    target_names: list[str] | None,
    evaluate_full: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(args.seed)
    student = copy.deepcopy(base_network)
    targets = inject_lora_qkv(student, args.lora_rank, args.lora_alpha, args.lora_dropout)
    student = student.to(device)
    trainable = lora_parameters(student)
    base_trainable = sum(
        parameter.numel()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    )
    if base_trainable != 0:
        raise AssertionError(f"base_trainable_parameters={base_trainable}, expected 0")
    if target_names is not None and targets != target_names:
        raise AssertionError("LoRA target modules differ across reinitializations")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    ordered_once = train_samples if order == "forward" else list(reversed(train_samples))
    ordered = ordered_once * args.training_rounds
    loss_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for step, sample in enumerate(ordered, start=1):
        image = sample["image"].to(device).clone()
        embedding_batch = sample["embedding"].to(device).clone().float()
        pseudo = sample["pseudo"].to(device).clone()
        reliability = torch.from_numpy(sample["weights"][source]).to(device=device, dtype=torch.float32).view_as(pseudo)
        optimizer.zero_grad(set_to_none=True)
        result = interface._network_forward_with_states(student, image, embedding_batch)
        logits = result["final_prediction"][:, 0:1].float()
        loss, step_stats = class_balanced_loss(logits, pseudo, reliability)
        loss.backward()
        gradient_norm = float(torch.sqrt(sum(
            parameter.grad.detach().float().pow(2).sum()
            for parameter in trainable if parameter.grad is not None
        )).detach().cpu())
        before = [parameter.detach().clone() for parameter in trainable]
        optimizer.step()
        update_delta = float(torch.sqrt(sum(
            (parameter.detach() - old).float().pow(2).sum()
            for parameter, old in zip(trainable, before, strict=True)
        )).detach().cpu())
        round_index = (step - 1) // len(train_samples) + 1
        loss_rows.append({
            "variant": variant, "order": order, "step": step, "round": round_index,
            "case": sample["case"], "patch_index": sample["patch_index"],
            "patch_kind": sample["patch_kind"], "augmentation": sample["augmentation"],
            "loss": float(loss.detach().cpu()), "gradient_norm": gradient_norm,
            "update_delta_norm": update_delta, "learning_rate": args.learning_rate,
            "weight_decay": 0.0, "base_trainable_parameters": base_trainable,
            "lora_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        })
        pseudo_rows.append({
            "variant": variant, "order": order, "step": step, "round": round_index,
            "case": sample["case"], "patch_index": sample["patch_index"],
            "reliability_source": source, **step_stats,
        })
        if step in EVAL_STEPS and (evaluate_full or step == args.training_rounds * len(train_samples)):
            full_rows.extend(evaluate_full_volume(
                interface, student, eval_cases, full_data, embedding,
                variant, order, step, args.label_value, args.prediction_threshold,
            ))
            debug_rows.extend(evaluate_network(
                variant, order, step, student, eval_cache, interface, device,
                args.prediction_threshold,
            ))
        del result, logits, loss, image, embedding_batch, pseudo, reliability, before
    stats = {
        "target_modules": targets,
        "base_trainable_parameters": base_trainable,
        "lora_parameter_count": sum(parameter.numel() for parameter in trainable),
        "lora_ratio_of_base_model": sum(parameter.numel() for parameter in trainable) / max(base_total, 1),
    }
    del student, optimizer, trainable
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_rows, pseudo_rows, full_rows, debug_rows, stats


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V7.1c class-balanced pseudo-label loss sanity")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir)); parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root)); parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--train-manifest", default="outputs/v7_1b_protocol_consolidation/train_patch_manifest.json")
    parser.add_argument("--output-dir", default="outputs/v7_1c_class_balanced_loss_sanity")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--train-cases", type=int, default=4); parser.add_argument("--evaluation-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=2); parser.add_argument("--foreground-patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16); parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5); parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--hidden-channels", type=int, default=16); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4); parser.add_argument("--lora-alpha", type=float, default=8.0); parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--training-rounds", type=int, default=5); parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V7.1c requires CUDA, resolved {device}")
    if args.training_rounds != 5:
        raise AssertionError("V7.1c is fixed to five training rounds")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(voxtell_root=Path(args.voxtell_root), voxtell_model_dir=Path(args.model_dir), data_root=Path(args.data_root), split_json=Path(args.split_json))
    train_cases = iter_image_cases(paths, "train", args.train_cases); eval_cases = iter_cases(paths, "test", args.evaluation_cases)
    train_names = [case.case for case in train_cases]; eval_names = [case.case for case in eval_cases]
    overlap = sorted(set(train_names) & set(eval_names))
    if overlap: raise AssertionError(f"train/evaluation case overlap: {overlap}")

    teacher = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
    prepare_functional_seg_head(teacher, args.selected_stage); prompt_embedding = teacher.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    checkpoint_path = Path(args.world_checkpoint); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = load_world_model(checkpoint_path, int(checkpoint["state_dict"]["output_projection.bias"].shape[0]), device, args.hidden_channels)

    manifest_path = Path(args.train_manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        train_cache = build_cache_from_manifest(teacher, world_model, train_cases, prompt_embedding, manifest, args, device)
        manifest_created = False
    else:
        from vls.v7_1b_protocol_consolidation import build_train_cache_with_slicers
        selected_cache = build_train_cache_with_slicers(teacher, world_model, train_cases, prompt_embedding, args, device)
        selected = []
        for case in train_cases:
            candidates = [sample for sample in selected_cache if sample["case"] == case.case and sample["patch_kind"] == "foreground" and sample["has_foreground"]]
            if not candidates: raise RuntimeError(f"No nonempty teacher foreground patch for {case.case}")
            selected.append(sorted(candidates, key=lambda sample: sample["patch_index"])[0])
        selected.sort(key=lambda sample: (sample["case_index"], sample["patch_index"]))
        manifest = make_manifest(selected, args)
        manifest["config"] = {"patches_per_case": args.patches_per_case, "foreground_patches_per_case": args.foreground_patches_per_case, "selected_stage": args.selected_stage}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists(): raise RuntimeError("manifest appeared during generation; refusing overwrite")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifest_created = True
        train_cache = build_cache_from_manifest(teacher, world_model, train_cases, prompt_embedding, manifest, args, device)
    if len(train_cache) != len(train_cases): raise AssertionError("manifest reconstruction did not produce all train cases")
    train_samples = train_cache
    eval_cache = build_evaluation_cache(teacher, world_model, eval_cases, prompt_embedding, args, device)
    full_data = {}
    for case in eval_cases:
        image, label, _ = read_image_and_label(case); full_data[case.case] = (image, label)

    world_model.to("cpu"); teacher.network.to("cpu"); teacher.functional_seg_head.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    base_network = copy.deepcopy(teacher.network).cpu().eval(); base_total = sum(parameter.numel() for parameter in base_network.parameters())
    for parameter in base_network.parameters(): parameter.requires_grad = False
    base_network.to(device)
    initial_full = evaluate_full_volume(teacher, base_network, eval_cases, full_data, prompt_embedding, "A_init_no_adaptation", "forward", 0, args.label_value, args.prediction_threshold)
    initial_debug = evaluate_network("A_init_no_adaptation", "forward", 0, base_network, eval_cache, teacher, device, args.prediction_threshold)
    base_network.to("cpu"); torch.cuda.empty_cache() if device.type == "cuda" else None

    losses = []; pseudo_stats = []; full_rows = list(initial_full); debug_rows = list(initial_debug); stats = []; target_names = None
    for variant, source in VARIANTS.items():
        print(f"[V7.1c] forward {variant}", flush=True)
        l, p, f, d, s = train_variant(variant, source, "forward", base_network, train_samples, eval_cases, full_data, eval_cache, teacher, prompt_embedding, args, device, base_total, target_names, True)
        target_names = s["target_modules"] if target_names is None else target_names; losses.extend(l); pseudo_stats.extend(p); full_rows.extend(f); debug_rows.extend(d); stats.append({"variant": variant, "order": "forward", **s})
        print(f"[V7.1c] reverse {variant}", flush=True)
        l, p, f, d, s = train_variant(variant, source, "reverse", base_network, train_samples, eval_cases, full_data, eval_cache, teacher, prompt_embedding, args, device, base_total, target_names, False)
        losses.extend(l); pseudo_stats.extend(p); full_rows.extend(f); debug_rows.extend(d); stats.append({"variant": variant, "order": "reverse", **s})

    curve = pool_full_volume(full_rows); init_by_case = {row["case"]: row for row in initial_full}; init_curve = next(row for row in curve if row["variant"] == "A_init_no_adaptation")
    for row in full_rows:
        if row["variant"] != "A_init_no_adaptation":
            base = init_by_case[row["case"]]; row["delta_dice_vs_A_init"] = row["dice"] - base["dice"]; row["delta_foreground_iou_vs_A_init"] = row["foreground_iou"] - base["foreground_iou"]; row["delta_precision_vs_A_init"] = row["precision"] - base["precision"]; row["delta_recall_vs_A_init"] = row["recall"] - base["recall"]
    for row in curve:
        if row["variant"] != "A_init_no_adaptation": row["delta_mean_dice_vs_A_init"] = row["mean_dice"] - init_curve["mean_dice"]; row["delta_mean_foreground_iou_vs_A_init"] = row["mean_foreground_iou"] - init_curve["mean_foreground_iou"]
    final = {(row["variant"], row["order"]): row for row in curve if int(row["step"]) == 20}; sensitivity = {}
    for variant in VARIANTS:
        f, r = final[(variant, "forward")], final[(variant, "reverse")]
        sensitivity[variant] = {"forward_minus_reverse_mean_dice": f["mean_dice"] - r["mean_dice"], "forward_minus_reverse_mean_foreground_iou": f["mean_foreground_iou"] - r["mean_foreground_iou"], "forward_minus_reverse_mean_precision": f["mean_precision"] - r["mean_precision"], "forward_minus_reverse_mean_recall": f["mean_recall"] - r["mean_recall"]}
    if device.type == "cuda": torch.cuda.synchronize(device)
    peak_allocated = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0
    write_csv(output_dir / "training_loss.csv", losses); write_csv(output_dir / "pseudo_label_stats.csv", pseudo_stats); write_csv(output_dir / "full_volume_by_case.csv", full_rows); write_csv(output_dir / "full_volume_curve.csv", curve); write_csv(output_dir / "sampled_patch_debug.csv", debug_rows)
    (output_dir / "parameter_stats.json").write_text(json.dumps(stats, indent=2))
    summary = {
        "stage": "V7.1c class-balanced pseudo-label loss sanity", "source_checkpoint": str(checkpoint_path), "selected_stage": args.selected_stage, "seed": args.seed,
        "resolved_device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "peak_cuda_allocated_mb": peak_allocated, "peak_cuda_reserved_mb": peak_reserved,
        "adaptation_cases": train_names, "evaluation_cases": eval_names, "case_overlap": overlap, "case_overlap_count": len(overlap),
        "train_manifest_path": str(manifest_path), "manifest_created_this_run": manifest_created, "manifest_reused_without_overwrite": not manifest_created,
        "fixed_manifest_reused_for_all_variants_and_rounds": True, "foreground_candidate_patches": args.foreground_candidate_patches, "effective_updates_per_variant_order": 20,
        "full_volume_eval_steps_forward": [0, *EVAL_STEPS], "reverse_confirmation_step": 20, "variants": VARIANTS,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "target_modules": target_names, "base_trainable_parameters": 0, "parameter_stats": stats},
        "training": {"rounds": args.training_rounds, "updates_per_round": 4, "total_updates": 20, "loss": "class-balanced reliability-weighted pseudo-label BCE: mean of existing L_pos/L_neg", "learning_rate": args.learning_rate, "weight_decay": 0.0, "teacher_gt_used_for_selection": False, "world_predictor_updated": False, "base_voxtell_parameters_frozen": True, "student_view": "fixed gamma(+0.30)/blur(1.5) from canonical manifest"},
        "full_volume_inference": {"implementation": "V7.1b single-window CUDA forward with VoxTell native slicers, Gaussian overlap accumulation, and bbox restoration", "overlap_aggregation": "VoxTell native Gaussian sliding-window aggregation", "sampled_patch_metrics_are_debug_only": True},
        "final_step20_full_volume": [row for row in curve if int(row["step"]) == 20], "order_sensitivity": sensitivity,
        "outputs": {name: str(output_dir / name) for name in ("training_loss.csv", "pseudo_label_stats.csv", "full_volume_by_case.csv", "full_volume_curve.csv", "sampled_patch_debug.csv", "parameter_stats.json", "summary.json")},
        "status": "complete; fixed 20-update forward plus reverse confirmation; no early stopping or hyperparameter selection",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__": run(parse_args())
