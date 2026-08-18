from __future__ import annotations

import argparse
import copy
import csv
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
from vls.v2_experiment import prepare_functional_seg_head, resolve_device, select_patch_slicers
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, load_world_model
from vls.v7_0d_protocol_sanity import (
    STRONG_ACTIONS,
    VARIANTS,
    binary_metrics,
    build_evaluation_cache,
    evaluate_network,
    iter_cases,
    iter_image_cases,
    reliability_from_source,
    set_seed,
    strong_padded_image,
    visual_action,
)
from vls.v7_1a_lora_qkv_smoke import inject_lora_qkv, lora_parameters
from vls.voxtell_states import VoxTellStateInterface


ORDERS = ("forward", "reverse")
EVAL_STEPS = (4, 8, 12, 16, 20)


def sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def slicer_coordinates(slicer: tuple[slice, ...]) -> dict[str, Any]:
    spatial = slicer[-3:]
    starts = [int(item.start) for item in spatial]
    stops = [int(item.stop) for item in spatial]
    steps = [None if item.step is None else int(item.step) for item in spatial]
    return {
        "spatial_axes": ["x", "y", "z"],
        "slicer_start": starts,
        "slicer_stop": stops,
        "slicer_step": steps,
    }


def build_train_cache_with_slicers(
    interface: VoxTellStateInterface,
    world_model: nn.Module,
    cases: list[CaseRecord],
    prompt_embedding: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    """V7.0d train cache with the selected slicer retained for the manifest."""
    cache: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        image, _ = read_image(case)
        original_padded, slicers, patch_kinds = select_patch_slicers(
            interface,
            image,
            prompt_embedding,
            args.patches_per_case,
            args.foreground_patches_per_case,
            args.foreground_candidate_patches,
            args.foreground_threshold,
        )
        action_family, strength = STRONG_ACTIONS[case_index % len(STRONG_ACTIONS)]
        strong_padded = strong_padded_image(
            interface, image, action_family, strength, original_padded,
        )
        for patch_index, slicer in enumerate(slicers):
            patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
            strong_patch = torch.clone(strong_padded[slicer][None], memory_format=torch.contiguous_format)
            result = interface.forward_with_states(patch, prompt_embedding)
            source_state = result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
            source_probability = torch.sigmoid(
                result["final_prediction"][:, 0:1].detach().float().to(device),
            )
            final_shape = tuple(int(size) for size in source_probability.shape[-3:])
            weights = reliability_from_source(
                interface,
                world_model,
                source_state,
                source_probability,
                args.selected_stage,
                final_shape,
                device,
            )
            pseudo = (source_probability > args.prediction_threshold).float()
            cache.append({
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "slicer": slicer,
                "image": strong_patch.detach().cpu(),
                "embedding": prompt_embedding.detach().cpu(),
                "pseudo": pseudo.detach().cpu(),
                "weights": weights,
                "has_foreground": bool(torch.count_nonzero(pseudo)),
                "augmentation": f"{action_family}:{strength}",
                "case_index": case_index,
            })
    return cache


def make_manifest(train_samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    records = []
    for sample in train_samples:
        records.append({
            "case": sample["case"],
            "patch_index": int(sample["patch_index"]),
            "patch_kind": sample["patch_kind"],
            **slicer_coordinates(sample["slicer"]),
            "augmentation": sample["augmentation"],
        })
    return {
        "stage": "V7.1b fixed train patch manifest",
        "selection_source": "teacher prediction only; target-train GT is not read",
        "foreground_candidate_patches": args.foreground_candidate_patches,
        "records": records,
    }


def load_manifest_samples(
    manifest_path: Path,
    train_cache: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    by_key = {(sample["case"], int(sample["patch_index"])): sample for sample in train_cache}
    samples = []
    for record in manifest["records"]:
        key = (record["case"], int(record["patch_index"]))
        if key not in by_key:
            raise KeyError(f"Manifest patch is absent from fixed train cache: {key}")
        sample = by_key[key]
        coordinates = slicer_coordinates(sample["slicer"])
        for field in ("slicer_start", "slicer_stop", "slicer_step"):
            if record[field] != coordinates[field]:
                raise AssertionError(f"Manifest slicer mismatch for {key}: {field}")
        samples.append(sample)
    return manifest, samples


def full_volume_prediction(
    interface: VoxTellStateInterface,
    network: nn.Module,
    image: np.ndarray,
    embedding: torch.Tensor,
) -> np.ndarray:
    """Run VoxTell's native Gaussian sliding-window path without its producer thread."""
    predictor = interface.predictor
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    data, slicer_revert_padding = pad_nd_image(
        preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None,
    )
    slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
    results = torch.zeros(
        (embedding.shape[1], *data.shape[1:]), dtype=torch.half, device="cpu",
    )
    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device="cpu")
    gaussian = compute_gaussian(
        tuple(predictor.patch_size), sigma_scale=1.0 / 8,
        value_scaling_factor=10, device=torch.device("cpu"),
    )
    network_device = next(network.parameters()).device
    embedding_device = embedding.to(network_device)
    network.eval()
    with torch.inference_mode():
        for tile_slice in slicers:
            patch = torch.clone(data[tile_slice][None], memory_format=torch.contiguous_format).to(network_device)
            context = torch.autocast(network_device.type, enabled=True) if network_device.type == "cuda" else torch.no_grad()
            with context:
                prediction = network(patch, embedding_device)[0].to("cpu")
            prediction = prediction * gaussian
            results[tile_slice] += prediction
            n_predictions[tile_slice[1:]] += gaussian
            del patch, prediction
    torch.div(results, n_predictions, out=results)
    results = results[(slice(None), *slicer_revert_padding[1:])].float().numpy()

    restored = np.zeros((results.shape[0], *original_shape), dtype=np.float32)
    restored = insert_crop_into_image(restored, results, bbox)
    if network_device.type == "cuda":
        torch.cuda.empty_cache()
    return restored[0]


def full_volume_metric_row(
    variant: str,
    order: str,
    step: int,
    case: CaseRecord,
    prediction_logits: np.ndarray,
    label: np.ndarray,
    label_value: int,
    threshold: float,
) -> dict[str, Any]:
    prediction = (sigmoid_numpy(prediction_logits) > threshold).reshape(-1)
    target = (label == label_value).reshape(-1)
    metrics = binary_metrics(prediction, target)
    iou = 1.0 if metrics["tp"] + metrics["fp"] + metrics["fn"] == 0 else metrics["tp"] / max(metrics["tp"] + metrics["fp"] + metrics["fn"], 1)
    return {
        "variant": variant,
        "order": order,
        "step": step,
        "case": case.case,
        "scope": "full_volume",
        **metrics,
        "foreground_iou": iou,
    }


def evaluate_full_volume(
    interface: VoxTellStateInterface,
    network: nn.Module,
    cases: list[CaseRecord],
    full_data: dict[str, tuple[np.ndarray, np.ndarray]],
    embedding: torch.Tensor,
    variant: str,
    order: str,
    step: int,
    label_value: int,
    threshold: float,
) -> list[dict[str, Any]]:
    network.eval()
    rows = []
    with torch.inference_mode():
        for case in cases:
            image, label = full_data[case.case]
            logits = full_volume_prediction(interface, network, image, embedding)
            rows.append(full_volume_metric_row(variant, order, step, case, logits, label, label_value, threshold))
            del logits
            network_device = next(network.parameters()).device
            if network_device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def pool_full_volume(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["variant"], row["order"], int(row["step"])), []).append(row)
    pooled = []
    for (variant, order, step), group in sorted(groups.items()):
        pooled.append({
            "variant": variant,
            "order": order,
            "step": step,
            "scope": "full_volume",
            "case_count": len(group),
            "mean_dice": float(np.mean([row["dice"] for row in group])),
            "std_dice": float(np.std([row["dice"] for row in group])),
            "mean_foreground_iou": float(np.mean([row["foreground_iou"] for row in group])),
            "std_foreground_iou": float(np.std([row["foreground_iou"] for row in group])),
            "mean_precision": float(np.mean([row["precision"] for row in group])),
            "mean_recall": float(np.mean([row["recall"] for row in group])),
        })
    return pooled


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
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
        weight = torch.from_numpy(sample["weights"][source]).to(device=device, dtype=torch.float32).view_as(pseudo)
        optimizer.zero_grad(set_to_none=True)
        result = interface._network_forward_with_states(student, image, embedding_batch)
        logits = result["final_prediction"][:, 0:1].float()
        per_voxel = F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
        loss = (per_voxel * weight).sum() / weight.sum().clamp_min(1e-6)
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
        loss_rows.append({
            "variant": variant,
            "order": order,
            "step": step,
            "round": (step - 1) // len(train_samples) + 1,
            "case": sample["case"],
            "patch_index": sample["patch_index"],
            "patch_kind": sample["patch_kind"],
            "augmentation": sample["augmentation"],
            "loss": float(loss.detach().cpu()),
            "gradient_norm": gradient_norm,
            "update_delta_norm": update_delta,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "base_trainable_parameters": base_trainable,
            "lora_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        })
        pseudo_rows.append({
            "variant": variant,
            "order": order,
            "step": step,
            "round": (step - 1) // len(train_samples) + 1,
            "case": sample["case"],
            "patch_index": sample["patch_index"],
            "reliability_source": source,
            "pseudo_positive_voxels": int(np.count_nonzero(sample["pseudo"].numpy())),
            "weight_sum": float(sample["weights"][source].sum()),
            "weight_mean": float(sample["weights"][source].mean()),
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
        del result, logits, per_voxel, loss, image, embedding_batch, pseudo, weight, before
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
    return loss_rows, pseudo_rows, full_rows, {"stats": stats, "debug_rows": debug_rows}


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V7.1b protocol consolidation and short LoRA stability training")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt")
    parser.add_argument("--output-dir", default="outputs/v7_1b_protocol_consolidation")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--train-cases", type=int, default=4)
    parser.add_argument("--evaluation-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--training-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V7.1b requires CUDA, resolved {device}")
    if args.training_rounds != 5:
        raise AssertionError("V7.1b is fixed to five training rounds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    train_cases = iter_image_cases(paths, "train", args.train_cases)
    eval_cases = iter_cases(paths, "test", args.evaluation_cases)
    train_names = [case.case for case in train_cases]
    eval_names = [case.case for case in eval_cases]
    overlap = sorted(set(train_names) & set(eval_names))
    if overlap:
        raise AssertionError(f"train/evaluation case overlap: {overlap}")

    teacher = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
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
    train_cache = build_train_cache_with_slicers(
        teacher, world_model, train_cases, prompt_embedding, args, device,
    )
    eval_cache = build_evaluation_cache(
        teacher, world_model, eval_cases, prompt_embedding, args, device,
    )
    train_samples = []
    for case in train_cases:
        candidates = [
            sample for sample in train_cache
            if sample["case"] == case.case and sample["patch_kind"] == "foreground" and sample["has_foreground"]
        ]
        if not candidates:
            raise RuntimeError(f"No nonempty teacher foreground patch for {case.case}")
        train_samples.append(sorted(candidates, key=lambda sample: sample["patch_index"])[0])
    train_samples.sort(key=lambda sample: (sample["case_index"], sample["patch_index"]))
    manifest_path = output_dir / "train_patch_manifest.json"
    manifest_path.write_text(json.dumps(make_manifest(train_samples, args), indent=2))
    manifest, train_samples = load_manifest_samples(manifest_path, train_cache)

    full_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in eval_cases:
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
    target_names: list[str] | None = None
    base_network.to(device)
    initial_full_rows = evaluate_full_volume(
        teacher, base_network, eval_cases, full_data, prompt_embedding,
        "A_init_no_adaptation", "forward", 0, args.label_value, args.prediction_threshold,
    )
    initial_debug_rows = evaluate_network(
        "A_init_no_adaptation", "forward", 0, base_network, eval_cache,
        teacher, device, args.prediction_threshold,
    )
    base_network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    loss_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    full_rows = list(initial_full_rows)
    debug_rows = list(initial_debug_rows)
    parameter_stats = []
    for variant, source in VARIANTS.items():
        print(f"[V7.1b] forward {variant}", flush=True)
        losses, pseudo, forward_full, details = train_variant(
            variant, source, "forward", base_network, train_samples, eval_cases,
            full_data, eval_cache, teacher, prompt_embedding, args, device,
            base_total, target_names, True,
        )
        target_names = details["stats"]["target_modules"] if target_names is None else target_names
        loss_rows.extend(losses); pseudo_rows.extend(pseudo); full_rows.extend(forward_full)
        debug_rows.extend(details["debug_rows"])
        parameter_stats.append({"variant": variant, "order": "forward", **details["stats"]})
        print(f"[V7.1b] reverse {variant}", flush=True)
        losses, pseudo, reverse_full, details = train_variant(
            variant, source, "reverse", base_network, train_samples, eval_cases,
            full_data, eval_cache, teacher, prompt_embedding, args, device,
            base_total, target_names, False,
        )
        loss_rows.extend(losses); pseudo_rows.extend(pseudo); full_rows.extend(reverse_full)
        debug_rows.extend(details["debug_rows"])
        parameter_stats.append({"variant": variant, "order": "reverse", **details["stats"]})

    by_case = full_rows
    curve = pool_full_volume(full_rows)
    initial_by_case = {row["case"]: row for row in initial_full_rows}
    for row in by_case:
        if row["variant"] != "A_init_no_adaptation":
            baseline = initial_by_case[row["case"]]
            row["delta_dice_vs_A_init"] = row["dice"] - baseline["dice"]
            row["delta_foreground_iou_vs_A_init"] = row["foreground_iou"] - baseline["foreground_iou"]
            row["delta_precision_vs_A_init"] = row["precision"] - baseline["precision"]
            row["delta_recall_vs_A_init"] = row["recall"] - baseline["recall"]
    initial_curve = next(row for row in curve if row["variant"] == "A_init_no_adaptation")
    for row in curve:
        if row["variant"] != "A_init_no_adaptation":
            row["delta_mean_dice_vs_A_init"] = row["mean_dice"] - initial_curve["mean_dice"]
            row["delta_mean_foreground_iou_vs_A_init"] = row["mean_foreground_iou"] - initial_curve["mean_foreground_iou"]

    final_curve = {
        (row["variant"], row["order"]): row
        for row in curve
        if int(row["step"]) == 20
    }
    order_sensitivity = {}
    for variant in VARIANTS:
        forward = final_curve[(variant, "forward")]
        reverse = final_curve[(variant, "reverse")]
        order_sensitivity[variant] = {
            "forward_minus_reverse_mean_dice": forward["mean_dice"] - reverse["mean_dice"],
            "forward_minus_reverse_mean_foreground_iou": forward["mean_foreground_iou"] - reverse["mean_foreground_iou"],
            "forward_minus_reverse_mean_precision": forward["mean_precision"] - reverse["mean_precision"],
            "forward_minus_reverse_mean_recall": forward["mean_recall"] - reverse["mean_recall"],
        }
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    peak_allocated = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else 0.0
    write_csv(output_dir / "training_loss.csv", loss_rows)
    write_csv(output_dir / "full_volume_by_case.csv", by_case)
    write_csv(output_dir / "full_volume_curve.csv", curve)
    write_csv(output_dir / "sampled_patch_debug.csv", debug_rows)
    (output_dir / "parameter_stats.json").write_text(json.dumps(parameter_stats, indent=2))
    summary = {
        "stage": "V7.1b protocol consolidation + short LoRA stability training",
        "development_evaluation_note": "The four evaluation cases are development evaluation cases used during method development, not sealed final test cases.",
        "source_checkpoint": str(checkpoint_path),
        "selected_stage": args.selected_stage,
        "seed": args.seed,
        "resolved_device": str(device),
        "gpu_name": gpu_name,
        "peak_cuda_allocated_mb": peak_allocated,
        "peak_cuda_reserved_mb": peak_reserved,
        "adaptation_cases": train_names,
        "evaluation_cases": eval_names,
        "case_overlap": overlap,
        "case_overlap_count": len(overlap),
        "train_manifest": manifest,
        "fixed_manifest_reused_for_all_variants_and_rounds": True,
        "foreground_candidate_patches": args.foreground_candidate_patches,
        "effective_updates_per_variant_order": len(train_samples) * args.training_rounds,
        "full_volume_eval_steps_forward": [0, *EVAL_STEPS],
        "reverse_confirmation_step": 20,
        "variants": VARIANTS,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_names,
            "base_trainable_parameters": 0,
            "parameter_stats": parameter_stats,
        },
        "training": {
            "rounds": args.training_rounds,
            "updates_per_round": len(train_samples),
            "total_updates": len(train_samples) * args.training_rounds,
            "loss": "V7.0d weighted pseudo-label BCE",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "student_view": "fixed gamma(+0.30)/blur(1.5) sequence from manifest",
            "teacher_gt_used_for_selection": False,
            "world_predictor_updated": False,
            "base_voxtell_parameters_frozen": True,
        },
        "full_volume_inference": {
            "implementation": "VoxTell predictor.preprocess + predict_sliding_window_return_logits + original bbox restoration",
            "overlap_aggregation": "VoxTell native Gaussian sliding-window aggregation",
            "sampled_patch_metrics_are_debug_only": True,
        },
        "final_step20_full_volume": [
            row for row in curve if int(row["step"]) == 20
        ],
        "order_sensitivity": order_sensitivity,
        "outputs": {name: str(output_dir / name) for name in (
            "train_patch_manifest.json", "training_loss.csv", "full_volume_by_case.csv",
            "full_volume_curve.csv", "sampled_patch_debug.csv", "parameter_stats.json", "summary.json",
        )},
        "status": "complete; stopped after fixed 20-update forward plus reverse confirmation",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
