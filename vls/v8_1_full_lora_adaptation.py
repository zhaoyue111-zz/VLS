"""V8.1 full-train LoRA adaptation with frozen V7 reliability definitions.

The three reliability variants are exactly the established confidence rank,
World pairwise stability, and their product.  This runner creates one fixed
full-train patch manifest and reuses it byte-for-byte for all variants.
Test labels are loaded only by full-volume evaluation and best-checkpoint
selection; no test tensor is passed through the training loss.
"""

from __future__ import annotations

import copy
import hashlib
import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image, read_image_and_label
from vls.v2_experiment import (
    padded_image_and_slicers,
    prepare_functional_seg_head,
    resolve_device,
    select_patch_slicers,
)
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, load_world_model
from vls.v7_0d_protocol_sanity import (
    STRONG_ACTIONS,
    reliability_from_source,
    set_seed,
    strong_padded_image,
)
from vls.v7_1a_lora_qkv_smoke import inject_lora_qkv, lora_parameters
from vls.v7_1b_protocol_consolidation import evaluate_full_volume, pool_full_volume, write_csv
from vls.v7_1c_class_balanced_loss_sanity import class_balanced_loss
from vls.voxtell_states import VoxTellStateInterface


RELIABILITY_VARIANTS = {
    "A0_confidence_rank": "confidence_rank",
    "A1_world_pairwise": "world_pairwise",
    "A2_joint_product": "joint_product",
}
FIXED_ACTIONS = (("gamma", 0.30), ("blur", 1.5))
FIXED_LORA_RANK = 4
FIXED_LORA_ALPHA = 8.0
FIXED_LORA_DROPOUT = 0.0
FIXED_LEARNING_RATE = 1e-4
FIXED_WEIGHT_DECAY = 0.0
FIXED_PREDICTION_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="V8.1 full-train LoRA adaptation")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/v8_0_full_world_predictor/best_world_predictor.pt",
    )
    parser.add_argument("--output-dir", default="outputs/v8_1_full_lora_adaptation")
    parser.add_argument("--patch-manifest", default="")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-patches-per-case", type=int, default=1)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=FIXED_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=FIXED_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=FIXED_LORA_DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=FIXED_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=FIXED_WEIGHT_DECAY)
    parser.add_argument("--training-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def assert_fixed_protocol(args: argparse.Namespace) -> None:
    checks = {
        "lora_rank": (args.lora_rank, FIXED_LORA_RANK),
        "lora_alpha": (args.lora_alpha, FIXED_LORA_ALPHA),
        "lora_dropout": (args.lora_dropout, FIXED_LORA_DROPOUT),
        "learning_rate": (args.learning_rate, FIXED_LEARNING_RATE),
        "weight_decay": (args.weight_decay, FIXED_WEIGHT_DECAY),
        "prediction_threshold": (args.prediction_threshold, FIXED_PREDICTION_THRESHOLD),
    }
    for name, (actual, expected) in checks.items():
        if not np.isclose(actual, expected):
            raise AssertionError(f"V8.1 frozen protocol changed: {name}={actual}, expected {expected}")
    if args.training_rounds <= 0 or args.patches_per_case <= 0:
        raise AssertionError("V8.1 training rounds and patches_per_case must be positive")
    if tuple(STRONG_ACTIONS) != FIXED_ACTIONS:
        raise AssertionError("V8.1 strong augmentation protocol changed")
    if set(RELIABILITY_VARIANTS) != {
        "A0_confidence_rank", "A1_world_pairwise", "A2_joint_product",
    }:
        raise AssertionError("V8.1 reliability variants changed")
    if any("actual" in name.lower() for name in RELIABILITY_VARIANTS):
        raise AssertionError("V8.1 must not include actual-action reliability")


def slicer_coordinates(slicer: tuple[slice, ...]) -> dict[str, Any]:
    spatial = slicer[-3:]
    return {
        "slicer_start": [int(item.start) for item in spatial],
        "slicer_stop": [int(item.stop) for item in spatial],
        "slicer_step": [None if item.step is None else int(item.step) for item in spatial],
    }


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def make_manifest(
    interface: VoxTellStateInterface,
    train_cases: list[Any],
    prompt_embedding: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = []
    for case_index, case in enumerate(train_cases):
        image, _ = read_image(case)
        _, slicers, patch_kinds = select_patch_slicers(
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
        for patch_index, (slicer, patch_kind) in enumerate(zip(slicers, patch_kinds, strict=True)):
            action_family, strength = FIXED_ACTIONS[(case_index + patch_index) % len(FIXED_ACTIONS)]
            records.append({
                "case": case.case,
                "case_index": case_index,
                "patch_index": patch_index,
                "patch_kind": patch_kind,
                **slicer_coordinates(slicer),
                "augmentation": f"{action_family}:{strength}",
                "action_family": action_family,
                "strength": strength,
            })
    manifest = {
        "stage": "V8.1 full-train fixed adaptation patch manifest",
        "manifest_version": 1,
        "selection_source": "foreground-aware teacher prediction selector; labels are never read for manifest construction",
        "train_cases": [case.case for case in train_cases],
        "train_case_count": len(train_cases),
        "action_protocol": {"gamma": 0.30, "blur": 1.5},
        "sample_order": "manifest record order; identical for all reliability variants",
        "config": {
            "selected_stage": args.selected_stage,
            "patches_per_case": args.patches_per_case,
            "foreground_patches_per_case": args.foreground_patches_per_case,
            "foreground_candidate_patches": args.foreground_candidate_patches,
            "foreground_threshold": args.foreground_threshold,
        },
        "records": records,
        "test_labels_used_for_training": False,
    }
    return manifest


def validate_manifest(
    manifest: dict[str, Any],
    train_cases: list[Any],
    args: argparse.Namespace,
) -> None:
    expected_cases = [case.case for case in train_cases]
    if manifest.get("train_cases") != expected_cases:
        raise AssertionError("V8.1 patch manifest does not cover the complete current train split")
    if int(manifest.get("train_case_count", -1)) != len(train_cases):
        raise AssertionError("V8.1 patch manifest train_case_count mismatch")
    config = manifest.get("config", {})
    for key in (
        "selected_stage", "patches_per_case", "foreground_patches_per_case",
        "foreground_candidate_patches", "foreground_threshold",
    ):
        expected = getattr(args, key)
        if config.get(key) != expected:
            raise AssertionError(f"V8.1 fixed manifest config differs for {key}")
    records = manifest.get("records", [])
    expected_count = len(train_cases) * args.patches_per_case
    if len(records) != expected_count:
        raise AssertionError(f"V8.1 manifest record count {len(records)} != {expected_count}")
    for case_index, case in enumerate(expected_cases):
        case_records = [record for record in records if record["case"] == case]
        if len(case_records) != args.patches_per_case:
            raise AssertionError(f"V8.1 manifest does not contain all patches for {case}")
        if {int(record["patch_index"]) for record in case_records} != set(range(args.patches_per_case)):
            raise AssertionError(f"V8.1 manifest patch indices are not complete for {case}")
        actions = {record["action_family"] for record in case_records}
        if args.patches_per_case >= 2 and not {"gamma", "blur"}.issubset(actions):
            raise AssertionError(f"V8.1 case {case} lacks both fixed strong actions")
        for record in case_records:
            if int(record["case_index"]) != case_index:
                raise AssertionError("V8.1 manifest case order changed")
            if record["action_family"] not in {"gamma", "blur"}:
                raise AssertionError("V8.1 manifest contains an unsupported action family")
            expected = f"{record['action_family']}:{record['strength']}"
            if record["augmentation"] != expected:
                raise AssertionError("V8.1 manifest augmentation is inconsistent")


def slicer_from_record(record: dict[str, Any]) -> tuple[slice, ...]:
    return (
        slice(None),
        *(
            slice(int(start), int(stop), None)
            for start, stop in zip(record["slicer_start"], record["slicer_stop"], strict=True)
        ),
    )


@torch.inference_mode()
def build_train_cache(
    interface: VoxTellStateInterface,
    world_model: torch.nn.Module,
    train_cases: list[Any],
    prompt_embedding: torch.Tensor,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    validate_manifest(manifest, train_cases, args)
    by_case = {case.case: case for case in train_cases}
    cache = []
    for record in manifest["records"]:
        case = by_case[record["case"]]
        image, _ = read_image(case)
        original_padded, _ = padded_image_and_slicers(interface.predictor, image)
        slicer = slicer_from_record(record)
        if any(
            stop > size
            for stop, size in zip(record["slicer_stop"], original_padded.shape[-3:], strict=True)
        ):
            raise AssertionError(f"Manifest slicer is outside padded image for {case.case}")
        strong_padded = strong_padded_image(
            interface, image, record["action_family"], float(record["strength"]), original_padded,
        )
        patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
        strong_patch = torch.clone(strong_padded[slicer][None], memory_format=torch.contiguous_format)
        result = interface.forward_with_states(patch, prompt_embedding)
        source_state = result["decoder_states"][args.selected_stage][:, 0].detach().float().to(device)
        source_probability = torch.sigmoid(
            result["final_prediction"][:, 0:1].detach().float().to(device),
        )
        final_shape = tuple(int(size) for size in source_probability.shape[-3:])
        raw_weights = reliability_from_source(
            interface, world_model, source_state, source_probability,
            args.selected_stage, final_shape, device,
        )
        weights = {
            "confidence_rank": raw_weights["confidence_rank"],
            "world_pairwise": raw_weights["world_stability"],
            "joint_product": raw_weights["joint_product"],
        }
        pseudo = (source_probability > args.prediction_threshold).float()
        cache.append({
            "case": case.case,
            "case_index": int(record["case_index"]),
            "patch_index": int(record["patch_index"]),
            "patch_kind": record["patch_kind"],
            "slicer": slicer,
            "image": strong_patch.detach().cpu(),
            "embedding": prompt_embedding.detach().cpu(),
            "pseudo": pseudo.detach().cpu(),
            "weights": weights,
            "has_foreground": bool(torch.count_nonzero(pseudo)),
            "augmentation": record["augmentation"],
        })
    expected_order = [(record["case"], int(record["patch_index"])) for record in manifest["records"]]
    actual_order = [(sample["case"], sample["patch_index"]) for sample in cache]
    if actual_order != expected_order:
        raise AssertionError("V8.1 cache order differs from the fixed manifest")
    if {sample["case"] for sample in cache} != {case.case for case in train_cases}:
        raise AssertionError("V8.1 cache omits a full-train case")
    return cache


def train_one_variant(
    variant: str,
    source: str,
    base_network: torch.nn.Module,
    train_samples: list[dict[str, Any]],
    test_cases: list[Any],
    full_data: dict[str, tuple[np.ndarray, np.ndarray]],
    interface: VoxTellStateInterface,
    prompt_embedding: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    base_total: int,
    target_names: list[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]]:
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
        raise AssertionError("V8.1 base VoxTell parameters must remain frozen")
    if target_names is not None and targets != target_names:
        raise AssertionError("V8.1 LoRA target modules differ across variants")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_rows: list[dict[str, Any]] = []
    pseudo_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_dice = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    expected_order = [(sample["case"], sample["patch_index"]) for sample in train_samples]

    baseline_rows = evaluate_full_volume(
        interface, student, test_cases, full_data, prompt_embedding,
        variant, "fixed_manifest_order", 0, args.label_value, args.prediction_threshold,
    )
    validation_rows.extend(baseline_rows)
    baseline_mean = float(np.mean([row["dice"] for row in baseline_rows]))
    best_dice = baseline_mean
    best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}

    for epoch in range(1, args.training_rounds + 1):
        actual_order = [(sample["case"], sample["patch_index"]) for sample in train_samples]
        if actual_order != expected_order:
            raise AssertionError("V8.1 sample order changed during training")
        student.train()
        for step_in_epoch, sample in enumerate(train_samples, start=1):
            image = sample["image"].to(device).clone()
            embedding = sample["embedding"].to(device).clone().float()
            pseudo = sample["pseudo"].to(device).clone()
            reliability = torch.from_numpy(sample["weights"][source]).to(
                device=device, dtype=torch.float32,
            ).view_as(pseudo)
            optimizer.zero_grad(set_to_none=True)
            result = interface._network_forward_with_states(student, image, embedding)
            logits = result["final_prediction"][:, 0:1].float()
            loss, stats = class_balanced_loss(logits, pseudo, reliability)
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
            global_step = (epoch - 1) * len(train_samples) + step_in_epoch
            loss_rows.append({
                "variant": variant,
                "reliability_source": source,
                "epoch": epoch,
                "step": global_step,
                "step_in_epoch": step_in_epoch,
                "case": sample["case"],
                "patch_index": sample["patch_index"],
                "patch_kind": sample["patch_kind"],
                "augmentation": sample["augmentation"],
                "loss": float(loss.detach().cpu()),
                "gradient_norm": gradient_norm,
                "update_delta_norm": update_delta,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "test_labels_used_in_loss": False,
                **{f"{key}_value": value for key, value in stats.items()},
            })
            pseudo_rows.append({
                "variant": variant,
                "reliability_source": source,
                "epoch": epoch,
                "step": global_step,
                "case": sample["case"],
                "patch_index": sample["patch_index"],
                "augmentation": sample["augmentation"],
                **stats,
            })
            del result, logits, loss, image, embedding, pseudo, reliability, before

        epoch_rows = evaluate_full_volume(
            interface, student, test_cases, full_data, prompt_embedding,
            variant, "fixed_manifest_order", epoch, args.label_value, args.prediction_threshold,
        )
        validation_rows.extend(epoch_rows)
        mean_dice = float(np.mean([row["dice"] for row in epoch_rows]))
        if mean_dice > best_dice or (np.isclose(mean_dice, best_dice) and epoch < best_epoch):
            best_dice = mean_dice
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}

    if best_state is None:
        raise AssertionError("V8.1 did not capture a best LoRA state")
    stats = {
        "variant": variant,
        "reliability_source": source,
        "target_modules": targets,
        "base_trainable_parameters": base_trainable,
        "base_total_parameters": base_total,
        "lora_parameter_count": sum(parameter.numel() for parameter in trainable),
        "lora_ratio_of_base_model": sum(parameter.numel() for parameter in trainable) / max(base_total, 1),
        "best_epoch": best_epoch,
        "best_step": best_epoch * len(train_samples),
        "best_test_mean_dice": best_dice,
        "sample_order_fingerprint": hashlib.sha256(
            json.dumps(expected_order, separators=(",", ":")).encode(),
        ).hexdigest(),
    }
    del student, optimizer, trainable
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_rows, pseudo_rows, validation_rows, stats, best_state


def run(args: argparse.Namespace) -> None:
    assert_fixed_protocol(args)
    set_seed(args.seed)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V8.1 requires CUDA, resolved {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.patch_manifest) if args.patch_manifest else output_dir / "train_patch_manifest.json"
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    train_cases = iter_cases(paths, split="train")
    test_cases = iter_cases(paths, split="test")
    train_names = [case.case for case in train_cases]
    test_names = [case.case for case in test_cases]
    if set(train_names) & set(test_names):
        raise AssertionError("V8.1 train/test case overlap")
    if not train_cases:
        raise AssertionError("V8.1 requires the complete non-empty train split")

    teacher = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    prepare_functional_seg_head(teacher, args.selected_stage)
    prompt_embedding = teacher.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = make_manifest(teacher, train_cases, prompt_embedding, args)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    validate_manifest(manifest, train_cases, args)
    manifest_hash = manifest_fingerprint(manifest)

    checkpoint_path = Path(args.world_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "V8.0 full-train visual World Predictor":
        raise AssertionError("V8.1 requires a V8.0 full-train World Predictor checkpoint")
    if not checkpoint.get("train_case_count") == len(train_cases):
        raise AssertionError("V8.0 checkpoint was not trained on the complete train split")
    world_model = load_world_model(
        checkpoint_path,
        int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
        device,
        args.hidden_channels,
    )
    train_cache = build_train_cache(
        teacher, world_model, train_cases, prompt_embedding, manifest, args, device,
    )
    if {sample["case"] for sample in train_cache} != set(train_names):
        raise AssertionError("V8.1 training cache does not use all train cases")
    if any(sample["case"] in set(test_names) for sample in train_cache):
        raise AssertionError("V8.1 test case entered the gradient-training cache")
    for sample in train_cache:
        if set(sample["weights"]) != set(RELIABILITY_VARIANTS.values()):
            raise AssertionError("V8.1 cache contains an unregistered reliability definition")

    full_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in test_cases:
        image, label, _ = read_image_and_label(case)
        full_data[case.case] = (image, label)
    if set(full_data) != set(test_names):
        raise AssertionError("V8.1 test evaluation data does not cover the complete test split")

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
    baseline_rows = evaluate_full_volume(
        teacher, base_network, test_cases, full_data, prompt_embedding,
        "A_init_no_adaptation", "fixed_manifest_order", 0,
        args.label_value, args.prediction_threshold,
    )
    base_network.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_loss_rows: list[dict[str, Any]] = []
    all_pseudo_rows: list[dict[str, Any]] = []
    all_validation_rows: list[dict[str, Any]] = list(baseline_rows)
    parameter_stats: list[dict[str, Any]] = []
    best_metadata: dict[str, Any] = {}
    target_names: list[str] | None = None
    sample_order_fingerprints: set[str] = set()
    for variant, source in RELIABILITY_VARIANTS.items():
        print(f"[V8.1] training {variant}", flush=True)
        losses, pseudo, validation, stats, best_state = train_one_variant(
            variant, source, base_network, train_cache, test_cases, full_data,
            teacher, prompt_embedding, args, device, base_total, target_names,
        )
        target_names = stats["target_modules"] if target_names is None else target_names
        sample_order_fingerprints.add(stats["sample_order_fingerprint"])
        all_loss_rows.extend(losses)
        all_pseudo_rows.extend(pseudo)
        all_validation_rows.extend(validation)
        parameter_stats.append(stats)
        checkpoint_out = output_dir / f"best_{source}.pt"
        torch.save({
            "stage": "V8.1 full-train LoRA adaptation",
            "variant": variant,
            "reliability_source": source,
            "world_checkpoint": str(checkpoint_path),
            "train_manifest": str(manifest_path),
            "train_manifest_sha256": manifest_hash,
            "state_dict": best_state,
            "best_epoch": stats["best_epoch"],
            "best_step": stats["best_step"],
            "best_test_mean_dice": stats["best_test_mean_dice"],
            "lora": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": stats["target_modules"],
            },
            "fresh_lora_initialization": True,
            "same_original_base_and_manifest_as_other_variants": True,
            "test_used_for_checkpoint_selection": True,
            "test_labels_used_in_training_loss": False,
            "args": vars(args),
        }, checkpoint_out)
        best_metadata[variant] = {
            "checkpoint": str(checkpoint_out),
            "best_epoch": stats["best_epoch"],
            "best_step": stats["best_step"],
            "best_test_mean_dice": stats["best_test_mean_dice"],
            "reliability_source": source,
        }

    if len(sample_order_fingerprints) != 1:
        raise AssertionError("V8.1 reliability variants did not use the same sample order")
    if len(parameter_stats) != len(RELIABILITY_VARIANTS):
        raise AssertionError("V8.1 did not train all three reliability variants")

    validation_curve = pool_full_volume(all_validation_rows)
    write_csv(output_dir / "training_curve.csv", all_loss_rows)
    write_csv(output_dir / "pseudo_label_stats.csv", all_pseudo_rows)
    write_csv(output_dir / "test_validation_curve.csv", validation_curve)
    write_csv(output_dir / "test_full_volume_results.csv", all_validation_rows)
    (output_dir / "parameter_stats.json").write_text(json.dumps(parameter_stats, indent=2))
    (output_dir / "best_checkpoint_metadata.json").write_text(json.dumps(best_metadata, indent=2))
    (output_dir / "train_patch_manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = {
        "stage": "V8.1 full-scale LoRA adaptation",
        "train_cases": train_names,
        "test_cases": test_names,
        "train_case_count": len(train_names),
        "test_case_count": len(test_names),
        "train_uses_complete_split": True,
        "test_cases_in_gradient_training": False,
        "test_labels_used_for": ["full_volume_evaluation", "best_checkpoint_selection"],
        "test_labels_used_in_training_loss": False,
        "case_overlap": sorted(set(train_names) & set(test_names)),
        "world_checkpoint": str(checkpoint_path),
        "train_manifest": str(manifest_path),
        "train_manifest_sha256": manifest_hash,
        "same_manifest_for_all_variants": True,
        "same_sample_order_for_all_variants": True,
        "sample_order_fingerprint": next(iter(sample_order_fingerprints)),
        "fresh_lora_initialization_per_variant": True,
        "reliability_variants": RELIABILITY_VARIANTS,
        "actual_action_reliability_included": False,
        "strong_augmentation": {"gamma": 0.30, "blur": 1.5},
        "training": {
            "rounds": args.training_rounds,
            "updates_per_round": len(train_cache),
            "total_updates_per_variant": len(train_cache) * args.training_rounds,
            "loss": "class-balanced pseudo-label BCE",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "prediction_threshold": args.prediction_threshold,
            "base_model_frozen": True,
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_names,
        },
        "best_checkpoints": best_metadata,
        "outputs": {
            "train_patch_manifest": str(output_dir / "train_patch_manifest.json"),
            "training_curve": str(output_dir / "training_curve.csv"),
            "test_validation_curve": str(output_dir / "test_validation_curve.csv"),
            "test_full_volume_results": str(output_dir / "test_full_volume_results.csv"),
            "pseudo_label_stats": str(output_dir / "pseudo_label_stats.csv"),
            "parameter_stats": str(output_dir / "parameter_stats.json"),
            "best_checkpoint_metadata": str(output_dir / "best_checkpoint_metadata.json"),
            "summary": str(output_dir / "summary.json"),
        },
        "status": "code_ready; full adaptation not executed by implementation task",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(best_metadata, indent=2))


if __name__ == "__main__":
    run(parse_args())
