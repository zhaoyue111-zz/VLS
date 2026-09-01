"""Formal Language WP on VoxTell's pre-projection shared prompt output.

VoxTell is frozen.  The trainable module predicts the residual between the
source shared prompt output and each target shared prompt output.  The only
training loss is normalized MSE on that shared output.  Test-time wrong text
actions and full-volume segmentation are evaluated separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Permit both ``python -m`` and the direct project command used in the saved run.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import binary_gt_from_label, iter_cases, read_image, read_image_and_label
from vls.metrics import binary_metrics
from vls.v2_experiment import resolve_device, select_patch_slicers
from vls.v9_0_world_state_selection_audit import pad_selected_patches
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import normalized_mse


SOURCE_PROMPT = "liver"
TARGET_PROMPTS = ("the liver", "human liver", "hepatic organ")
WRONG_PROMPTS = ("spleen", "left kidney", "right kidney")
ALL_PROMPTS = (SOURCE_PROMPT, *TARGET_PROMPTS, *WRONG_PROMPTS)
ACTION_NAMES = ("source_identity", "correct_action", "wrong_spleen", "wrong_left_kidney", "wrong_right_kidney")


class LanguageResidualWP(nn.Module):
    """Predict ``M_s + WP(M_s, delta_e)`` with zero residual initialization."""

    def __init__(self, shared_dim: int, text_delta_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.source_projection = nn.Linear(shared_dim, hidden_dim)
        self.delta_projection = nn.Linear(text_delta_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shared_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, source_shared: torch.Tensor, text_delta: torch.Tensor) -> torch.Tensor:
        if source_shared.ndim != 2 or text_delta.ndim != 2:
            raise ValueError("LanguageResidualWP expects [batch, feature] tensors")
        return source_shared + self.fusion(
            self.source_projection(source_shared) + self.delta_projection(text_delta)
        )


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description="Formal pre-projection shared-output Language WP")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--output-dir", default="outputs/v10_language_wp_shared_output")
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0, help="debug only; 0 uses all formal cases")
    parser.add_argument("--patch-limit", type=int, default=0, help="debug only; 0 uses all selected patches")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def flatten_prompt_embedding(embedding: torch.Tensor, index: int) -> torch.Tensor:
    if embedding.ndim not in (3, 4):
        raise ValueError(f"Unexpected prompt embedding shape: {tuple(embedding.shape)}")
    return embedding[:, index].detach().float().flatten()


def serialize_slicer(slicer: tuple) -> list[list[int | None]]:
    return [[item.start, item.stop, item.step] for item in slicer]


def deserialize_slicer(value: list[list[int | None]]) -> tuple[slice, ...]:
    return tuple(slice(start, stop, step) for start, stop, step in value)


def build_manifest(interface: VoxTellStateInterface, cases: list[Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    source_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    for case in cases:
        image, _ = read_image(case)
        _, slicers, kinds = select_patch_slicers(
            interface, image, [SOURCE_PROMPT], args.patches_per_case,
            args.foreground_patches_per_case, args.foreground_candidate_patches,
            args.foreground_threshold,
        )
        slicers, kinds = pad_selected_patches(slicers, kinds, args.patches_per_case)
        if args.patch_limit:
            slicers, kinds = slicers[:args.patch_limit], kinds[:args.patch_limit]
        for patch_index, (slicer, kind) in enumerate(zip(slicers, kinds, strict=True)):
            records.append({
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": kind,
                "slicer": serialize_slicer(slicer),
            })
    return records


def grouped_manifest(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["case"])].append(record)
    return grouped


@torch.inference_mode()
def cache_shared_outputs(
    interface: VoxTellStateInterface,
    cases: list[Any],
    manifest: list[dict[str, Any]],
    prompt_embeddings: torch.Tensor,
    text_deltas: torch.Tensor,
) -> dict[str, Any]:
    case_lookup = {case.case: case for case in cases}
    grouped = grouped_manifest(manifest)
    source_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    delta_values: list[torch.Tensor] = []
    case_values: list[str] = []
    patch_values: list[int] = []
    kind_values: list[str] = []
    target_values_names: list[str] = []
    for case_name, case_records in grouped.items():
        image, _ = read_image(case_lookup[case_name])
        padded, _, _ = interface.predictor.preprocess(image)
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        padded, _ = pad_nd_image(padded, interface.predictor.patch_size, "constant", {"value": 0}, True, None)
        for record in case_records:
            patch = torch.clone(padded[deserialize_slicer(record["slicer"])][None], memory_format=torch.contiguous_format)
            source_result = interface.forward_with_audit_context(patch, prompt_embeddings[:, :1])
            source_shared = source_result["shared_prompt_outputs"].float()
            if source_shared.ndim != 3 or source_shared.shape[1] != 1:
                raise AssertionError(f"Unexpected source shared output shape: {tuple(source_shared.shape)}")
            source = source_shared[:, 0].cpu()
            for target_index, target_name in enumerate(TARGET_PROMPTS, start=1):
                target_result = interface.forward_with_audit_context(patch, prompt_embeddings[:, target_index:target_index + 1])
                target_shared = target_result["shared_prompt_outputs"].float()
                if target_shared.ndim != 3 or target_shared.shape[1] != 1:
                    raise AssertionError(f"Unexpected target shared output shape: {tuple(target_shared.shape)}")
                source_values.append(source)
                target_values.append(target_shared[:, 0].cpu())
                delta_values.append(text_deltas[target_index - 1].cpu()[None])
                case_values.append(case_name)
                patch_values.append(int(record["patch_index"]))
                kind_values.append(str(record["patch_kind"]))
                target_values_names.append(target_name)
    return {
        "source": torch.cat(source_values, dim=0),
        "target": torch.cat(target_values, dim=0),
        "delta": torch.cat(delta_values, dim=0),
        "case": case_values,
        "patch_index": patch_values,
        "patch_kind": kind_values,
        "target_prompt": target_values_names,
    }


def train_wp(model: LanguageResidualWP, data: dict[str, Any], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    model.train()
    source = data["source"].to(device)
    target = data["target"].to(device)
    delta = data["delta"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rows = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(source.shape[0], generator=generator)
        losses = []
        for batch_indices in order.split(args.batch_size):
            indices = batch_indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source.index_select(0, indices), delta.index_select(0, indices))
            loss = normalized_mse(prediction, target.index_select(0, indices))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        rows.append({"epoch": epoch, "split": "train", "samples": int(source.shape[0]), "nmse": float(np.mean(losses))})
    return rows


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def transition_metrics(source: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float | None]:
    source = source.float().reshape(-1)
    target = target.float().reshape(-1)
    prediction = prediction.float().reshape(-1)
    true_delta = target - source
    predicted_delta = prediction - source
    source_norm = torch.linalg.vector_norm(source)
    true_norm = torch.linalg.vector_norm(true_delta)
    predicted_norm = torch.linalg.vector_norm(predicted_delta)
    cosine_denominator = true_norm * predicted_norm
    cosine = None if scalar(cosine_denominator) <= 1e-12 else scalar(torch.dot(true_delta, predicted_delta) / cosine_denominator)
    return {
        "real_change_rate": None if scalar(source_norm) <= 1e-12 else scalar(true_norm / source_norm),
        "predicted_change_rate": None if scalar(source_norm) <= 1e-12 else scalar(predicted_norm / source_norm),
        "change_direction_cosine": cosine,
        "relative_target_nmse": scalar(normalized_mse(prediction, target)),
    }


def evaluate_patch_rows(
    model: LanguageResidualWP,
    data: dict[str, Any],
    wrong_deltas: dict[str, torch.Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for index in range(data["source"].shape[0]):
            source = data["source"][index].to(device)
            target = data["target"][index].to(device)
            correct_delta = data["delta"][index].to(device)
            predictions = {"source_identity": source}
            predictions["correct_action"] = model(source[None], correct_delta[None])[0]
            for name, delta in wrong_deltas.items():
                predictions[name] = model(source[None], delta.to(device)[None])[0]
            for action in ACTION_NAMES:
                metrics = transition_metrics(source, target, predictions[action])
                rows.append({
                    "case": data["case"][index],
                    "patch_index": data["patch_index"][index],
                    "patch_kind": data["patch_kind"][index],
                    "target_prompt": data["target_prompt"][index],
                    "action": action,
                    **metrics,
                })
    return rows


def mean_or_none(values: list[Any]) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not values else float(np.mean(values))


def aggregate_patch_rows(rows: list[dict[str, Any]], by: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in by)].append(row)
    output = []
    metric_names = ("real_change_rate", "predicted_change_rate", "change_direction_cosine", "relative_target_nmse")
    for key, group in sorted(groups.items()):
        item = {field: value for field, value in zip(by, key, strict=True)}
        item["n_patches"] = len(group)
        for metric in metric_names:
            item[metric] = mean_or_none([row[metric] for row in group])
        output.append(item)
    return output


def add_improvements(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in case_rows:
        grouped[(str(row["case"]), str(row["target_prompt"]))][str(row["action"])] = row
    output = []
    for row in case_rows:
        item = dict(row)
        if item["action"] == "correct_action":
            identity = grouped[(str(row["case"]), str(row["target_prompt"]))]["source_identity"]
            item["correct_nmse_improvement_vs_source_pct"] = 100.0 * (identity["relative_target_nmse"] - row["relative_target_nmse"]) / max(identity["relative_target_nmse"], 1e-12)
            for wrong in ("wrong_spleen", "wrong_left_kidney", "wrong_right_kidney"):
                comparator = grouped[(str(row["case"]), str(row["target_prompt"]))][wrong]
                item[f"correct_nmse_improvement_vs_{wrong}_pct"] = 100.0 * (comparator["relative_target_nmse"] - row["relative_target_nmse"]) / max(comparator["relative_target_nmse"], 1e-12)
        output.append(item)
    return output


def summarize_language_rows(patch_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows = add_improvements(aggregate_patch_rows(patch_rows, ("case", "target_prompt", "action")))
    summary_rows = []
    for target in TARGET_PROMPTS:
        target_rows = [row for row in case_rows if row["target_prompt"] == target]
        for action in ACTION_NAMES:
            rows = [row for row in target_rows if row["action"] == action]
            item = {
                "target_prompt": target,
                "action": action,
                "case_macro_n": len(rows),
                "real_change_rate_case_macro": mean_or_none([row["real_change_rate"] for row in rows]),
                "predicted_change_rate_case_macro": mean_or_none([row["predicted_change_rate"] for row in rows]),
                "change_direction_cosine_case_macro": mean_or_none([row["change_direction_cosine"] for row in rows]),
                "relative_target_nmse_case_macro": mean_or_none([row["relative_target_nmse"] for row in rows]),
            }
            if action == "correct_action":
                item["correct_nmse_improvement_vs_source_pct_case_macro"] = mean_or_none([row.get("correct_nmse_improvement_vs_source_pct") for row in rows])
                item["correct_nmse_improvement_vs_source_win_rate"] = float(np.mean([
                    row["relative_target_nmse"] < next(other["relative_target_nmse"] for other in target_rows if other["case"] == row["case"] and other["action"] == "source_identity")
                    for row in rows
                ])) if rows else None
                for wrong in ("wrong_spleen", "wrong_left_kidney", "wrong_right_kidney"):
                    item[f"correct_nmse_improvement_vs_{wrong}_pct_case_macro"] = mean_or_none([row.get(f"correct_nmse_improvement_vs_{wrong}_pct") for row in rows])
                    item[f"correct_nmse_improvement_vs_{wrong}_win_rate"] = float(np.mean([
                        row["relative_target_nmse"] < next(other["relative_target_nmse"] for other in target_rows if other["case"] == row["case"] and other["action"] == wrong)
                        for row in rows
                    ])) if rows else None
            summary_rows.append(item)
    return case_rows, summary_rows


def decode_from_shared_output(interface: VoxTellStateInterface, source_context: dict[str, Any], shared_output: torch.Tensor) -> torch.Tensor:
    network = interface.network
    mask_embeddings = [projection(shared_output) for projection in network.project_to_decoder_channels]
    decoder = source_context["decoder_audit"]["decoder"]
    output = decoder(source_context["decoder_audit"]["skips"], mask_embeddings)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return output


def restore_volume(results: torch.Tensor, slicer_revert_padding: tuple, original_shape: tuple[int, int, int], bbox: Any) -> np.ndarray:
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
    cropped = results[(slice(None), *slicer_revert_padding[1:])].float().cpu().numpy()
    restored = np.zeros((1, *original_shape), dtype=np.float32)
    insert_crop_into_image(restored, cropped, bbox)
    return restored[0]


@torch.inference_mode()
def full_volume_case(
    interface: VoxTellStateInterface,
    model: LanguageResidualWP,
    case: Any,
    prompt_embeddings: torch.Tensor,
    target_deltas: torch.Tensor,
    device: torch.device,
    label_value: int,
) -> tuple[list[dict[str, Any]], int]:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    image, label, _ = read_image_and_label(case)
    predictor = interface.predictor
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    padded, slicer_revert = pad_nd_image(preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None)
    slicers = predictor._internal_get_sliding_window_slicers(padded.shape[1:])
    gaussian = compute_gaussian(tuple(predictor.patch_size), sigma_scale=1.0 / 8, value_scaling_factor=10, device=torch.device("cpu"))
    accumulators = {
        name: torch.zeros((1, *padded.shape[1:]), dtype=torch.float32, device="cpu")
        for name in ("source", "target_the_liver", "target_human_liver", "target_hepatic_organ", "wp_the_liver", "wp_human_liver", "wp_hepatic_organ")
    }
    counts = torch.zeros(padded.shape[1:], dtype=torch.float32, device="cpu")
    source_embedding = prompt_embeddings[:, :1]
    for tile_slice in slicers:
        patch = torch.clone(padded[tile_slice][None], memory_format=torch.contiguous_format)
        source_context = interface.forward_with_audit_context(patch, source_embedding)
        source_probability = torch.sigmoid(source_context["final_prediction"][:, :1])[0].cpu()
        shared_source = source_context["shared_prompt_outputs"][:, 0]
        for target_index, target_name in enumerate(TARGET_PROMPTS, start=1):
            target_context = interface.forward_with_audit_context(
                patch, prompt_embeddings[:, target_index:target_index + 1],
            )
            target_probability = torch.sigmoid(target_context["final_prediction"][:, :1])[0].cpu()
            predicted_shared = model(shared_source, target_deltas[target_index - 1][None].to(device))
            predicted_logits = decode_from_shared_output(interface, source_context, predicted_shared[:, None])
            predicted_probability = torch.sigmoid(predicted_logits)[0].cpu()
            accumulators[f"target_{target_name.replace(' ', '_')}"][tile_slice] += target_probability * gaussian
            accumulators[f"wp_{target_name.replace(' ', '_')}"][tile_slice] += predicted_probability * gaussian
        accumulators["source"][tile_slice] += source_probability * gaussian
        counts[tile_slice[1:]] += gaussian
    probabilities = {
        name: restore_volume(value / counts.clamp_min(1e-8), slicer_revert, tuple(int(size) for size in original_shape[-3:]), bbox)
        for name, value in accumulators.items()
    }
    gt = binary_gt_from_label(label, label_value).astype(bool)
    rows = []
    for target_name in TARGET_PROMPTS:
        source_metrics = binary_metrics(probabilities["source"] > 0.5, gt)
        target_metrics = binary_metrics(probabilities[f"target_{target_name.replace(' ', '_')}"] > 0.5, gt)
        wp_metrics = binary_metrics(probabilities[f"wp_{target_name.replace(' ', '_')}"] > 0.5, gt)
        rows.append({
            "case": case.case,
            "target_prompt": target_name,
            "source_dice": source_metrics["dice"],
            "source_iou": source_metrics["iou"],
            "real_target_dice": target_metrics["dice"],
            "real_target_iou": target_metrics["iou"],
            "wp_dice": wp_metrics["dice"],
            "wp_iou": wp_metrics["iou"],
            "wp_dice_absolute_improvement": wp_metrics["dice"] - source_metrics["dice"],
            "wp_iou_absolute_improvement": wp_metrics["iou"] - source_metrics["iou"],
            "wp_dice_relative_improvement_pct": 100.0 * (wp_metrics["dice"] - source_metrics["dice"]) / max(source_metrics["dice"], 1e-12),
            "wp_iou_relative_improvement_pct": 100.0 * (wp_metrics["iou"] - source_metrics["iou"]) / max(source_metrics["iou"], 1e-12),
            "sliding_window_tiles": len(slicers),
        })
    return rows, len(slicers)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if args.patches_per_case != 4 or args.foreground_patches_per_case != 2 or args.foreground_candidate_patches != 16:
        raise AssertionError("Formal protocol requires patches=4, foreground=2, candidates=16")
    if args.case_limit < 0 or args.patch_limit < 0:
        raise AssertionError("Debug limits must be non-negative")
    if args.case_limit or args.patch_limit:
        raise AssertionError("This formal experiment does not permit tiny/debug limits")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_command.txt").write_text("python -u " + " ".join(shlex.quote(item) for item in sys.argv) + "\n")
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root), voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root), split_json=Path(args.split_json),
    )
    train_cases = iter_cases(paths, split="train")
    test_cases = iter_cases(paths, split="test")
    if len(train_cases) != 30 or len(test_cases) != 8:
        raise AssertionError(f"Formal split requires 30 train and 8 test cases, got {len(train_cases)} and {len(test_cases)}")
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("Formal run requested CUDA, but CUDA is unavailable")
    interface = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root)
    for parameter in interface.network.parameters():
        parameter.requires_grad_(False)
    interface.network.eval()
    text_backbone = getattr(interface.predictor, "text_backbone", None)
    if isinstance(text_backbone, nn.Module):
        for parameter in text_backbone.parameters():
            parameter.requires_grad_(False)
        text_backbone.eval()
    prompt_embeddings = interface.embed_text_prompts(list(ALL_PROMPTS)).detach().cpu()
    source_text = flatten_prompt_embedding(prompt_embeddings, 0)
    target_deltas = torch.stack([flatten_prompt_embedding(prompt_embeddings, index) - source_text for index in range(1, 4)])
    wrong_deltas = {
        f"wrong_{name.replace(' ', '_')}": flatten_prompt_embedding(prompt_embeddings, index) - source_text
        for index, name in zip(range(4, 7), WRONG_PROMPTS, strict=True)
    }
    train_manifest = build_manifest(interface, train_cases, args)
    test_manifest = build_manifest(interface, test_cases, args)
    (output_dir / "train_patch_manifest.json").write_text(json.dumps(train_manifest, indent=2))
    (output_dir / "test_patch_manifest.json").write_text(json.dumps(test_manifest, indent=2))
    train_data = cache_shared_outputs(interface, train_cases, train_manifest, prompt_embeddings, target_deltas)
    test_data = cache_shared_outputs(interface, test_cases, test_manifest, prompt_embeddings, target_deltas)
    shared_dim = int(train_data["source"].shape[-1])
    text_delta_dim = int(target_deltas.shape[-1])
    model = LanguageResidualWP(shared_dim, text_delta_dim, args.hidden_dim).to(device)
    training_rows = train_wp(model, train_data, args, device)
    checkpoint_path = output_dir / "language_wp_final.pt"
    checkpoint = {
        "stage": "V10 pre-projection shared-output Language WP",
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "shared_output_shape": [1, shared_dim],
        "shared_dim": shared_dim,
        "text_delta_dim": text_delta_dim,
        "hidden_dim": args.hidden_dim,
        "source_prompt": SOURCE_PROMPT,
        "target_prompts": list(TARGET_PROMPTS),
        "wrong_prompts_test_only": list(WRONG_PROMPTS),
        "selected_epoch": args.epochs,
        "train_case_count": len(train_cases),
        "test_case_count": len(test_cases),
        "train_sample_count": int(train_data["source"].shape[0]),
        "voxtell_network_frozen": True,
        "voxtell_text_backbone_frozen": isinstance(getattr(interface.predictor, "text_backbone", None), nn.Module),
        "loss": "NMSE(predicted_shared_output, native_target_shared_output) only",
    }
    torch.save(checkpoint, checkpoint_path)
    write_csv(output_dir / "training_curve.csv", training_rows)
    patch_rows = evaluate_patch_rows(model, test_data, wrong_deltas, device)
    case_rows, language_summary = summarize_language_rows(patch_rows)
    write_csv(output_dir / "shared_output_patch_metrics.csv", patch_rows)
    write_csv(output_dir / "shared_output_case_metrics.csv", case_rows)
    write_csv(output_dir / "table1_shared_output_summary.csv", language_summary)
    full_volume_rows = []
    tile_counts = {}
    for case in test_cases:
        rows, tile_count = full_volume_case(interface, model, case, prompt_embeddings, target_deltas, device, DEFAULT_LABEL_VALUE)
        full_volume_rows.extend(rows)
        tile_counts[case.case] = tile_count
    write_csv(output_dir / "full_volume_case_metrics.csv", full_volume_rows)
    table2_rows = []
    for target in TARGET_PROMPTS:
        rows = [row for row in full_volume_rows if row["target_prompt"] == target]
        table2_rows.append({
            "target_prompt": target,
            "case_macro_n": len(rows),
            "source_dice_case_macro": mean_or_none([row["source_dice"] for row in rows]),
            "source_iou_case_macro": mean_or_none([row["source_iou"] for row in rows]),
            "real_target_dice_case_macro": mean_or_none([row["real_target_dice"] for row in rows]),
            "real_target_iou_case_macro": mean_or_none([row["real_target_iou"] for row in rows]),
            "wp_dice_case_macro": mean_or_none([row["wp_dice"] for row in rows]),
            "wp_iou_case_macro": mean_or_none([row["wp_iou"] for row in rows]),
            "wp_dice_absolute_improvement_case_macro": mean_or_none([row["wp_dice_absolute_improvement"] for row in rows]),
            "wp_iou_absolute_improvement_case_macro": mean_or_none([row["wp_iou_absolute_improvement"] for row in rows]),
            "wp_dice_relative_improvement_pct_case_macro": mean_or_none([row["wp_dice_relative_improvement_pct"] for row in rows]),
            "wp_iou_relative_improvement_pct_case_macro": mean_or_none([row["wp_iou_relative_improvement_pct"] for row in rows]),
            "wp_dice_win_rate_vs_source": float(np.mean([row["wp_dice"] > row["source_dice"] for row in rows])) if rows else None,
            "wp_iou_win_rate_vs_source": float(np.mean([row["wp_iou"] > row["source_iou"] for row in rows])) if rows else None,
        })
    write_csv(output_dir / "table2_full_volume_summary.csv", table2_rows)
    summary = {
        "stage": "V10 pre-projection shared-output Language WP",
        "status": "completed",
        "source_prompt": SOURCE_PROMPT,
        "target_prompts": list(TARGET_PROMPTS),
        "wrong_prompts_test_only": list(WRONG_PROMPTS),
        "voxtell_frozen": True,
        "voxtell_network_frozen": True,
        "voxtell_text_backbone_frozen": isinstance(getattr(interface.predictor, "text_backbone", None), nn.Module),
        "full_train_split_used": True,
        "loss": "NMSE(M_hat_t, M_t) only; no multi-level projection loss and no segmentation loss",
        "shared_output_definition": "transformer_decoder output after repeat to [B,N,Q], before project_to_decoder_channels",
        "shared_output_shape": [1, shared_dim],
        "train_case_count": len(train_cases),
        "test_case_count": len(test_cases),
        "train_patch_count": len(train_manifest),
        "test_patch_count": len(test_manifest),
        "train_sample_count": int(train_data["source"].shape[0]),
        "test_sample_count": int(test_data["source"].shape[0]),
        "patch_protocol": {"patches_per_case": 4, "foreground_patches_per_case": 2, "foreground_candidate_patches": 16, "foreground_threshold": args.foreground_threshold, "patch_size": list(interface.predictor.patch_size)},
        "full_volume_sliding_window_tiles_by_case": tile_counts,
        "checkpoint": str(checkpoint_path),
        "outputs": {
            "patch_metrics": str(output_dir / "shared_output_patch_metrics.csv"),
            "case_metrics": str(output_dir / "shared_output_case_metrics.csv"),
            "table1": str(output_dir / "table1_shared_output_summary.csv"),
            "full_volume_case_metrics": str(output_dir / "full_volume_case_metrics.csv"),
            "table2": str(output_dir / "table2_full_volume_summary.csv"),
            "training_curve": str(output_dir / "training_curve.csv"),
            "run_command": str(output_dir / "run_command.txt"),
        },
        "table1": language_summary,
        "table2": table2_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "checkpoint": str(checkpoint_path)}, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
