from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vls.config import ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import (
    padded_visual_action_and_slicers,
    prepare_functional_seg_head,
    resolve_device,
    select_patch_slicers,
    state_to_intermediate_prediction,
    visual_action,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D


SOURCE_PROMPT = "liver"
TARGET_PROMPT = "the liver"
LANGUAGE_ACTION = "liver_to_the_liver"
ORDERS = [
    ("gamma_0.30_then_language", ("gamma", 0.30), ("language", 0.0)),
    ("language_then_gamma_0.30", ("language", 0.0), ("gamma", 0.30)),
    ("blur_1.5_then_language", ("blur", 1.5), ("language", 0.0)),
    ("language_then_blur_1.5", ("language", 0.0), ("blur", 1.5)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5.0 semantic-preserving visual-language rollout sanity.")
    paths = ProjectPaths()
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt",
    )
    parser.add_argument("--output-dir", default="outputs/v5_0_rollout_sanity")
    parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--val-cases", type=int, default=4)
    parser.add_argument("--patches-per-case", type=int, default=4)
    parser.add_argument("--foreground-patches-per-case", type=int, default=2)
    parser.add_argument("--foreground-candidate-patches", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def flatten_prompt_embedding(text_embedding: torch.Tensor, prompt_index: int) -> torch.Tensor:
    if text_embedding.ndim not in (3, 4):
        raise ValueError(f"Unexpected text embedding shape: {tuple(text_embedding.shape)}")
    return text_embedding[:, prompt_index].detach().float().flatten()


def load_world_model(
    checkpoint_path: Path,
    in_channels: int,
    device: torch.device,
    hidden_channels: int,
) -> VisualWorldPredictor3D:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["hidden_channels"]) != hidden_channels:
        raise ValueError("World checkpoint hidden_channels does not match the requested model")
    model = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        action_dim=3,
        num_blocks=2,
        use_action=True,
        text_delta_dim=int(checkpoint["text_delta_dim"]),
        use_language=True,
        allow_unconditioned=True,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def normalized_components(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    diff = prediction.float() - target.float()
    numerator = float(diff.pow(2).sum().detach().cpu())
    denominator = float(target.float().pow(2).sum().detach().cpu())
    return numerator / max(denominator, 1e-6), numerator, denominator


def language_delta(interface: VoxTellStateInterface, embedding: torch.Tensor) -> torch.Tensor:
    source = flatten_prompt_embedding(embedding, 0)
    target = flatten_prompt_embedding(embedding, 1)
    return (target - source).float()


def patch_state_and_logit(
    interface: VoxTellStateInterface,
    patch: torch.Tensor,
    embedding: torch.Tensor,
    prompt_index: int,
    selected_stage: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = interface.forward_with_states(patch, embedding)
    state = result["decoder_states"][selected_stage][:, prompt_index].detach().float()
    logits = result["intermediate_predictions"][selected_stage][:, prompt_index : prompt_index + 1].detach().float()
    return state, logits


def predicted_logit(
    interface: VoxTellStateInterface,
    selected_stage: str,
    state: torch.Tensor,
) -> torch.Tensor:
    return state_to_intermediate_prediction(interface, selected_stage, state).detach().float()


def _visual_padded_and_slicers(
    interface: VoxTellStateInterface,
    image: np.ndarray,
    family: str,
    strength: float,
) -> tuple[torch.Tensor, list[tuple]]:
    return padded_visual_action_and_slicers(interface.predictor, image, family, strength)


def metric_record(
    prefix: str,
    prediction_state: torch.Tensor,
    target_state: torch.Tensor,
    prediction_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> dict[str, float]:
    state_mse, state_num, state_den = normalized_components(prediction_state, target_state)
    mask_mse, mask_num, mask_den = normalized_components(prediction_logits, target_logits)
    return {
        f"{prefix}_state_normalized_mse": state_mse,
        f"{prefix}_state_sse": state_num,
        f"{prefix}_state_target_sq": state_den,
        f"{prefix}_mask_logit_normalized_mse": mask_mse,
        f"{prefix}_mask_logit_sse": mask_num,
        f"{prefix}_mask_logit_target_sq": mask_den,
    }


@torch.inference_mode()
def evaluate_case(
    interface: VoxTellStateInterface,
    world_model: VisualWorldPredictor3D,
    case: Any,
    embedding: torch.Tensor,
    text_delta: torch.Tensor,
    selected_stage: str,
    patches_per_case: int,
    foreground_patches_per_case: int,
    foreground_candidate_patches: int,
    foreground_threshold: float,
) -> list[dict[str, Any]]:
    image, _, _ = read_image_and_label(case)
    original_padded, slicers, patch_kinds = select_patch_slicers(
        interface,
        image,
        [SOURCE_PROMPT, TARGET_PROMPT],
        patches_per_case,
        foreground_patches_per_case,
        foreground_candidate_patches,
        foreground_threshold,
    )
    gamma_padded, _ = _visual_padded_and_slicers(interface, image, "gamma", 0.30)
    blur_padded, _ = _visual_padded_and_slicers(interface, image, "blur", 1.5)
    device = next(world_model.parameters()).device
    language_action = text_delta[None].to(device)
    gamma_action = visual_action("gamma", 0.30, device)
    blur_action = visual_action("blur", 1.5, device)
    rows: list[dict[str, Any]] = []

    for patch_index, slicer in enumerate(slicers):
        original_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
        gamma_patch = torch.clone(gamma_padded[slicer][None], memory_format=torch.contiguous_format)
        blur_patch = torch.clone(blur_padded[slicer][None], memory_format=torch.contiguous_format)
        s0, s0_logits = patch_state_and_logit(interface, original_patch, embedding, 0, selected_stage)
        gamma_s1, gamma_s1_logits = patch_state_and_logit(interface, gamma_patch, embedding, 0, selected_stage)
        blur_s1, blur_s1_logits = patch_state_and_logit(interface, blur_patch, embedding, 0, selected_stage)
        language_s1, language_s1_logits = patch_state_and_logit(interface, original_patch, embedding, 1, selected_stage)
        gamma_language_s2, gamma_language_s2_logits = patch_state_and_logit(
            interface, gamma_patch, embedding, 1, selected_stage,
        )
        blur_language_s2, blur_language_s2_logits = patch_state_and_logit(
            interface, blur_patch, embedding, 1, selected_stage,
        )

        real_steps = {
            "gamma_0.30_then_language": (gamma_s1, gamma_s1_logits, gamma_language_s2, gamma_language_s2_logits, gamma_action, language_action),
            "language_then_gamma_0.30": (language_s1, language_s1_logits, gamma_language_s2, gamma_language_s2_logits, language_action, gamma_action),
            "blur_1.5_then_language": (blur_s1, blur_s1_logits, blur_language_s2, blur_language_s2_logits, blur_action, language_action),
            "language_then_blur_1.5": (language_s1, language_s1_logits, blur_language_s2, blur_language_s2_logits, language_action, blur_action),
        }
        for order, first, second in ORDERS:
            s1, s1_logits, s2, s2_logits, action1, action2 = real_steps[order]
            s0_device = s0.to(device)
            s1_device = s1.to(device)
            s2_device = s2.to(device)
            first_type = first[0]
            second_type = second[0]
            free_s1 = world_model(
                s0_device,
                action=action1 if first_type != "language" else None,
                text_delta=action1 if first_type == "language" else None,
            )
            free_s2 = world_model(
                free_s1,
                action=action2 if second_type != "language" else None,
                text_delta=action2 if second_type == "language" else None,
            )
            teacher_s2 = world_model(
                s1_device,
                action=action2 if second_type != "language" else None,
                text_delta=action2 if second_type == "language" else None,
            )
            identity_s2 = s0_device
            free_s1_logits = predicted_logit(interface, selected_stage, free_s1)
            free_s2_logits = predicted_logit(interface, selected_stage, free_s2)
            teacher_s2_logits = predicted_logit(interface, selected_stage, teacher_s2)
            identity_s2_logits = predicted_logit(interface, selected_stage, identity_s2)
            row: dict[str, Any] = {
                "case": case.case,
                "patch_index": patch_index,
                "patch_kind": patch_kinds[patch_index],
                "action_order": order,
                "first_action": first_type,
                "second_action": second_type,
            }
            row.update(metric_record("step1", free_s1, s1_device, free_s1_logits, s1_logits.to(device)))
            row.update(metric_record("step2_free_rollout", free_s2, s2_device, free_s2_logits, s2_logits.to(device)))
            row.update(metric_record("step2_teacher_forced", teacher_s2, s2_device, teacher_s2_logits, s2_logits.to(device)))
            row.update(metric_record("step2_identity", identity_s2, s2_device, identity_s2_logits, s2_logits.to(device)))
            rows.append(row)
    return rows


METRIC_PREFIXES = ["step1", "step2_free_rollout", "step2_teacher_forced", "step2_identity"]


def aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[key_name]) for key_name in group_keys)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group_rows in sorted(groups.items()):
        summary: dict[str, Any] = {"num_samples": len(group_rows)}
        for key_name, value in zip(group_keys, key, strict=True):
            summary[key_name] = value
        for prefix in METRIC_PREFIXES:
            for metric in ["state", "mask_logit"]:
                num_key = f"{prefix}_{metric}_sse"
                den_key = f"{prefix}_{metric}_target_sq"
                numerator = sum(float(row[num_key]) for row in group_rows)
                denominator = sum(float(row[den_key]) for row in group_rows)
                summary[f"{prefix}_{metric}_normalized_mse"] = numerator / max(denominator, 1e-6)
        summaries.append(summary)
    return summaries


def macro_from_case_rows(case_rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in case_rows:
        key = tuple(str(row[key_name]) for key_name in group_keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        item: dict[str, Any] = {"num_cases": len(rows)}
        for key_name, value in zip(group_keys, key, strict=True):
            item[key_name] = value
        for prefix in METRIC_PREFIXES:
            for metric in ["state", "mask_logit"]:
                name = f"{prefix}_{metric}_normalized_mse"
                item[name] = float(np.mean([float(row[name]) for row in rows]))
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
    cases = iter_cases(paths, split="test", limit=args.val_cases)
    embedding = interface.embed_text_prompts([SOURCE_PROMPT, TARGET_PROMPT])
    text_delta = language_delta(interface, embedding)
    prepare_functional_seg_head(interface, args.selected_stage)
    world_checkpoint = Path(args.world_checkpoint)
    checkpoint = torch.load(world_checkpoint, map_location="cpu", weights_only=False)
    model = load_world_model(
        world_checkpoint,
        int(checkpoint["state_dict"]["output_projection.bias"].shape[0]),
        device,
        args.hidden_channels,
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(evaluate_case(
            interface,
            model,
            case,
            embedding,
            text_delta,
            args.selected_stage,
            args.patches_per_case,
            args.foreground_patches_per_case,
            args.foreground_candidate_patches,
            args.foreground_threshold,
        ))
    per_order = aggregate(rows, ["action_order"])
    per_case = aggregate(rows, ["case", "action_order"])
    per_case_overall = aggregate(rows, ["case"])
    overall_micro = aggregate([{**row, "_overall": "all"} for row in rows], ["_overall"])
    case_rows = per_case
    overall_macro = macro_from_case_rows(case_rows, ["action_order"])
    macro_all = macro_from_case_rows(per_case_overall, [])
    detail_path = output_dir / "rollout_detail.csv"
    per_order_path = output_dir / "rollout_by_order.csv"
    per_case_path = output_dir / "rollout_by_case.csv"
    write_csv(detail_path, rows)
    write_csv(per_order_path, per_order)
    write_csv(per_case_path, per_case)
    summary = {
        "args": vars(args),
        "world_checkpoint": str(world_checkpoint),
        "selected_stage": args.selected_stage,
        "cases": [case.case for case in cases],
        "num_rows": len(rows),
        "prompt_pair": [SOURCE_PROMPT, TARGET_PROMPT],
        "text_embedding_cosine": float(torch.nn.functional.cosine_similarity(
            flatten_prompt_embedding(embedding, 0)[None], flatten_prompt_embedding(embedding, 1)[None],
        ).item()),
        "text_embedding_delta_norm": float(torch.linalg.vector_norm(text_delta).item()),
        "orders": [order for order, _, _ in ORDERS],
        "detail_csv": str(detail_path),
        "per_order_csv": str(per_order_path),
        "per_case_csv": str(per_case_path),
        "overall_micro": overall_micro,
        "overall_macro_by_order": overall_macro,
        "overall_macro": macro_all,
        "by_order": per_order,
        "interpretation_note": "Evaluation only. Free rollout is compared with teacher-forced second step and identity s0->s2; no training or rollout sampling is performed.",
    }
    summary_path = output_dir / "v5_0_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "summary_path": str(summary_path),
        "num_rows": len(rows),
        "orders": per_order,
        "overall_micro": overall_micro,
        "overall_macro": macro_all,
    }, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
