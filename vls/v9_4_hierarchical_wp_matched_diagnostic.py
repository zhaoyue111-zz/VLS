"""V9.4 evaluation-only diagnostic for the frozen V9.3 hierarchical WP.

The V9.3 checkpoint and its fixed test patch manifest are evaluated without
training, SFDA, model-structure changes, or segmentation-head additions.  For
each real gamma(+0.30) or blur(sigma=1.5) action, this script compares:

* ``identity_source``: the unmodified source VoxTell prediction;
* ``correct_action_wp``: V9.3 WP prediction with the matching action;
* ``wrong_action_wp``: V9.3 WP prediction with gamma/blur swapped;
* ``real_action_oracle``: native VoxTell prediction on the real augmented view.

The predicted multi-level skips are connected to the original VoxTell native
decoder through the existing V9.3 decoder contract.  The decoder is never
edited or replaced.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_image_cases, read_image
from vls.v2_experiment import (
    padded_image_and_slicers,
    padded_visual_action_and_slicers,
    resolve_device,
    visual_action,
)
from vls.v3_language_experiment import flatten_prompt_embedding
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT
from vls.v7_0d_protocol_sanity import set_seed
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import normalized_mse
from vls.v9_3_hierarchical_residual_world_predictor import (
    ACTION_PROTOCOL,
    GAMMA_STRENGTH,
    BLUR_SIGMA,
    HierarchicalResidualWorldPredictor,
    LEVEL_NAMES,
    level_features,
    native_decoder_from_predicted_skips,
)


DEFAULT_CHECKPOINT = Path("outputs/v9_3_hierarchical_residual_world_predictor/best_hierarchical_world_predictor.pt")
DEFAULT_MANIFEST = Path("outputs/v9_3_hierarchical_residual_world_predictor/test_patch_manifest.json")
OUTPUT_DIR = Path("outputs/v9_4_hierarchical_wp_matched_diagnostic")
COMPARISON_GROUPS = (
    "identity_source",
    "correct_action_wp",
    "wrong_action_wp",
    "real_action_oracle",
)
ACTION_NAMES = ("gamma", "blur")

ENCODER_FIELDS = (
    "case",
    "patch_index",
    "patch_kind",
    "action",
    "comparison_group",
    "layer",
    "feature_shape",
    "normalized_mse_to_real_action",
    "delta_cosine_to_real_action",
    "delta_cosine_valid",
    "magnitude_ratio_to_real_action",
    "magnitude_ratio_valid",
    "group_delta_norm",
    "real_delta_norm",
)
FINAL_FIELDS = (
    "case",
    "patch_index",
    "patch_kind",
    "action",
    "comparison_group",
    "probability_mse_to_real_oracle",
    "probability_mae_to_real_oracle",
    "logits_mse_to_real_oracle",
    "mask_dice_to_real_oracle",
    "mask_iou_to_real_oracle",
)
GROUPED_FIELDS = (
    "table",
    "scope",
    "case",
    "action",
    "comparison_group",
    "layer",
    "patch_count",
    "normalized_mse_to_real_action",
    "delta_cosine_to_real_action",
    "magnitude_ratio_to_real_action",
    "probability_mse_to_real_oracle",
    "probability_mae_to_real_oracle",
    "logits_mse_to_real_oracle",
    "mask_dice_to_real_oracle",
    "mask_iou_to_real_oracle",
)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(
        description="V9.4 evaluation-only matched-action diagnostic for the V9.3 hierarchical WP"
    )
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir))
    parser.add_argument("--voxtell-root", default=str(paths.voxtell_root))
    parser.add_argument("--data-root", default=str(paths.data_root))
    parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--patch-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0, help="debug smoke-test limit; 0 uses all 8 test cases")
    parser.add_argument("--patch-limit", type=int, default=0, help="debug smoke-test limit per case; 0 uses all manifest patches")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="evaluate one fixed manifest patch and print checks without writing formal outputs",
    )
    return parser.parse_args()


def make_paths(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )


def deserialize_slicer(value: list[list[int | None]]) -> tuple[slice, ...]:
    return tuple(slice(start, stop, step) for start, stop, step in value)


def load_test_manifest(path: Path, cases: list[Any], patch_limit: int = 0) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"V9.3 test patch manifest not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise AssertionError(f"V9.3 test patch manifest must be a list: {path}")
    expected_names = [case.case for case in cases]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw:
        if record.get("case") in expected_names:
            grouped[str(record["case"])].append(record)
    if set(grouped) != set(expected_names):
        missing = sorted(set(expected_names) - set(grouped))
        extra = sorted(set(grouped) - set(expected_names))
        raise AssertionError(f"Manifest/test split mismatch; missing={missing}, extra={extra}")
    for case_name in expected_names:
        records = grouped[case_name]
        records.sort(key=lambda record: int(record["patch_index"]))
        if len(records) != 4:
            raise AssertionError(f"V9.3 test manifest must have exactly 4 patches for {case_name}, got {len(records)}")
        for expected_index, record in enumerate(records):
            if int(record["patch_index"]) != expected_index:
                raise AssertionError(f"Non-contiguous patch indices for {case_name}")
            if record.get("patch_kind") not in {"context", "foreground", "context_fill", "repeat_fill"}:
                raise AssertionError(f"Unexpected patch kind in manifest: {record.get('patch_kind')}")
            if "slicer" not in record:
                raise AssertionError(f"Manifest record has no slicer: {record}")
        if patch_limit:
            grouped[case_name] = records[:patch_limit]
    return grouped


def canonical_patch_kind(value: str) -> str:
    return "context_fill" if value == "repeat_fill" else value


def load_v93_world_predictor(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HierarchicalResidualWorldPredictor, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"V9.3 WP checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "V9.3 hierarchical residual World Predictor":
        raise AssertionError(f"Unexpected checkpoint stage: {checkpoint.get('stage')}")
    if checkpoint.get("action_protocol") != {"gamma": 0.3, "blur": 1.5}:
        raise AssertionError(f"Checkpoint action protocol is not V9.3 gamma=0.30/blur=1.5: {checkpoint.get('action_protocol')}")
    checkpoint_levels = tuple(checkpoint.get("selected_encoder_levels", ()))
    if checkpoint_levels != LEVEL_NAMES:
        raise AssertionError(f"Checkpoint encoder levels differ from V9.3: {checkpoint_levels}")
    level_channels = {str(name): int(value) for name, value in checkpoint["level_channels"].items()}
    model = HierarchicalResidualWorldPredictor(
        level_channels=level_channels,
        hidden_channels=int(checkpoint["hidden_channels"]),
        text_delta_dim=int(checkpoint["text_delta_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "path": str(checkpoint_path),
        "stage": checkpoint["stage"],
        "selected_epoch": checkpoint.get("selected_epoch"),
        "selected_encoder_levels": list(checkpoint_levels),
        "level_channels": level_channels,
        "hidden_channels": int(checkpoint["hidden_channels"]),
        "text_delta_dim": int(checkpoint["text_delta_dim"]),
        "train_case_count": checkpoint.get("train_case_count"),
        "test_case_count": checkpoint.get("test_case_count"),
    }
    return model, metadata


def build_text_delta(interface: VoxTellStateInterface, device: torch.device) -> torch.Tensor:
    text_embedding = interface.embed_text_prompts(DEFAULT_PROMPTS + ["the liver"])
    text_delta = flatten_prompt_embedding(text_embedding, 1) - flatten_prompt_embedding(text_embedding, 0)
    return text_delta.to(device)[None]


def l2_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float().reshape(-1)).detach().cpu())


def transition_metrics(
    source: torch.Tensor,
    real: torch.Tensor,
    group: torch.Tensor,
) -> dict[str, Any]:
    source = source.float()
    real = real.float()
    group = group.float()
    true_delta = real - source
    group_delta = group - source
    true_norm = torch.linalg.vector_norm(true_delta.reshape(-1))
    group_norm = torch.linalg.vector_norm(group_delta.reshape(-1))
    denominator = group_norm * true_norm
    valid_cosine = float(denominator.detach().cpu()) > 1e-12
    valid_ratio = float(true_norm.detach().cpu()) > 1e-12
    return {
        "normalized_mse_to_real_action": float(normalized_mse(group, real).detach().cpu()),
        "delta_cosine_to_real_action": None if not valid_cosine else float((group_delta.flatten() @ true_delta.flatten() / denominator).detach().cpu()),
        "delta_cosine_valid": valid_cosine,
        "magnitude_ratio_to_real_action": None if not valid_ratio else float((group_norm / true_norm).detach().cpu()),
        "magnitude_ratio_valid": valid_ratio,
        "group_delta_norm": float(group_norm.detach().cpu()),
        "real_delta_norm": float(true_norm.detach().cpu()),
    }


def encoder_rows(
    source_features: dict[str, torch.Tensor],
    real_features: dict[str, torch.Tensor],
    states: dict[str, dict[str, torch.Tensor]],
    case_name: str,
    patch_index: int,
    patch_kind: str,
    action: str,
) -> list[dict[str, Any]]:
    rows = []
    for layer in LEVEL_NAMES:
        source = source_features[layer]
        real = real_features[layer]
        for group in COMPARISON_GROUPS:
            state = states[group][layer]
            if tuple(state.shape) != tuple(real.shape):
                raise AssertionError(f"State shape mismatch at {layer}/{group}: {state.shape} vs {real.shape}")
            metrics = transition_metrics(source, real, state)
            rows.append({
                "case": case_name,
                "patch_index": patch_index,
                "patch_kind": patch_kind,
                "action": action,
                "comparison_group": group,
                "layer": layer,
                "feature_shape": "x".join(str(size) for size in state.shape),
                **metrics,
            })
    return rows


def binary_mask_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    tp = int(np.count_nonzero(left & right))
    fp = int(np.count_nonzero(left & ~right))
    fn = int(np.count_nonzero(~left & right))
    empty = tp + fp + fn == 0
    dice = 1.0 if empty else 2.0 * tp / max(2 * tp + fp + fn, 1)
    iou = 1.0 if empty else tp / max(tp + fp + fn, 1)
    return float(dice), float(iou)


def final_prediction_metrics(
    logits: torch.Tensor,
    real_logits: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    logits = logits.float()
    real_logits = real_logits.float()
    if tuple(logits.shape) != tuple(real_logits.shape):
        raise AssertionError(f"Final logit shape mismatch: {logits.shape} vs {real_logits.shape}")
    probability = torch.sigmoid(logits)
    real_probability = torch.sigmoid(real_logits)
    dice, iou = binary_mask_metrics(
        (probability > threshold).detach().cpu().numpy(),
        (real_probability > threshold).detach().cpu().numpy(),
    )
    delta = probability - real_probability
    return {
        "probability_mse_to_real_oracle": float(torch.mean(delta.square()).detach().cpu()),
        "probability_mae_to_real_oracle": float(torch.mean(delta.abs()).detach().cpu()),
        "logits_mse_to_real_oracle": float(torch.mean((logits - real_logits).square()).detach().cpu()),
        "mask_dice_to_real_oracle": dice,
        "mask_iou_to_real_oracle": iou,
    }


def final_rows(
    source_logits: torch.Tensor,
    real_logits: torch.Tensor,
    predicted_logits: dict[str, torch.Tensor],
    case_name: str,
    patch_index: int,
    patch_kind: str,
    action: str,
    threshold: float,
) -> list[dict[str, Any]]:
    logits = {
        "identity_source": source_logits,
        "correct_action_wp": predicted_logits["correct_action_wp"],
        "wrong_action_wp": predicted_logits["wrong_action_wp"],
        "real_action_oracle": real_logits,
    }
    rows = []
    for group in COMPARISON_GROUPS:
        rows.append({
            "case": case_name,
            "patch_index": patch_index,
            "patch_kind": patch_kind,
            "action": action,
            "comparison_group": group,
            **final_prediction_metrics(logits[group], real_logits, threshold),
        })
    return rows


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def group_rows(
    encoder: list[dict[str, Any]],
    final: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    scopes: list[tuple[str, str | None, Any]] = [("overall", None, lambda row: True)]
    scopes.append(("foreground", None, lambda row: row["patch_kind"] == "foreground"))
    scopes.append(("context", None, lambda row: row["patch_kind"] in {"context", "context_fill"}))
    for case_name in sorted({row["case"] for row in final}):
        scopes.append(("case", case_name, lambda row, case_name=case_name: row["case"] == case_name))

    for scope, case_name, predicate in scopes:
        for action in ACTION_NAMES:
            for comparison_group in COMPARISON_GROUPS:
                final_group = [
                    row for row in final
                    if row["action"] == action
                    and row["comparison_group"] == comparison_group
                    and predicate(row)
                ]
                for layer in LEVEL_NAMES:
                    encoder_group = [
                        row for row in encoder
                        if row["action"] == action
                        and row["comparison_group"] == comparison_group
                        and row["layer"] == layer
                        and predicate(row)
                    ]
                    grouped.append({
                        "table": "encoder",
                        "scope": scope,
                        "case": case_name,
                        "action": action,
                        "comparison_group": comparison_group,
                        "layer": layer,
                        "patch_count": len(encoder_group),
                        "normalized_mse_to_real_action": mean_field(encoder_group, "normalized_mse_to_real_action"),
                        "delta_cosine_to_real_action": mean_field(encoder_group, "delta_cosine_to_real_action"),
                        "magnitude_ratio_to_real_action": mean_field(encoder_group, "magnitude_ratio_to_real_action"),
                        "probability_mse_to_real_oracle": None,
                        "probability_mae_to_real_oracle": None,
                        "logits_mse_to_real_oracle": None,
                        "mask_dice_to_real_oracle": None,
                        "mask_iou_to_real_oracle": None,
                    })
                grouped.append({
                    "table": "final",
                    "scope": scope,
                    "case": case_name,
                    "action": action,
                    "comparison_group": comparison_group,
                    "layer": None,
                    "patch_count": len(final_group),
                    "normalized_mse_to_real_action": None,
                    "delta_cosine_to_real_action": None,
                    "magnitude_ratio_to_real_action": None,
                    "probability_mse_to_real_oracle": mean_field(final_group, "probability_mse_to_real_oracle"),
                    "probability_mae_to_real_oracle": mean_field(final_group, "probability_mae_to_real_oracle"),
                    "logits_mse_to_real_oracle": mean_field(final_group, "logits_mse_to_real_oracle"),
                    "mask_dice_to_real_oracle": mean_field(final_group, "mask_dice_to_real_oracle"),
                    "mask_iou_to_real_oracle": mean_field(final_group, "mask_iou_to_real_oracle"),
                })
    return grouped


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def final_metric_lookup(rows: list[dict[str, Any]], action: str, group: str) -> dict[str, float]:
    values = [row for row in rows if row["action"] == action and row["comparison_group"] == group]
    if not values:
        raise AssertionError(f"No final rows for action={action}, group={group}")
    return {
        field: float(np.mean([row[field] for row in values], dtype=np.float64))
        for field in (
            "probability_mse_to_real_oracle",
            "probability_mae_to_real_oracle",
            "mask_dice_to_real_oracle",
            "mask_iou_to_real_oracle",
        )
    }


def compare_final_metrics(rows: list[dict[str, Any]], action: str, left: str, right: str) -> dict[str, Any]:
    left_values = final_metric_lookup(rows, action, left)
    right_values = final_metric_lookup(rows, action, right)
    lower_is_better = ("probability_mse_to_real_oracle", "probability_mae_to_real_oracle")
    higher_is_better = ("mask_dice_to_real_oracle", "mask_iou_to_real_oracle")
    return {
        "left": left,
        "right": right,
        "left_metrics": left_values,
        "right_metrics": right_values,
        "metric_wise": {
            field: left_values[field] < right_values[field] for field in lower_is_better
        } | {
            field: left_values[field] > right_values[field] for field in higher_is_better
        },
        "all_four_strict": (
            all(left_values[field] < right_values[field] for field in lower_is_better)
            and all(left_values[field] > right_values[field] for field in higher_is_better)
        ),
        "probability_primary_strict": all(
            left_values[field] < right_values[field] for field in lower_is_better
        ),
        "mask_non_worse": all(
            left_values[field] >= right_values[field] for field in higher_is_better
        ),
    }


def build_summary(
    args: argparse.Namespace,
    cases: list[Any],
    checkpoint_metadata: dict[str, Any],
    encoder: list[dict[str, Any]],
    final: list[dict[str, Any]],
    grouped: list[dict[str, Any]],
) -> dict[str, Any]:
    blur_correct_vs_identity = compare_final_metrics(final, "blur", "correct_action_wp", "identity_source")
    blur_correct_vs_wrong = compare_final_metrics(final, "blur", "correct_action_wp", "wrong_action_wp")
    blur_mask_closer = {
        "correct_action_wp_mask_dice": final_metric_lookup(final, "blur", "correct_action_wp")["mask_dice_to_real_oracle"],
        "identity_source_mask_dice": final_metric_lookup(final, "blur", "identity_source")["mask_dice_to_real_oracle"],
        "correct_action_wp_mask_iou": final_metric_lookup(final, "blur", "correct_action_wp")["mask_iou_to_real_oracle"],
        "identity_source_mask_iou": final_metric_lookup(final, "blur", "identity_source")["mask_iou_to_real_oracle"],
    }
    blur_mask_closer["correct_wp_mask_closer_than_identity"] = (
        blur_mask_closer["correct_action_wp_mask_dice"] > blur_mask_closer["identity_source_mask_dice"]
        and blur_mask_closer["correct_action_wp_mask_iou"] > blur_mask_closer["identity_source_mask_iou"]
    )
    return {
        "stage": "V9.4 hierarchical residual WP matched-action evaluation-only diagnostic",
        "evaluation_only": True,
        "models_trained": False,
        "sfda_run": False,
        "voxtell_modified": False,
        "v93_network_modified": False,
        "loss_modified": False,
        "checkpoint_modified": False,
        "segmentation_head_added": False,
        "test_case_count": len(cases),
        "test_cases": [case.case for case in cases],
        "manifest_records_evaluated": len({(row["case"], row["patch_index"]) for row in final}),
        "action_protocol": {"gamma": GAMMA_STRENGTH, "blur": BLUR_SIGMA},
        "comparison_groups": list(COMPARISON_GROUPS),
        "grouping": {
            "overall": "all fixed test-manifest patches",
            "foreground": "patch_kind == foreground",
            "context": "patch_kind in {context, context_fill}",
            "case": "all fixed manifest patches within each test case",
        },
        "checkpoint": checkpoint_metadata,
        "patch_manifest": str(Path(args.patch_manifest)),
        "encoder_fidelity_definition": "Each group state is compared with the real-action feature at every V9.3 predicted encoder level; delta cosine and magnitude ratio use (group-source) versus (real-source).",
        "final_fidelity_definition": "Each group final probability/mask is compared with the real augmented image native VoxTell final prediction.",
        "blur_automatic_judgment": {
            "correct_wp_better_than_identity": blur_correct_vs_identity,
            "correct_wp_better_than_wrong_action": blur_correct_vs_wrong,
            "correct_wp_final_mask_closer_than_identity": blur_mask_closer,
            "main_pass_rule": "Primary pass requires correct_action_wp to have strictly lower probability MSE and MAE than the comparator; mask closeness is reported separately because tied empty/unchanged thresholded masks can hide probability improvement.",
        },
        "gamma_diagnostic": {
            "correct_vs_identity": compare_final_metrics(final, "gamma", "correct_action_wp", "identity_source"),
            "correct_vs_wrong_action": compare_final_metrics(final, "gamma", "correct_action_wp", "wrong_action_wp"),
            "excluded_from_main_pass_condition": True,
        },
        "row_counts": {"encoder": len(encoder), "final": len(final), "grouped": len(grouped)},
        "outputs": {
            "encoder_transition_fidelity": str(OUTPUT_DIR / "encoder_transition_fidelity.csv"),
            "final_prediction_fidelity": str(OUTPUT_DIR / "final_prediction_fidelity.csv"),
            "grouped_fidelity": str(OUTPUT_DIR / "grouped_fidelity.csv"),
            "summary": str(OUTPUT_DIR / "summary.json"),
        },
    }


def run(args: argparse.Namespace) -> None:
    if ACTION_PROTOCOL != (("gamma", 0.30), ("blur", 1.5)):
        raise AssertionError("V9.4 requires V9.3 gamma=0.30 and blur=1.5")
    if args.case_limit < 0 or args.patch_limit < 0:
        raise AssertionError("Debug limits must be non-negative")
    if args.prediction_threshold != 0.5:
        raise AssertionError("V9.3 prediction threshold must remain 0.5")
    set_seed(args.seed)
    paths = make_paths(args)
    device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"V9.4 requires CUDA, resolved {device}")
    cases = iter_image_cases(paths, split="test")
    if len(cases) != 8:
        raise AssertionError(f"V9.4 requires the original 8-case V9.3 test split, got {len(cases)}")
    if args.case_limit:
        cases = cases[: args.case_limit]
    manifest = load_test_manifest(Path(args.patch_manifest), cases, args.patch_limit)

    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    model, checkpoint_metadata = load_v93_world_predictor(Path(args.world_checkpoint), device)
    prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    text_delta = build_text_delta(interface, device)
    text_backbone = getattr(interface.predictor, "text_backbone", None)
    interface.predictor.text_backbone = None
    interface.predictor.tokenizer = None
    interface.predictor._text_embedding_cache.clear()
    del text_backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    encoder_rows_all: list[dict[str, Any]] = []
    final_rows_all: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        print(f"[V9.4] case {case_index}/{len(cases)} start {case.case}", flush=True)
        image, _ = read_image(case)
        source_padded, _ = padded_image_and_slicers(interface.predictor, image)
        action_padded = {
            action: padded_visual_action_and_slicers(interface.predictor, image, action, strength)[0]
            for action, strength in ACTION_PROTOCOL
        }
        if any(tuple(value.shape) != tuple(source_padded.shape) for value in action_padded.values()):
            raise AssertionError(f"Padded source/action shape mismatch for {case.case}")
        records = manifest[case.case]
        for record in records:
            patch_index = int(record["patch_index"])
            patch_kind = canonical_patch_kind(str(record["patch_kind"]))
            slicer = deserialize_slicer(record["slicer"])
            source_patch = torch.clone(source_padded[slicer][None], memory_format=torch.contiguous_format)
            source_context = interface.forward_with_audit_context(source_patch, prompt_embedding)
            source_skips = source_context["decoder_audit"]["skips"]
            source_features = level_features(source_skips, LEVEL_NAMES)
            source_logits = source_context["final_prediction"][:, :1]
            for action, strength in ACTION_PROTOCOL:
                action_patch = torch.clone(action_padded[action][slicer][None], memory_format=torch.contiguous_format)
                action_context = interface.forward_with_audit_context(action_patch, prompt_embedding)
                real_features = level_features(action_context["decoder_audit"]["skips"], LEVEL_NAMES)
                correct_action = visual_action(action, strength, device)
                wrong_name = "blur" if action == "gamma" else "gamma"
                wrong_strength = BLUR_SIGMA if action == "gamma" else GAMMA_STRENGTH
                wrong_action = visual_action(wrong_name, wrong_strength, device)
                correct_features = model(source_features, correct_action, text_delta)
                wrong_features = model(source_features, wrong_action, text_delta)
                states = {
                    "identity_source": source_features,
                    "correct_action_wp": correct_features,
                    "wrong_action_wp": wrong_features,
                    "real_action_oracle": real_features,
                }
                encoder_rows_all.extend(encoder_rows(
                    source_features, real_features, states,
                    case.case, patch_index, patch_kind, action,
                ))
                predicted_logits = {
                    "correct_action_wp": native_decoder_from_predicted_skips(
                        interface, source_context, correct_features, LEVEL_NAMES,
                    ),
                    "wrong_action_wp": native_decoder_from_predicted_skips(
                        interface, source_context, wrong_features, LEVEL_NAMES,
                    ),
                }
                final_rows_all.extend(final_rows(
                    source_logits,
                    action_context["final_prediction"][:, :1],
                    predicted_logits,
                    case.case,
                    patch_index,
                    patch_kind,
                    action,
                    args.prediction_threshold,
                ))
                del action_patch, action_context, real_features, correct_features, wrong_features
                del states, predicted_logits
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del source_patch, source_context, source_features, source_skips
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[V9.4] case {case_index}/{len(cases)} patch {patch_index + 1}/{len(records)} complete", flush=True)
        del action_padded, source_padded, image
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.smoke_only:
        first_encoder = encoder_rows_all[0]
        first_final = final_rows_all[0]
        print(json.dumps({
            "smoke": "passed",
            "test_cases_evaluated": len(cases),
            "patches_evaluated": len({(row["case"], row["patch_index"]) for row in final_rows_all}),
            "encoder_rows": len(encoder_rows_all),
            "final_rows": len(final_rows_all),
            "first_encoder_row": first_encoder,
            "first_final_row": first_final,
            "checkpoint": checkpoint_metadata["path"],
            "training_run": False,
            "sfda_run": False,
        }, indent=2), flush=True)
        return

    grouped = group_rows(encoder_rows_all, final_rows_all)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder_path = OUTPUT_DIR / "encoder_transition_fidelity.csv"
    final_path = OUTPUT_DIR / "final_prediction_fidelity.csv"
    grouped_path = OUTPUT_DIR / "grouped_fidelity.csv"
    summary_path = OUTPUT_DIR / "summary.json"
    write_csv(encoder_path, ENCODER_FIELDS, encoder_rows_all)
    write_csv(final_path, FINAL_FIELDS, final_rows_all)
    write_csv(grouped_path, GROUPED_FIELDS, grouped)
    summary = build_summary(args, cases, checkpoint_metadata, encoder_rows_all, final_rows_all, grouped)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "complete": True,
        "test_cases": len(cases),
        "patches": len({(row["case"], row["patch_index"]) for row in final_rows_all}),
        "outputs": summary["outputs"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
