"""V9.1 diagnostic for the existing V8.0 World Predictor with native tail.

This script is diagnostic-only.  It loads the conditioned V8.0 World
Predictor, predicts decoder-stage-1 states, and reconnects those states to
the original VoxTell decoder through ``native_tail_from_audit_context``.
It never trains a model and never creates or calls a functional segmentation
head.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import padded_visual_action_and_slicers, resolve_device, select_patch_slicers, visual_action
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, pad_label_like_image
from vls.v7_0d_protocol_sanity import set_seed
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, normalized_mse
from vls.v9_0_world_state_selection_audit import (
    ACTION_PROTOCOL,
    CANDIDATE_NAMES,
    CsvSink,
    METRIC_FIELDS,
    METRIC_NAMES,
    REGION_FIELDS,
    MetricAccumulator,
    RegionAccumulator,
    RunningMean,
    candidate_reliability_maps,
    final_prediction_metrics,
    get_candidate_state,
    mean_rows,
    pad_selected_patches,
    reliability_patch_update,
    write_case_and_global_reliability_rows,
    write_progress,
)


SELECTED_STAGE = 'decoder_stage_1_low_to_high'
ACTION_SPECS = {
    'gamma': {'strength': 0.30, 'vector': (1.0, 0.0, 0.30)},
    'blur': {'strength': 1.5, 'vector': (0.0, 1.0, 1.50)},
}
GROUPS = ('full_real', 'real_A_state_oracle', 'existing_WP_imagined')

LATENT_FIELDS = (
    'case', 'patch_index', 'patch_kind', 'action', 'comparison_group',
    'state_shape', 'channels', 'spatial_resolution',
    'normalized_mse_to_real_A', 'group_delta_norm', 'true_delta_norm',
    'predicted_delta_norm', 'magnitude_ratio', 'delta_cosine_to_true',
    'delta_cosine_valid', 'group_delta_rms', 'true_delta_rms',
)
FINAL_FIELDS = (
    'case', 'patch_index', 'patch_kind', 'action', 'comparison_group',
    'context_mode', 'probability_mse', 'mean_absolute_probability_error',
    'logits_mse', 'imagined_vs_real_mask_dice', 'imagined_vs_real_mask_iou',
)
PATCH_FIELDS = ('case', 'patch_index', 'patch_kind', 'slicer')


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description='V9.1 existing V8.0 World Predictor native-tail diagnostic')
    parser.add_argument('--model-dir', default=str(paths.voxtell_model_dir))
    parser.add_argument('--voxtell-root', default=str(paths.voxtell_root))
    parser.add_argument('--data-root', default=str(paths.data_root))
    parser.add_argument('--split-json', default=str(paths.split_json))
    parser.add_argument('--world-checkpoint', default='outputs/v8_0_full_world_predictor/best_world_predictor.pt')
    parser.add_argument('--output-dir', default='outputs/v9_1_existing_wp_native_tail_diagnostic')
    parser.add_argument('--patches-per-case', type=int, default=4)
    parser.add_argument('--foreground-patches-per-case', type=int, default=2)
    parser.add_argument('--foreground-candidate-patches', type=int, default=16)
    parser.add_argument('--foreground-threshold', type=float, default=0.5)
    parser.add_argument('--prediction-threshold', type=float, default=0.5)
    parser.add_argument('--label-value', type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case-limit', type=int, default=0, help='debug smoke-test limit; 0 uses all 30 train cases')
    parser.add_argument('--patch-limit', type=int, default=0, help='debug smoke-test limit; 0 uses all selected patches')
    return parser.parse_args()


def memory_status(device: torch.device) -> dict[str, float]:
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    result = {'cpu_rss_mb': float(rss_mb)}
    if device.type == 'cuda' and torch.cuda.is_available():
        result['gpu_allocated_mb'] = float(torch.cuda.memory_allocated(device) / 1024**2)
        result['gpu_reserved_mb'] = float(torch.cuda.memory_reserved(device) / 1024**2)
    else:
        result['gpu_allocated_mb'] = 0.0
        result['gpu_reserved_mb'] = 0.0
    return result


def load_existing_world_predictor(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[VisualWorldPredictor3D, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('selected_stage') != SELECTED_STAGE:
        raise AssertionError(
            f'V9.1 requires V8.0 selected_stage={SELECTED_STAGE}, got {checkpoint.get("selected_stage")}'
        )
    if checkpoint.get('stage') != 'V8.0 full-train visual World Predictor':
        raise AssertionError(f'Unexpected World Predictor checkpoint stage: {checkpoint.get("stage")}')
    architecture = checkpoint.get('architecture', {})
    if architecture.get('use_action') is not True or architecture.get('use_language') is not True:
        raise AssertionError('V9.1 requires the conditioned V8.0 World Predictor checkpoint')
    state_dict = checkpoint['state_dict']
    in_channels = int(state_dict['output_projection.bias'].shape[0])
    hidden_channels = int(checkpoint['hidden_channels'])
    text_delta_dim = int(checkpoint['text_delta_dim'])
    model = VisualWorldPredictor3D(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        action_dim=3,
        num_blocks=int(architecture.get('num_blocks', 2)),
        use_action=True,
        text_delta_dim=text_delta_dim,
        use_language=True,
        allow_unconditioned=True,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        'checkpoint_path': str(checkpoint_path),
        'selected_stage': checkpoint['selected_stage'],
        'hidden_channels': hidden_channels,
        'text_delta_dim': text_delta_dim,
        'input_channels': in_channels,
        'action_dim': int(checkpoint.get('action_dim', 3)),
        'architecture': architecture,
        'train_case_count': checkpoint.get('train_case_count'),
        'test_case_count': checkpoint.get('test_case_count'),
        'selected_epoch': checkpoint.get('selected_epoch'),
    }
    return model, metadata


@torch.inference_mode()
def predict_world_state(
    model: VisualWorldPredictor3D,
    source_state: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    return model(source_state.float(), action=action.float())


def l2_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float().reshape(-1)).detach().cpu())


def rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.float().square())).detach().cpu())


def cosine_to_true(predicted_delta: torch.Tensor, true_delta: torch.Tensor) -> tuple[float | None, bool]:
    denominator = torch.linalg.vector_norm(predicted_delta.float().reshape(-1)) * torch.linalg.vector_norm(true_delta.float().reshape(-1))
    valid = float(denominator.detach().cpu()) > 1e-12
    if not valid:
        return None, False
    value = torch.sum(predicted_delta.float() * true_delta.float()) / denominator
    return float(value.detach().cpu()), True


def latent_rows(
    source_state: torch.Tensor,
    real_state: torch.Tensor,
    predicted_state: torch.Tensor,
    case: str,
    patch_index: int,
    patch_kind: str,
    action: str,
) -> list[dict[str, Any]]:
    source = source_state.float()
    real = real_state.float()
    predicted = predicted_state.float()
    true_delta = real - source
    predicted_delta = predicted - source
    true_norm = l2_norm(true_delta)
    predicted_norm = l2_norm(predicted_delta)
    predicted_cosine, valid = cosine_to_true(predicted_delta, true_delta)
    groups = {
        'identity_source': source,
        'existing_WP_predicted': predicted,
        'real_A_state_oracle': real,
    }
    rows = []
    for group, state in groups.items():
        group_delta = state.float() - source
        group_norm = l2_norm(group_delta)
        group_cosine, group_valid = cosine_to_true(group_delta, true_delta)
        rows.append({
            'case': case,
            'patch_index': patch_index,
            'patch_kind': patch_kind,
            'action': action,
            'comparison_group': group,
            'state_shape': 'x'.join(str(size) for size in state.shape),
            'channels': int(state.shape[1]),
            'spatial_resolution': 'x'.join(str(size) for size in state.shape[-3:]),
            'normalized_mse_to_real_A': float(normalized_mse(state, real).detach().cpu()),
            'group_delta_norm': group_norm,
            'true_delta_norm': true_norm,
            'predicted_delta_norm': predicted_norm if group == 'existing_WP_predicted' else group_norm,
            'magnitude_ratio': None if true_norm <= 1e-12 else group_norm / true_norm,
            'delta_cosine_to_true': group_cosine,
            'delta_cosine_valid': group_valid,
            'group_delta_rms': rms(group_delta),
            'true_delta_rms': rms(true_delta),
        })
    return rows


def final_rows(
    source_context: dict[str, Any],
    action_context: dict[str, Any],
    source_state: torch.Tensor,
    predicted_state: torch.Tensor,
    real_state: torch.Tensor,
    case: str,
    patch_index: int,
    patch_kind: str,
    action: str,
    prediction_threshold: float,
    stage_idx: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    real_logits = action_context['final_prediction'][:, :1]
    groups = {
        'identity_source': source_state,
        'existing_WP_predicted': predicted_state,
        'real_A_state_oracle': real_state,
    }
    logits: dict[str, torch.Tensor] = {}
    rows = []
    for group, state in groups.items():
        imagined_logits = VoxTellStateInterface.native_tail_from_audit_context(
            source_context,
            stage_idx,
            'stage_output',
            state,
            mask_context=source_context,
            skip_context=source_context,
        )
        logits[group] = imagined_logits
        metrics = final_prediction_metrics(imagined_logits, real_logits, prediction_threshold)
        rows.append({
            'case': case,
            'patch_index': patch_index,
            'patch_kind': patch_kind,
            'action': action,
            'comparison_group': group,
            'context_mode': 'source-side native skip and mask context',
            'probability_mse': metrics['probability_mse'],
            'mean_absolute_probability_error': metrics['mean_absolute_probability_error'],
            'logits_mse': metrics['logits_mse'],
            'imagined_vs_real_mask_dice': metrics['predicted_mask_dice_vs_real'],
            'imagined_vs_real_mask_iou': metrics['predicted_mask_iou_vs_real'],
        })
    return rows, logits


def summary_metric(global_metrics: dict[tuple[str, str, str], MetricAccumulator], group: str, method: str, scope: str, name: str) -> float | None:
    accumulator = global_metrics.get((group, method, scope))
    return None if accumulator is None else accumulator.metrics.get(name, RunningMean()).mean()


def summary_region(global_regions: dict[tuple[str, str, str], RegionAccumulator], group: str, method: str, region: str) -> float | None:
    accumulator = global_regions.get((group, method, region))
    return None if accumulator is None else accumulator.mean()


def build_summary(
    args: argparse.Namespace,
    train_names: list[str],
    test_names: list[str],
    checkpoint_metadata: dict[str, Any],
    state_metadata: dict[str, Any],
    latent_rows_all: list[dict[str, Any]],
    final_rows_all: list[dict[str, Any]],
    global_metrics: dict[tuple[str, str, str], MetricAccumulator],
    global_regions: dict[tuple[str, str, str], RegionAccumulator],
    output_dir: Path,
) -> dict[str, Any]:
    reliability = {}
    for group in GROUPS:
        reliability[group] = {}
        for method in (
            'confidence_rank', 'gamma_world_stability', 'blur_world_stability',
            'pairwise_world_stability', 'gamma_joint_product',
            'blur_joint_product', 'pairwise_joint_product',
        ):
            reliability[group][method] = {
                scope: {
                    name: summary_metric(global_metrics, group, method, scope, name)
                    for name in METRIC_NAMES
                }
                for scope in ('overall', 'foreground', 'background')
            }
    region_reliability = {
        group: {
            method: {
                region: summary_region(global_regions, group, method, region)
                for region in ('TP', 'FP', 'FN', 'TN')
            }
            for method in reliability[group]
        }
        for group in GROUPS
    }
    latent_summary = {
        group: {
            action: {
                field: mean_rows([
                    row for row in latent_rows_all
                    if row['comparison_group'] == group and row['action'] == action
                ], field)
                for field in ('normalized_mse_to_real_A', 'group_delta_norm', 'true_delta_norm', 'predicted_delta_norm', 'magnitude_ratio', 'delta_cosine_to_true')
            }
            for action, _ in ACTION_PROTOCOL
        }
        for group in ('identity_source', 'existing_WP_predicted', 'real_A_state_oracle')
    }
    final_summary = {
        group: {
            action: {
                field: mean_rows([
                    row for row in final_rows_all
                    if row['comparison_group'] == group and row['action'] == action
                ], field)
                for field in ('probability_mse', 'mean_absolute_probability_error', 'logits_mse', 'imagined_vs_real_mask_dice', 'imagined_vs_real_mask_iou')
            }
            for action, _ in ACTION_PROTOCOL
        }
        for group in ('identity_source', 'existing_WP_predicted', 'real_A_state_oracle')
    }
    return {
        'stage': 'V9.1 Existing V8.0 World Predictor Native-Tail Diagnostic',
        'models_trained': False,
        'world_predictor_loaded': True,
        'gt_used_only_for_reliability_diagnosis': True,
        'extra_segmentation_head_used': False,
        'functional_seg_head_used': False,
        'state_to_intermediate_prediction_used': False,
        'full_train_split_used': args.case_limit == 0 and args.patch_limit == 0,
        'train_cases': train_names,
        'train_case_count': len(train_names),
        'test_cases_not_processed': test_names,
        'action_protocol': ACTION_SPECS,
        'patch_protocol': {
            'patches_per_case': args.patches_per_case,
            'foreground_patches_per_case': args.foreground_patches_per_case,
            'foreground_candidate_patches': args.foreground_candidate_patches,
            'foreground_threshold': args.foreground_threshold,
            'prediction_threshold': args.prediction_threshold,
            'selection_uses_gt': False,
            'short_volume_fill': 'deterministic repeat_fill',
        },
        'world_predictor': checkpoint_metadata,
        'A_state': state_metadata,
        'native_tail': {
            'start': 'decoder_stage_1_low_to_high post-block output',
            'stage_idx': 1,
            'continuation': 'original remaining decoder stages plus native final mask projection',
            'skip_context': 'source-side downstream skips',
            'mask_embedding_context': 'source-side mask embeddings',
        },
        'comparison_groups': {
            'full_real': 'full native VoxTell source/gamma/blur final predictions',
            'real_A_state_oracle': 'real action A state transplanted through source-side native tail',
            'existing_WP_imagined': 'existing conditioned WP predicted A state transplanted through source-side native tail',
        },
        'latent_transition_fidelity': latent_summary,
        'final_prediction_fidelity': final_summary,
        'oracle_vs_existing_wp_gap': {
            'foreground_pairwise_auroc': {
                'full_real': reliability['full_real']['pairwise_world_stability']['foreground']['auroc'],
                'real_A_state_oracle': reliability['real_A_state_oracle']['pairwise_world_stability']['foreground']['auroc'],
                'existing_WP_imagined': reliability['existing_WP_imagined']['pairwise_world_stability']['foreground']['auroc'],
            },
            'foreground_pairwise_spearman': {
                'full_real': reliability['full_real']['pairwise_world_stability']['foreground']['spearman'],
                'real_A_state_oracle': reliability['real_A_state_oracle']['pairwise_world_stability']['foreground']['spearman'],
                'existing_WP_imagined': reliability['existing_WP_imagined']['pairwise_world_stability']['foreground']['spearman'],
            },
        },
        'reliability': reliability,
        'region_reliability_means': region_reliability,
        'interpretation': {
            'comparison_rule': 'Compare existing_WP_imagined against real_A_state_oracle on native-tail final fidelity and reliability, then compare both against full_real.',
            'intermediate_head_isolated': 'Yes: no functional or intermediate segmentation head is used.',
            'transition_fidelity_isolated': 'Yes: World Predictor output is reconnected through the same native tail and source-side context.',
        },
        'outputs': {
            'latent_transition_fidelity': str(output_dir / 'latent_transition_fidelity.csv'),
            'final_prediction_fidelity': str(output_dir / 'final_prediction_fidelity.csv'),
            'reliability_metrics': str(output_dir / 'reliability_metrics.csv'),
            'reliability_regions': str(output_dir / 'reliability_regions.csv'),
            'patch_manifest': str(output_dir / 'patch_manifest.csv'),
            'summary': str(output_dir / 'summary.json'),
        },
    }


def run(args: argparse.Namespace) -> None:
    if ACTION_PROTOCOL != (('gamma', 0.30), ('blur', 1.5)):
        raise AssertionError('V9.1 action protocol must be gamma=0.30 and blur=1.5')
    if args.patches_per_case != 4 or args.foreground_patches_per_case != 2 or args.foreground_candidate_patches != 16:
        raise AssertionError('V9.1 patch protocol must match V9.0: 4/2/16')
    if args.foreground_threshold != 0.5 or args.prediction_threshold != 0.5:
        raise AssertionError('V9.1 thresholds must remain 0.5')
    if args.case_limit < 0 or args.patch_limit < 0:
        raise AssertionError('Debug limits must be non-negative')
    set_seed(args.seed)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    device = resolve_device(args)
    if args.device == 'cuda' and device.type != 'cuda':
        raise RuntimeError(f'V9.1 requires CUDA, resolved {device}')
    train_cases = iter_cases(paths, split='train')
    test_cases = iter_cases(paths, split='test')
    if len(train_cases) != 30:
        raise AssertionError(f'V9.1 requires the complete 30-case train split, got {len(train_cases)}')
    train_names = [case.case for case in train_cases]
    test_names = [case.case for case in test_cases]
    if set(train_names) & set(test_names):
        raise AssertionError('V9.1 train/test overlap')
    if args.case_limit:
        train_cases = train_cases[:args.case_limit]
        train_names = [case.case for case in train_cases]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_sink = CsvSink(output_dir / 'latent_transition_fidelity.csv', LATENT_FIELDS)
    final_sink = CsvSink(output_dir / 'final_prediction_fidelity.csv', FINAL_FIELDS)
    metric_sink = CsvSink(output_dir / 'reliability_metrics.csv', METRIC_FIELDS)
    region_sink = CsvSink(output_dir / 'reliability_regions.csv', REGION_FIELDS)
    manifest_sink = CsvSink(output_dir / 'patch_manifest.csv', PATCH_FIELDS)
    progress_path = output_dir / 'progress.json'
    completed_cases: list[str] = []
    patch_rows = 0
    write_progress(progress_path, completed_cases, patch_rows, device, 'initializing')
    global_metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
    global_regions: dict[tuple[str, str, str], RegionAccumulator] = {}
    latent_rows_all: list[dict[str, Any]] = []
    final_rows_all: list[dict[str, Any]] = []
    state_metadata: dict[str, Any] = {}

    try:
        checkpoint_path = Path(args.world_checkpoint)
        print(f'[V9.1] loading V8.0 conditioned World Predictor: {checkpoint_path}', flush=True)
        world_model, checkpoint_metadata = load_existing_world_predictor(checkpoint_path, device)
        print(f'[V9.1] selected stage={SELECTED_STAGE}', flush=True)
        interface = VoxTellStateInterface.from_model_dir(
            paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
        )
        prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
        text_backbone = getattr(interface.predictor, 'text_backbone', None)
        interface.predictor.text_backbone = None
        interface.predictor.tokenizer = None
        interface.predictor._text_embedding_cache.clear()
        del text_backbone
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        stage_spec = {'stage_idx': 1, 'state_kind': 'stage_output'}
        for case_index, case in enumerate(train_cases, start=1):
            print(f'[V9.1] case {case_index}/{len(train_cases)} start {case.case}', flush=True)
            image, label, _ = read_image_and_label(case)
            label_padded = pad_label_like_image(interface, label)
            original_padded, slicers, patch_kinds = select_patch_slicers(
                interface, image, prompt_embedding,
                args.patches_per_case, args.foreground_patches_per_case,
                args.foreground_candidate_patches, args.foreground_threshold,
            )
            slicers, patch_kinds = pad_selected_patches(slicers, patch_kinds, args.patches_per_case)
            if args.patch_limit:
                slicers, patch_kinds = slicers[:args.patch_limit], patch_kinds[:args.patch_limit]
            action_padded = {
                action: padded_visual_action_and_slicers(interface.predictor, image, action, spec['strength'])[0]
                for action, spec in ACTION_SPECS.items()
            }
            case_metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
            case_regions: dict[tuple[str, str, str], RegionAccumulator] = {}
            for patch_zero, slicer in enumerate(slicers):
                patch_kind = patch_kinds[patch_zero]
                manifest_sink.write({
                    'case': case.case, 'patch_index': patch_zero,
                    'patch_kind': patch_kind, 'slicer': repr(slicer),
                })
                source_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
                source_context = interface.forward_with_audit_context(source_patch, prompt_embedding)
                source_A = get_candidate_state(source_context, stage_spec)
                source_full_logits = source_context['final_prediction'][:, :1]
                source_identity_logits = VoxTellStateInterface.native_tail_from_audit_context(
                    source_context, 1, 'stage_output', source_A,
                    mask_context=source_context, skip_context=source_context,
                )
                if not state_metadata:
                    state_metadata = {
                        'name': SELECTED_STAGE,
                        'stage_idx': 1,
                        'state_kind': 'stage_output',
                        'location': 'decoder_stage_1_low_to_high post-block output',
                        'input_shape': 'x'.join(str(size) for size in source_A.shape),
                        'output_shape': 'x'.join(str(size) for size in source_A.shape),
                        'channels': int(source_A.shape[1]),
                        'spatial_resolution': 'x'.join(str(size) for size in source_A.shape[-3:]),
                    }
                if int(source_A.shape[1]) != checkpoint_metadata['input_channels']:
                    raise AssertionError(
                        f'V8.0 WP input channels {checkpoint_metadata["input_channels"]} != A channels {source_A.shape[1]}'
                    )
                gt_np = (label_padded[slicer][None].detach().cpu().numpy() == args.label_value).astype(bool)
                source_full_probability = torch.sigmoid(source_full_logits).detach().float().cpu().numpy()
                source_A_probability = torch.sigmoid(source_identity_logits).detach().float().cpu().numpy()
                group_probabilities: dict[str, dict[str, np.ndarray]] = {
                    'full_real': {'source': source_full_probability},
                    'real_A_state_oracle': {'source': source_A_probability},
                    'existing_WP_imagined': {'source': source_A_probability},
                }
                for action, action_spec in ACTION_SPECS.items():
                    print(f'[V9.1] case {case_index}/{len(train_cases)} patch {patch_zero + 1}/{len(slicers)} action={action}', flush=True)
                    action_patch = torch.clone(action_padded[action][slicer][None], memory_format=torch.contiguous_format)
                    action_context = interface.forward_with_audit_context(action_patch, prompt_embedding)
                    real_A = get_candidate_state(action_context, stage_spec)
                    action_vector = visual_action(action, action_spec['strength'], device)
                    predicted_A = predict_world_state(world_model, source_A, action_vector)
                    latent_patch_rows = latent_rows(
                        source_A, real_A, predicted_A, case.case, patch_zero, patch_kind, action,
                    )
                    for row in latent_patch_rows:
                        latent_sink.write(row)
                    latent_rows_all.extend(latent_patch_rows)
                    final_patch_rows, final_logits = final_rows(
                        source_context, action_context, source_A, predicted_A, real_A,
                        case.case, patch_zero, patch_kind, action, args.prediction_threshold,
                    )
                    for row in final_patch_rows:
                        final_sink.write(row)
                    final_rows_all.extend(final_patch_rows)
                    real_action_probability = torch.sigmoid(action_context['final_prediction'][:, :1]).detach().float().cpu().numpy()
                    oracle_probability = torch.sigmoid(final_logits['real_A_state_oracle']).detach().float().cpu().numpy()
                    wp_probability = torch.sigmoid(final_logits['existing_WP_predicted']).detach().float().cpu().numpy()
                    group_probabilities['full_real'][action] = real_action_probability
                    group_probabilities['real_A_state_oracle'][action] = oracle_probability
                    group_probabilities['existing_WP_imagined'][action] = wp_probability
                    del action_patch, action_context, real_A, predicted_A, action_vector, final_logits
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                for group in GROUPS:
                    reliability_patch_update(
                        group,
                        group_probabilities[group]['source'],
                        group_probabilities[group]['gamma'],
                        group_probabilities[group]['blur'],
                        gt_np, args.prediction_threshold, case.case, patch_zero,
                        case_metrics, case_regions,
                    )
                del group_probabilities, gt_np, source_context, source_patch, source_A, source_full_logits, source_identity_logits
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                patch_rows += 1
            write_case_and_global_reliability_rows(
                metric_sink, region_sink, case_metrics, case_regions,
                global_metrics, global_regions, case.case,
            )
            completed_cases.append(case.case)
            write_progress(progress_path, completed_cases, patch_rows, device, 'running')
            print(f'[V9.1] case {case_index}/{len(train_cases)} complete memory={memory_status(device)}', flush=True)
            del case_metrics, case_regions, action_padded, original_padded, label_padded, image, label
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        for key, accumulator in sorted(global_metrics.items()):
            group, method, scope = key
            metric_sink.write({
                'candidate_state': group, 'case': '__all__', 'scope': scope,
                'method': method, 'patch_count': accumulator.patch_count,
                **accumulator.values(), 'aggregation': 'full_train_case_macro',
            })
        for key, accumulator in sorted(global_regions.items()):
            group, method, region = key
            region_sink.write({
                'candidate_state': group, 'case': '__all__', 'method': method,
                'region': region, 'voxel_count': accumulator.voxel_count,
                'reliability_mean': accumulator.mean(), 'aggregation': 'full_train_case_macro',
            })
        summary = build_summary(
            args, train_names, test_names, checkpoint_metadata, state_metadata,
            latent_rows_all, final_rows_all, global_metrics, global_regions, output_dir,
        )
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        write_progress(progress_path, completed_cases, patch_rows, device, 'complete')
        print('[V9.1] complete', flush=True)
    finally:
        for sink in (latent_sink, final_sink, metric_sink, region_sink, manifest_sink):
            sink.close()


if __name__ == '__main__':
    run(parse_args())
