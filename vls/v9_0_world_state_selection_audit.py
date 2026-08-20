"""V9.0 World State Selection Audit.

Diagnostic-only audit of real augmentation effects at several native VoxTell
decoder boundaries.  It does not train or load a World Predictor, does not
change VoxTell parameters, and never creates a functional segmentation head.
All final predictions are produced by the original VoxTell decoder tail and
its native mask projection.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.data import iter_cases, read_image_and_label
from vls.v2_experiment import (
    padded_visual_action_and_slicers,
    resolve_device,
    select_patch_slicers,
)
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, pad_label_like_image
from vls.v7_0d_protocol_sanity import binary_metrics, set_seed
from vls.v8_3_real_aug_oracle_reliability import (
    rank_metrics,
    real_augmentation_reliabilities,
)
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import normalized_mse


ACTION_PROTOCOL = (('gamma', 0.30), ('blur', 1.5))
CANDIDATE_NAMES = (
    'decoder_stage_1_low_to_high',
    'penultimate_decoder_stage_output',
    'final_decoder_stage_input',
    'final_decoder_feature',
    'full_voxtell_reference',
)
RELIABILITY_METHODS = (
    'confidence_rank',
    'gamma_world_stability',
    'blur_world_stability',
    'pairwise_world_stability',
    'gamma_joint_product',
    'blur_joint_product',
    'pairwise_joint_product',
)
SCOPES = ('overall', 'foreground', 'background')
REGIONS = ('TP', 'FP', 'FN', 'TN')
METRIC_NAMES = ('pseudo_accuracy', 'reliability_mean', 'auroc', 'auprc', 'spearman')


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description='V9.0 native VoxTell world-state selection audit')
    parser.add_argument('--model-dir', default=str(paths.voxtell_model_dir))
    parser.add_argument('--voxtell-root', default=str(paths.voxtell_root))
    parser.add_argument('--data-root', default=str(paths.data_root))
    parser.add_argument('--split-json', default=str(paths.split_json))
    parser.add_argument('--output-dir', default='outputs/v9_0_world_state_selection_audit')
    parser.add_argument('--patches-per-case', type=int, default=4)
    parser.add_argument('--foreground-patches-per-case', type=int, default=2)
    parser.add_argument('--foreground-candidate-patches', type=int, default=16)
    parser.add_argument('--foreground-threshold', type=float, default=0.5)
    parser.add_argument('--prediction-threshold', type=float, default=0.5)
    parser.add_argument('--label-value', type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--case-limit', type=int, default=0, help='debug smoke-test limit; 0 uses all train cases')
    parser.add_argument('--patch-limit', type=int, default=0, help='debug smoke-test limit; 0 uses all selected patches')
    parser.add_argument('--resume', action='store_true', help='append to an existing run and skip completed train cases')
    parser.add_argument('--rebuild-summary-only', action='store_true', help='rebuild summary/ranking from existing CSVs without model inference')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


class CsvSink:
    def __init__(self, path: Path, fields: tuple[str, ...], append: bool = False) -> None:
        existing = append and path.exists() and path.stat().st_size > 0
        self.handle = path.open('a' if existing else 'w', newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=list(fields), extrasaction='ignore')
        if not existing:
            self.writer.writeheader()
        self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: Any) -> None:
        if value is None:
            return
        value = float(value)
        if np.isfinite(value):
            self.total += value
            self.count += 1

    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


@dataclass
class MetricAccumulator:
    patch_count: int = 0
    metrics: dict[str, RunningMean] = field(default_factory=dict)

    def update(self, row: dict[str, Any]) -> None:
        self.patch_count += 1
        for name in METRIC_NAMES:
            self.metrics.setdefault(name, RunningMean()).update(row.get(name))

    def values(self) -> dict[str, Any]:
        return {name: self.metrics.get(name, RunningMean()).mean() for name in METRIC_NAMES}


@dataclass
class RegionAccumulator:
    values: RunningMean = field(default_factory=RunningMean)
    voxel_count: int = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size:
            self.voxel_count += int(values.size)
            self.values.update(float(values.mean()))

    def mean(self) -> float | None:
        return self.values.mean()


ACTION_FIELDS = (
    'case', 'patch_index', 'patch_kind', 'candidate_state', 'action',
    'state_shape', 'channels', 'spatial_resolution', 'tensor_memory_mb',
    'normalized_mse', 'delta_rms', 'delta_l2_norm', 'relative_delta_norm',
    'relative_delta_norm_valid', 'gamma_vs_blur_delta_cosine',
    'gamma_vs_blur_delta_cosine_valid', 'state_variance', 'delta_variance',
)
RECOVERY_FIELDS = (
    'case', 'patch_index', 'patch_kind', 'candidate_state', 'action',
    'state_shape', 'context_mode', 'skip_context', 'mask_embedding_context',
    'probability_mse', 'logits_mse', 'mean_absolute_probability_error',
    'predicted_mask_dice_vs_real', 'predicted_mask_iou_vs_real',
)
CONTEXT_FIELDS = (
    'case', 'patch_index', 'patch_kind', 'candidate_state', 'action',
    'context_test', 'skip_context', 'mask_embedding_context',
    'probability_mse', 'logits_mse', 'predicted_mask_dice_vs_real',
    'predicted_mask_iou_vs_real', 'mean_absolute_probability_error',
)
METRIC_FIELDS = (
    'candidate_state', 'case', 'scope', 'method', 'patch_count',
    'pseudo_accuracy', 'reliability_mean', 'auroc', 'auprc', 'spearman',
    'aggregation',
)
REGION_FIELDS = (
    'candidate_state', 'case', 'method', 'region', 'voxel_count',
    'reliability_mean', 'aggregation',
)
RANKING_FIELDS = (
    'candidate_state', 'gamma_relative_delta_norm', 'blur_relative_delta_norm',
    'gamma_recovery_probability_mse', 'blur_recovery_probability_mse',
    'gamma_recovery_mask_dice', 'blur_recovery_mask_dice',
    'foreground_gamma_auroc', 'foreground_gamma_spearman',
    'foreground_blur_auroc', 'foreground_blur_spearman',
    'foreground_pairwise_auroc', 'foreground_pairwise_spearman',
    'foreground_tp_reliability_mean', 'foreground_fp_reliability_mean',
    'foreground_fn_reliability_mean', 'foreground_tn_reliability_mean',
    'source_mask_hybrid_probability_mse', 'action_mask_probability_mse',
    'state_memory_mb',
)


def tensor_mb(value: torch.Tensor) -> float:
    return value.numel() * value.element_size() / (1024.0 ** 2)


def shape_text(value: torch.Tensor) -> str:
    return 'x'.join(str(size) for size in value.shape)


def spatial_text(value: torch.Tensor) -> str:
    return 'x'.join(str(size) for size in value.shape[-3:])


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


def write_progress(path: Path, completed_cases: list[str], patch_rows: int, device: torch.device, stage: str) -> None:
    path.write_text(json.dumps({
        'stage': stage,
        'completed_cases': completed_cases,
        'completed_case_count': len(completed_cases),
        'completed_patch_rows': patch_rows,
        'memory': memory_status(device),
    }, indent=2))


def load_resume_state(
    output_dir: Path,
    train_names: list[str],
) -> tuple[list[str], int]:
    progress_path = output_dir / 'progress.json'
    if not progress_path.exists():
        raise FileNotFoundError(f'--resume requested but progress file is missing: {progress_path}')
    progress = json.loads(progress_path.read_text())
    completed = [str(name) for name in progress.get('completed_cases', [])]
    expected_prefix = train_names[:len(completed)]
    if completed != expected_prefix:
        raise AssertionError(
            'Resume progress must contain an ordered prefix of the current train split; '
            f'got {completed[-3:]}, expected prefix ending {expected_prefix[-3:]}'
        )
    patch_rows = int(progress.get('completed_patch_rows', 0))
    if patch_rows < 0:
        raise AssertionError('Resume progress has a negative completed_patch_rows')
    return completed, patch_rows


def restore_resume_aggregates(
    output_dir: Path,
    completed_cases: set[str],
    global_metrics: dict[tuple[str, str, str], MetricAccumulator],
    global_regions: dict[tuple[str, str, str], RegionAccumulator],
) -> None:
    """Restore case-level scalar aggregates without retaining patch tensors."""
    metrics_path = output_dir / 'candidate_oracle_reliability_metrics.csv'
    if metrics_path.exists():
        with metrics_path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('case') not in completed_cases or row.get('aggregation') != 'case_patch_macro':
                    continue
                key = (row['candidate_state'], row['method'], row['scope'])
                global_metrics.setdefault(key, MetricAccumulator()).update({
                    name: None if row.get(name, '') in ('', 'None') else float(row[name])
                    for name in METRIC_NAMES
                })
    regions_path = output_dir / 'candidate_oracle_reliability_regions.csv'
    if regions_path.exists():
        with regions_path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('case') not in completed_cases or row.get('aggregation') != 'case_patch_macro':
                    continue
                key = (row['candidate_state'], row['method'], row['region'])
                accumulator = global_regions.setdefault(key, RegionAccumulator())
                mean = row.get('reliability_mean', '')
                if mean not in ('', 'None'):
                    accumulator.values.update(float(mean))
                count = row.get('voxel_count', '')
                if count not in ('', 'None'):
                    accumulator.voxel_count += int(float(count))


def pad_selected_patches(
    slicers: list[tuple],
    patch_kinds: list[str],
    required_count: int,
) -> tuple[list[tuple], list[str]]:
    """Deterministically repeat source-selected patches when a volume has <4 windows."""
    if len(slicers) >= required_count:
        return list(slicers[:required_count]), list(patch_kinds[:required_count])
    if not slicers:
        raise AssertionError('Patch selector returned no usable patches')
    padded_slicers = list(slicers)
    padded_kinds = list(patch_kinds)
    source_pool = list(slicers)
    repeat_index = 0
    while len(padded_slicers) < required_count:
        padded_slicers.append(source_pool[repeat_index % len(source_pool)])
        padded_kinds.append('repeat_fill')
        repeat_index += 1
    return padded_slicers, padded_kinds


def parse_csv_value(value: str) -> Any:
    if value in ('', 'None', 'null'):
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def restore_resume_detail_rows(
    output_dir: Path,
    completed_cases: set[str],
    sensitivity_rows: dict[tuple[str, str, str], list[dict[str, Any]]],
    recovery_rows: dict[tuple[str, str, str], list[dict[str, Any]]],
    context_rows: dict[tuple[str, str, str], list[dict[str, Any]]],
    state_metadata: dict[str, dict[str, Any]],
) -> None:
    """Restore all scalar detail rows needed by V9.0 summary/ranking."""
    sources = (
        ('candidate_action_sensitivity.csv', sensitivity_rows, 'action', 'all'),
        ('candidate_final_prediction_recovery.csv', recovery_rows, 'context_mode', 'source_context'),
        ('candidate_context_dependency.csv', context_rows, 'context_test', None),
    )
    for filename, target, grouping_field, fixed_group in sources:
        path = output_dir / filename
        if not path.exists():
            continue
        with path.open(newline='') as handle:
            for raw in csv.DictReader(handle):
                if raw.get('case') not in completed_cases:
                    continue
                row = {key: parse_csv_value(value) for key, value in raw.items()}
                candidate = str(row['candidate_state'])
                action = str(row.get('action', ''))
                group = fixed_group if fixed_group is not None else str(row.get(grouping_field, ''))
                target[(candidate, action, group)].append(row)
                if filename == 'candidate_action_sensitivity.csv':
                    state_metadata.setdefault(candidate, {
                        'state_name': candidate,
                        'state_kind': 'restored_from_csv',
                        'stage_idx': None,
                        'location': 'restored from existing V9.0 action-sensitivity CSV',
                        'tensor_shape': row.get('state_shape'),
                        'channels': row.get('channels'),
                        'spatial_resolution': row.get('spatial_resolution'),
                        'tensor_memory_mb': row.get('tensor_memory_mb'),
                    })


def rebuild_summary_only(args: argparse.Namespace, paths: ProjectPaths, output_dir: Path) -> None:
    train_cases = iter_cases(paths, split='train')
    test_cases = iter_cases(paths, split='test')
    train_names = [case.case for case in train_cases]
    test_names = [case.case for case in test_cases]
    completed_cases, _ = load_resume_state(output_dir, train_names)
    if completed_cases != train_names:
        raise AssertionError('--rebuild-summary-only requires all train cases to be complete')
    existing_summary_path = output_dir / 'summary.json'
    existing_summary = json.loads(existing_summary_path.read_text()) if existing_summary_path.exists() else {}
    definitions = existing_summary.get('candidate_definitions', {})
    specs = {
        name: definitions[name]
        for name in CANDIDATE_NAMES
        if name != 'full_voxtell_reference' and name in definitions
    }
    global_metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
    global_regions: dict[tuple[str, str, str], RegionAccumulator] = {}
    restore_resume_aggregates(output_dir, set(completed_cases), global_metrics, global_regions)
    sensitivity_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    recovery_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    context_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    state_metadata: dict[str, dict[str, Any]] = {}
    restore_resume_detail_rows(
        output_dir, set(completed_cases), sensitivity_rows, recovery_rows,
        context_rows, state_metadata,
    )
    summary = build_summary(
        args, paths, train_names, test_names, specs, state_metadata,
        global_metrics, global_regions, sensitivity_rows, recovery_rows,
        context_rows, output_dir,
    )
    summary['resume']['summary_rebuilt_from_existing_csv'] = True
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    ranking_sink = CsvSink(output_dir / 'candidate_ranking.csv', RANKING_FIELDS, append=False)
    try:
        write_ranking_csv(ranking_sink, summary['candidate_summaries'])
    finally:
        ranking_sink.close()
    print(json.dumps({'summary': str(output_dir / 'summary.json'), 'inference': False}, indent=2))


def candidate_specs(interface: VoxTellStateInterface) -> dict[str, dict[str, Any]]:
    stage_count = len(interface.network.decoder.stages)
    if stage_count < 3:
        raise AssertionError('V9.0 requires at least three decoder stages for A/B/C/D')
    final_idx = stage_count - 1
    return {
        'decoder_stage_1_low_to_high': {
            'stage_idx': 1,
            'state_kind': 'stage_output',
            'location': 'decoder stage index 1 post-block output (legacy A)',
        },
        'penultimate_decoder_stage_output': {
            'stage_idx': final_idx - 1,
            'state_kind': 'stage_output',
            'location': 'decoder stage index final-1 post-block output (B)',
        },
        'final_decoder_stage_input': {
            'stage_idx': final_idx,
            'state_kind': 'final_input',
            'location': 'final transpconv + final encoder skip concat, before final decoder block (C)',
        },
        'final_decoder_feature': {
            'stage_idx': final_idx,
            'state_kind': 'final_output',
            'location': 'last decoder stage output before native final mask projection (D)',
        },
    }


def get_candidate_state(context: dict[str, Any], spec: dict[str, Any]) -> torch.Tensor:
    audit = context['decoder_audit']
    key = f"decoder_stage_{spec['stage_idx']}_low_to_high"
    if spec['state_kind'] == 'final_input':
        return audit['stage_inputs'][key]
    return audit['decoder_states'][key]


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def action_sensitivity_rows(
    source_state: torch.Tensor,
    action_state: torch.Tensor,
    gamma_state: torch.Tensor,
    blur_state: torch.Tensor,
    case: str,
    patch_index: int,
    patch_kind: str,
    candidate: str,
    action: str,
) -> dict[str, Any]:
    source = source_state.float()
    target = action_state.float()
    delta = target - source
    gamma_delta = gamma_state.float() - source
    blur_delta = blur_state.float() - source
    source_norm = torch.linalg.vector_norm(source.reshape(-1))
    denominator = source_norm.clamp_min(1e-6)
    delta_norm = torch.linalg.vector_norm(delta.reshape(-1))
    cosine_denominator = torch.linalg.vector_norm(gamma_delta.reshape(-1)) * torch.linalg.vector_norm(blur_delta.reshape(-1))
    relative_valid = scalar(source_norm) > 1e-12
    cosine_valid = scalar(cosine_denominator) > 1e-12
    cosine = None if not cosine_valid else scalar(torch.sum(gamma_delta * blur_delta) / cosine_denominator)
    return {
        'case': case,
        'patch_index': patch_index,
        'patch_kind': patch_kind,
        'candidate_state': candidate,
        'action': action,
        'state_shape': shape_text(source_state),
        'channels': int(source_state.shape[1]),
        'spatial_resolution': spatial_text(source_state),
        'tensor_memory_mb': tensor_mb(source_state),
        'normalized_mse': scalar(normalized_mse(source, target)),
        'delta_rms': scalar(torch.sqrt(torch.mean(delta.square()))),
        'delta_l2_norm': scalar(delta_norm),
        'relative_delta_norm': scalar(delta_norm / denominator),
        'relative_delta_norm_valid': relative_valid,
        'gamma_vs_blur_delta_cosine': cosine,
        'gamma_vs_blur_delta_cosine_valid': cosine_valid,
        'state_variance': scalar(source.var(unbiased=False)),
        'delta_variance': scalar(delta.var(unbiased=False)),
    }


def final_prediction_metrics(
    transplant_logits: torch.Tensor,
    real_logits: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    transplant_logits = transplant_logits.float()
    real_logits = real_logits.float()
    transplant_probability = torch.sigmoid(transplant_logits)
    real_probability = torch.sigmoid(real_logits)
    transplant_mask = (transplant_probability > threshold).detach().cpu().numpy()
    real_mask = (real_probability > threshold).detach().cpu().numpy()
    overlap = binary_metrics(transplant_mask, real_mask)
    return {
        'probability_mse': scalar(torch.mean((transplant_probability - real_probability).square())),
        'logits_mse': scalar(torch.mean((transplant_logits - real_logits).square())),
        'mean_absolute_probability_error': scalar(torch.mean(torch.abs(transplant_probability - real_probability))),
        'predicted_mask_dice_vs_real': float(overlap['dice']),
        'predicted_mask_iou_vs_real': float(
            overlap['tp'] / max(overlap['tp'] + overlap['fp'] + overlap['fn'], 1)
        ),
    }


def candidate_reliability_maps(
    source_probability: np.ndarray,
    gamma_probability: np.ndarray,
    blur_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    base = real_augmentation_reliabilities(source_probability, gamma_probability, blur_probability)
    return {
        'confidence_rank': base['confidence_rank'],
        'gamma_world_stability': base['real_gamma_world_stability'],
        'blur_world_stability': base['real_blur_world_stability'],
        'pairwise_world_stability': base['real_pairwise_world_stability'],
        'gamma_joint_product': base['real_gamma_joint_product'],
        'blur_joint_product': base['real_blur_joint_product'],
        'pairwise_joint_product': base['real_pairwise_joint_product'],
    }


def reliability_patch_update(
    candidate: str,
    source_probability: np.ndarray,
    gamma_probability: np.ndarray,
    blur_probability: np.ndarray,
    gt: np.ndarray,
    threshold: float,
    case: str,
    patch_index: int,
    case_metrics: dict[tuple[str, str, str], MetricAccumulator],
    case_regions: dict[tuple[str, str, str], RegionAccumulator],
) -> None:
    reliability_maps = candidate_reliability_maps(source_probability, gamma_probability, blur_probability)
    pseudo = source_probability > threshold
    correct = pseudo == gt
    flat_gt = gt.reshape(-1)
    flat_pseudo = pseudo.reshape(-1)
    flat_correct = correct.reshape(-1)
    scopes = {
        'overall': np.ones(flat_gt.size, dtype=bool),
        'foreground': flat_gt,
        'background': ~flat_gt,
    }
    regions = {
        'TP': flat_pseudo & flat_gt,
        'FP': flat_pseudo & ~flat_gt,
        'FN': ~flat_pseudo & flat_gt,
        'TN': ~flat_pseudo & ~flat_gt,
    }
    for method, reliability in reliability_maps.items():
        flat_reliability = reliability.reshape(-1)
        for scope, mask in scopes.items():
            metrics = rank_metrics(flat_reliability, flat_correct, mask)
            case_metrics.setdefault((candidate, method, scope), MetricAccumulator()).update(metrics)
        for region, mask in regions.items():
            case_regions.setdefault((candidate, method, region), RegionAccumulator()).update(flat_reliability[mask])
    del reliability_maps, pseudo, correct, flat_gt, flat_pseudo, flat_correct, scopes, regions


def metric_row(candidate: str, case: str, scope: str, method: str, accumulator: MetricAccumulator, aggregation: str) -> dict[str, Any]:
    return {
        'candidate_state': candidate,
        'case': case,
        'scope': scope,
        'method': method,
        'patch_count': accumulator.patch_count,
        **accumulator.values(),
        'aggregation': aggregation,
    }


def region_row(candidate: str, case: str, method: str, region: str, accumulator: RegionAccumulator, aggregation: str) -> dict[str, Any]:
    return {
        'candidate_state': candidate,
        'case': case,
        'method': method,
        'region': region,
        'voxel_count': accumulator.voxel_count,
        'reliability_mean': accumulator.mean(),
        'aggregation': aggregation,
    }


def write_case_and_global_reliability_rows(
    metric_sink: CsvSink,
    region_sink: CsvSink,
    case_metrics: dict[tuple[str, str, str], MetricAccumulator],
    case_regions: dict[tuple[str, str, str], RegionAccumulator],
    global_metrics: dict[tuple[str, str, str], MetricAccumulator],
    global_regions: dict[tuple[str, str, str], RegionAccumulator],
    case_name: str,
) -> None:
    for key, accumulator in sorted(case_metrics.items()):
        candidate, method, scope = key
        row = metric_row(candidate, case_name, scope, method, accumulator, 'case_patch_macro')
        metric_sink.write(row)
        global_metrics.setdefault(key, MetricAccumulator()).update(accumulator.values())
    for key, accumulator in sorted(case_regions.items()):
        candidate, method, region = key
        row = region_row(candidate, case_name, method, region, accumulator, 'case_patch_macro')
        region_sink.write(row)
        global_regions.setdefault(key, RegionAccumulator()).values.update(accumulator.mean())
        global_regions[key].voxel_count += accumulator.voxel_count


def mean_rows(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def run(args: argparse.Namespace) -> None:
    if ACTION_PROTOCOL != (('gamma', 0.30), ('blur', 1.5)):
        raise AssertionError('V9.0 action protocol must remain gamma=0.30 and blur=1.5')
    if args.patches_per_case != 4 or args.foreground_patches_per_case != 2 or args.foreground_candidate_patches != 16:
        raise AssertionError('V9.0 defaults must match the V8.2 patch protocol (4/2/16)')
    if args.foreground_threshold != 0.5 or args.prediction_threshold != 0.5:
        raise AssertionError('V9.0 defaults require threshold=0.5')
    if args.patches_per_case <= 0 or args.foreground_patches_per_case < 0:
        raise AssertionError('Invalid patch protocol')
    if args.case_limit < 0 or args.patch_limit < 0:
        raise AssertionError('Debug limits must be non-negative')
    if args.resume and (args.case_limit or args.patch_limit):
        raise AssertionError('--resume cannot be combined with debug case/patch limits')

    set_seed(args.seed)
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    if args.rebuild_summary_only:
        rebuild_summary_only(args, paths, Path(args.output_dir))
        return
    device = resolve_device(args)
    if args.device == 'cuda' and device.type != 'cuda':
        raise RuntimeError(f'V9.0 requires CUDA, resolved {device}')
    train_cases = iter_cases(paths, split='train')
    train_cases_again = iter_cases(paths, split='train')
    test_cases = iter_cases(paths, split='test')
    train_names = [case.case for case in train_cases]
    test_names = [case.case for case in test_cases]
    if not train_cases or train_names != [case.case for case in train_cases_again]:
        raise AssertionError('V9.0 requires the complete stable non-empty train split')
    if set(train_names) & set(test_names):
        raise AssertionError('V9.0 train/test split overlap')
    if args.case_limit:
        train_cases = train_cases[:args.case_limit]
        train_names = [case.case for case in train_cases]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        existing_completed, _ = load_resume_state(output_dir, train_names)
        if len(existing_completed) == len(train_names):
            rebuild_summary_only(args, paths, output_dir)
            return
    action_sink = CsvSink(output_dir / 'candidate_action_sensitivity.csv', ACTION_FIELDS, append=args.resume)
    recovery_sink = CsvSink(output_dir / 'candidate_final_prediction_recovery.csv', RECOVERY_FIELDS, append=args.resume)
    context_sink = CsvSink(output_dir / 'candidate_context_dependency.csv', CONTEXT_FIELDS, append=args.resume)
    metric_sink = CsvSink(output_dir / 'candidate_oracle_reliability_metrics.csv', METRIC_FIELDS, append=args.resume)
    region_sink = CsvSink(output_dir / 'candidate_oracle_reliability_regions.csv', REGION_FIELDS, append=args.resume)
    ranking_sink = CsvSink(output_dir / 'candidate_ranking.csv', RANKING_FIELDS, append=False)
    progress_path = output_dir / 'progress.json'
    if args.resume:
        completed_cases, patch_rows = load_resume_state(output_dir, train_names)
    else:
        completed_cases, patch_rows = [], 0

    global_metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
    global_regions: dict[tuple[str, str, str], RegionAccumulator] = {}
    sensitivity_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    recovery_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    context_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    state_metadata: dict[str, dict[str, Any]] = {}
    if args.resume:
        restore_resume_aggregates(output_dir, set(completed_cases), global_metrics, global_regions)
        write_progress(progress_path, completed_cases, patch_rows, device, 'resuming')
    else:
        write_progress(progress_path, completed_cases, patch_rows, device, 'initializing')

    try:
        print('[V9.0] loading frozen VoxTell/base model; no World Predictor', flush=True)
        interface = VoxTellStateInterface.from_model_dir(
            paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
        )
        specs = candidate_specs(interface)
        if args.resume:
            restore_resume_detail_rows(
                output_dir, set(completed_cases), sensitivity_rows,
                recovery_rows, context_rows, state_metadata,
            )
            for candidate, spec in specs.items():
                if candidate in state_metadata:
                    state_metadata[candidate].update({
                        'state_kind': spec['state_kind'],
                        'stage_idx': spec['stage_idx'],
                        'location': spec['location'],
                    })
        prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
        text_backbone = getattr(interface.predictor, 'text_backbone', None)
        interface.predictor.text_backbone = None
        interface.predictor.tokenizer = None
        interface.predictor._text_embedding_cache.clear()
        del text_backbone
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        write_progress(progress_path, completed_cases, patch_rows, device, 'running')

        for case_index, case in enumerate(train_cases, start=1):
            if case.case in completed_cases:
                print(f'[V9.0] case {case_index}/{len(train_cases)} skip completed {case.case}', flush=True)
                continue
            print(f'[V9.0] case {case_index}/{len(train_cases)} start {case.case}', flush=True)
            image, label, _ = read_image_and_label(case)
            label_padded = pad_label_like_image(interface, label)
            original_padded, slicers, patch_kinds = select_patch_slicers(
                interface, image, prompt_embedding,
                args.patches_per_case, args.foreground_patches_per_case,
                args.foreground_candidate_patches, args.foreground_threshold,
            )
            slicers, patch_kinds = pad_selected_patches(
                slicers, patch_kinds, args.patches_per_case,
            )
            if args.patch_limit:
                slicers = slicers[:args.patch_limit]
                patch_kinds = patch_kinds[:args.patch_limit]
            action_padded = {
                action: padded_visual_action_and_slicers(interface.predictor, image, action, strength)[0]
                for action, strength in ACTION_PROTOCOL
            }
            if any(tuple(value.shape) != tuple(original_padded.shape) for value in action_padded.values()):
                raise AssertionError(f'Action padded shape mismatch for {case.case}')

            case_metrics: dict[tuple[str, str, str], MetricAccumulator] = {}
            case_regions: dict[tuple[str, str, str], RegionAccumulator] = {}
            for patch_zero, slicer in enumerate(slicers):
                patch_index = patch_zero + 1
                patch_kind = patch_kinds[patch_zero]
                print(f'[V9.0] case {case_index}/{len(train_cases)} patch {patch_index}/{len(slicers)} source forward', flush=True)
                source_patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
                source_context = interface.forward_with_audit_context(source_patch, prompt_embedding)
                source_full_logits = source_context['final_prediction'][:, :1]
                gt_np = (label_padded[slicer][None].detach().cpu().numpy() == args.label_value).astype(bool)
                full_probabilities = {
                    'source': torch.sigmoid(source_full_logits).detach().float().cpu().numpy(),
                }
                candidate_probabilities: dict[str, dict[str, np.ndarray]] = {
                    candidate: {'source': None} for candidate in specs
                }
                action_states_cpu: dict[str, dict[str, torch.Tensor]] = {
                    action: {} for action, _ in ACTION_PROTOCOL
                }
                for candidate, spec in specs.items():
                    source_state = get_candidate_state(source_context, spec)
                    candidate_probabilities[candidate]['source'] = torch.sigmoid(
                        interface.native_tail_from_audit_context(
                            source_context, spec['stage_idx'], spec['state_kind'], source_state,
                            mask_context=source_context, skip_context=source_context,
                        )
                    ).detach().float().cpu().numpy()
                    state_metadata.setdefault(candidate, {
                        'state_name': candidate,
                        'state_kind': spec['state_kind'],
                        'stage_idx': spec['stage_idx'],
                        'location': spec['location'],
                        'tensor_shape': shape_text(source_state),
                        'channels': int(source_state.shape[1]),
                        'spatial_resolution': spatial_text(source_state),
                        'tensor_memory_mb': tensor_mb(source_state),
                    })
                state_metadata.setdefault('full_voxtell_reference', {
                    'state_name': 'full_voxtell_reference',
                    'state_kind': 'native_final_prediction',
                    'stage_idx': None,
                    'location': 'native complete VoxTell final prediction; no transplant',
                    'tensor_shape': shape_text(source_full_logits),
                    'channels': int(source_full_logits.shape[1]),
                    'spatial_resolution': spatial_text(source_full_logits),
                    'tensor_memory_mb': tensor_mb(source_full_logits),
                })
                # Keep only one action context on the accelerator at a time.
                # Candidate states are copied to CPU after their native tail
                # and context checks finish, so three full decoder contexts are
                # never retained together.
                for action, _ in ACTION_PROTOCOL:
                    print(f'[V9.0] case {case_index}/{len(train_cases)} patch {patch_index}/{len(slicers)} action={action} forward', flush=True)
                    action_patch = torch.clone(action_padded[action][slicer][None], memory_format=torch.contiguous_format)
                    action_context = interface.forward_with_audit_context(action_patch, prompt_embedding)
                    del action_patch
                    full_probabilities[action] = torch.sigmoid(
                        action_context['final_prediction'][:, :1]
                    ).detach().float().cpu().numpy()
                    for candidate, spec in specs.items():
                        source_state = get_candidate_state(source_context, spec)
                        action_state = get_candidate_state(action_context, spec)
                        action_states_cpu[action][candidate] = action_state.detach().float().cpu()
                        # Standard transplant deliberately uses source-side
                        # skip/mask context, isolating information carried by
                        # the candidate state itself.
                        transplant_logits = interface.native_tail_from_audit_context(
                            action_context, spec['stage_idx'], spec['state_kind'], action_state,
                            mask_context=source_context, skip_context=source_context,
                        )
                        real_logits = action_context['final_prediction'][:, :1]
                        recovery = final_prediction_metrics(transplant_logits, real_logits, args.prediction_threshold)
                        recovery_row = {
                            'case': case.case,
                            'patch_index': patch_zero,
                            'patch_kind': patch_kind,
                            'candidate_state': candidate,
                            'action': action,
                            'state_shape': shape_text(action_state),
                            'context_mode': 'hybrid_candidate_action_state_source_tail_context',
                            'skip_context': 'source_side_downstream_skip',
                            'mask_embedding_context': 'source_side_mask_embedding',
                            **recovery,
                        }
                        recovery_sink.write(recovery_row)
                        recovery_rows[(candidate, action, 'source_context')].append(recovery_row)
                        candidate_probabilities[candidate][action] = torch.sigmoid(
                            transplant_logits
                        ).detach().float().cpu().numpy()

                        if candidate in ('final_decoder_stage_input', 'final_decoder_feature'):
                            action_state = get_candidate_state(action_context, spec)
                            real_logits = action_context['final_prediction'][:, :1]
                            source_mask_logits = interface.native_tail_from_audit_context(
                                action_context, spec['stage_idx'], spec['state_kind'], action_state,
                                mask_context=source_context, skip_context=source_context,
                            )
                            action_mask_logits = interface.native_tail_from_audit_context(
                                action_context, spec['stage_idx'], spec['state_kind'], action_state,
                                mask_context=action_context, skip_context=action_context,
                            )
                            source_mask_metrics = final_prediction_metrics(source_mask_logits, real_logits, args.prediction_threshold)
                            action_mask_metrics = final_prediction_metrics(action_mask_logits, real_logits, args.prediction_threshold)
                            for test_name, metrics, mask_mode in (
                                ('source_mask_embedding', source_mask_metrics, 'source_side_mask_embedding'),
                                ('action_mask_embedding', action_mask_metrics, 'action_specific_mask_embedding'),
                            ):
                                context_row = {
                                    'case': case.case,
                                    'patch_index': patch_zero,
                                    'patch_kind': patch_kind,
                                    'candidate_state': candidate,
                                    'action': action,
                                    'context_test': test_name,
                                    'skip_context': 'not_used_after_final_boundary; skip_embedded_in_candidate_state',
                                    'mask_embedding_context': mask_mode,
                                    **metrics,
                                }
                                context_sink.write(context_row)
                                context_rows[(candidate, action, test_name)].append(context_row)
                        print(f'[V9.0] case {case_index}/{len(train_cases)} patch {patch_index}/{len(slicers)} feature/segmentation done candidate={candidate} action={action}', flush=True)
                        del transplant_logits, real_logits, action_state, source_state
                    del action_context
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                reliability_patch_update(
                    'full_voxtell_reference', full_probabilities['source'], full_probabilities['gamma'],
                    full_probabilities['blur'], gt_np, args.prediction_threshold, case.case,
                    patch_zero, case_metrics, case_regions,
                )
                for candidate, spec in specs.items():
                    source_state = get_candidate_state(source_context, spec).detach().float().cpu()
                    gamma_state = action_states_cpu['gamma'][candidate]
                    blur_state = action_states_cpu['blur'][candidate]
                    for action, action_state in (('gamma', gamma_state), ('blur', blur_state)):
                        row = action_sensitivity_rows(
                            source_state, action_state, gamma_state, blur_state,
                            case.case, patch_zero, patch_kind, candidate, action,
                        )
                        action_sink.write(row)
                        sensitivity_rows[(candidate, action, 'all')].append(row)
                    reliability_patch_update(
                        candidate, candidate_probabilities[candidate]['source'],
                        candidate_probabilities[candidate]['gamma'], candidate_probabilities[candidate]['blur'],
                        gt_np, args.prediction_threshold, case.case, patch_zero,
                        case_metrics, case_regions,
                    )
                    print(f'[V9.0] case {case_index}/{len(train_cases)} patch {patch_index}/{len(slicers)} reliability done candidate={candidate}', flush=True)
                    del source_state, gamma_state, blur_state

                del full_probabilities, candidate_probabilities, action_states_cpu, gt_np, source_context, source_patch
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f'[V9.0] case {case_index}/{len(train_cases)} patch {patch_index}/{len(slicers)} complete memory={memory_status(device)}', flush=True)
                patch_rows += 1

            write_case_and_global_reliability_rows(
                metric_sink, region_sink, case_metrics, case_regions,
                global_metrics, global_regions, case.case,
            )
            completed_cases.append(case.case)
            write_progress(progress_path, completed_cases, patch_rows, device, 'running')
            print(f'[V9.0] case {case_index}/{len(train_cases)} complete memory={memory_status(device)}', flush=True)
            del case_metrics, case_regions, action_padded, original_padded, label_padded, image, label
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        for key, accumulator in sorted(global_metrics.items()):
            candidate, method, scope = key
            metric_sink.write(metric_row(candidate, '__all__', scope, method, accumulator, 'full_train_case_macro'))
        for key, accumulator in sorted(global_regions.items()):
            candidate, method, region = key
            region_sink.write(region_row(candidate, '__all__', method, region, accumulator, 'full_train_case_macro'))

        summary = build_summary(
            args, paths, train_names, test_names, specs, state_metadata,
            global_metrics, global_regions, sensitivity_rows, recovery_rows, context_rows,
            output_dir,
        )
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        write_ranking_csv(ranking_sink, summary['candidate_summaries'])
        write_progress(progress_path, completed_cases, patch_rows, device, 'complete')
        print('[V9.0] complete', flush=True)
    finally:
        for sink in (action_sink, recovery_sink, context_sink, metric_sink, region_sink, ranking_sink):
            sink.close()


def summary_metric(
    global_metrics: dict[tuple[str, str, str], MetricAccumulator],
    candidate: str,
    method: str,
    scope: str,
    name: str,
) -> float | None:
    accumulator = global_metrics.get((candidate, method, scope))
    return None if accumulator is None else accumulator.metrics.get(name, RunningMean()).mean()


def summary_region(
    global_regions: dict[tuple[str, str, str], RegionAccumulator],
    candidate: str,
    method: str,
    region: str,
) -> float | None:
    accumulator = global_regions.get((candidate, method, region))
    return None if accumulator is None else accumulator.mean()


def build_summary(
    args: argparse.Namespace,
    paths: ProjectPaths,
    train_names: list[str],
    test_names: list[str],
    specs: dict[str, dict[str, Any]],
    state_metadata: dict[str, dict[str, Any]],
    global_metrics: dict[tuple[str, str, str], MetricAccumulator],
    global_regions: dict[tuple[str, str, str], RegionAccumulator],
    sensitivity_rows: dict[tuple[str, str, str], list[dict[str, Any]]],
    recovery_rows: dict[tuple[str, str, str], list[dict[str, Any]]],
    context_rows: dict[tuple[str, str, str], list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    candidates = list(CANDIDATE_NAMES)
    candidate_summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_sensitivity = {}
        for action, _ in ACTION_PROTOCOL:
            rows = sensitivity_rows.get((candidate, action, 'all'), [])
            candidate_sensitivity[action] = {
                'relative_delta_norm': mean_rows(rows, 'relative_delta_norm'),
                'normalized_mse': mean_rows(rows, 'normalized_mse'),
                'delta_rms': mean_rows(rows, 'delta_rms'),
                'delta_l2_norm': mean_rows(rows, 'delta_l2_norm'),
                'gamma_vs_blur_delta_cosine': mean_rows(rows, 'gamma_vs_blur_delta_cosine'),
            }
        recovery = {}
        for action, _ in ACTION_PROTOCOL:
            rows = recovery_rows.get((candidate, action, 'source_context'), [])
            recovery[action] = {
                'probability_mse': mean_rows(rows, 'probability_mse'),
                'logits_mse': mean_rows(rows, 'logits_mse'),
                'mean_absolute_probability_error': mean_rows(rows, 'mean_absolute_probability_error'),
                'predicted_mask_dice_vs_real': mean_rows(rows, 'predicted_mask_dice_vs_real'),
                'predicted_mask_iou_vs_real': mean_rows(rows, 'predicted_mask_iou_vs_real'),
                'context': 'source-side downstream skip and source-side mask embedding',
            }
        reliability = {}
        for method in RELIABILITY_METHODS:
            reliability[method] = {
                scope: {
                    name: summary_metric(global_metrics, candidate, method, scope, name)
                    for name in METRIC_NAMES
                }
                for scope in SCOPES
            }
        regions = {
            method: {region: summary_region(global_regions, candidate, method, region) for region in REGIONS}
            for method in RELIABILITY_METHODS
        }
        context = {}
        for action, _ in ACTION_PROTOCOL:
            for test_name in ('source_mask_embedding', 'action_mask_embedding'):
                rows = context_rows.get((candidate, action, test_name), [])
                if rows:
                    context[f'{action}_{test_name}'] = {
                        'probability_mse': mean_rows(rows, 'probability_mse'),
                        'logits_mse': mean_rows(rows, 'logits_mse'),
                        'predicted_mask_dice_vs_real': mean_rows(rows, 'predicted_mask_dice_vs_real'),
                        'predicted_mask_iou_vs_real': mean_rows(rows, 'predicted_mask_iou_vs_real'),
                        'mean_absolute_probability_error': mean_rows(rows, 'mean_absolute_probability_error'),
                    }
        candidate_summaries[candidate] = {
            'state': state_metadata.get(candidate, {
                'state_name': candidate,
                'location': 'native full final prediction reference' if candidate == 'full_voxtell_reference' else specs.get(candidate),
            }),
            'action_sensitivity': candidate_sensitivity,
            'final_prediction_recovery': recovery,
            'oracle_reliability': reliability,
            'region_reliability_means': regions,
            'context_dependency': context,
            'descriptive_recommendation': descriptive_recommendation(candidate, candidate_summaries, candidate_sensitivity, recovery, context),
        }

    for candidate in candidates:
        if candidate not in candidate_summaries:
            continue
        if candidate_summaries[candidate]['descriptive_recommendation'] is None:
            candidate_summaries[candidate]['descriptive_recommendation'] = 'native reference or insufficient completed rows'

    return {
        'stage': 'V9.0 World State Selection Audit',
        'models_trained': False,
        'world_predictor_loaded': False,
        'gt_used_only_for_reliability_diagnosis': True,
        'extra_segmentation_head_used': False,
        'full_train_split_used': args.case_limit == 0 and args.patch_limit == 0,
        'train_cases': train_names,
        'train_case_count': len(train_names),
        'test_cases_not_processed': test_names,
        'test_cases_in_diagnostic': False,
        'action_protocol': {'gamma': 0.30, 'blur': 1.5},
        'patch_protocol': {
            'patches_per_case': args.patches_per_case,
            'foreground_patches_per_case': args.foreground_patches_per_case,
            'foreground_candidate_patches': args.foreground_candidate_patches,
            'foreground_threshold': args.foreground_threshold,
            'prediction_threshold': args.prediction_threshold,
            'selection_uses_gt': False,
            'short_volume_fill': 'deterministic repeat_fill of source-selected slicers when fewer than patches_per_case exist',
        },
        'resume': {
            'enabled': args.resume,
            'completed_cases_are_skipped': True,
            'csv_case_rows_are_appended': args.resume,
        },
        'candidate_definitions': {
            **{name: specs[name] for name in specs},
            'full_voxtell_reference': {
                'location': 'native complete VoxTell final prediction; no transplant',
            },
        },
        'native_tail_contract': {
            'A': 'replace stage 1 post-block output, then native remaining decoder stages and native mask projection',
            'B': 'replace penultimate post-block output, then native final decoder stage and native mask projection',
            'C': 'replace final pre-block input, then native final decoder block and native mask projection',
            'D': 'replace final post-block feature, then native mask projection',
            'hybrid_recovery_context': 'action candidate state with source-side downstream skips and mask embeddings',
            'context_dependency': 'C/D compare source-side versus action-specific mask embeddings; no downstream skip remains after C/D boundary',
        },
        'candidate_summaries': candidate_summaries,
        'outputs': {
            'candidate_action_sensitivity': str(output_dir / 'candidate_action_sensitivity.csv'),
            'candidate_final_prediction_recovery': str(output_dir / 'candidate_final_prediction_recovery.csv'),
            'candidate_context_dependency': str(output_dir / 'candidate_context_dependency.csv'),
            'candidate_oracle_reliability_metrics': str(output_dir / 'candidate_oracle_reliability_metrics.csv'),
            'candidate_oracle_reliability_regions': str(output_dir / 'candidate_oracle_reliability_regions.csv'),
            'candidate_ranking': str(output_dir / 'candidate_ranking.csv'),
            'progress': str(output_dir / 'progress.json'),
            'summary': str(output_dir / 'summary.json'),
        },
    }


def descriptive_recommendation(
    candidate: str,
    all_summaries: dict[str, dict[str, Any]],
    sensitivity: dict[str, dict[str, Any]],
    recovery: dict[str, dict[str, Any]],
    context: dict[str, dict[str, Any]],
) -> str:
    if candidate == 'full_voxtell_reference':
        return 'full native VoxTell reference for apples-to-apples oracle reliability; not a transplant candidate'
    statements: list[str] = []
    if all(value.get('relative_delta_norm') is not None and value.get('relative_delta_norm') > 0 for value in sensitivity.values()):
        statements.append('real gamma/blur augmentation enters this state')
    else:
        statements.append('one or more real augmentation shifts are zero or invalid')
    dice_values = [value.get('predicted_mask_dice_vs_real') for value in recovery.values()]
    if dice_values and all(value is not None for value in dice_values):
        statements.append('native-tail transplant recovery is measurable')
    else:
        statements.append('native-tail recovery is incomplete')
    if context:
        statements.append('mask-embedding context dependency is explicitly measured')
    return '; '.join(statements)


def write_ranking_csv(sink: CsvSink, summaries: dict[str, dict[str, Any]]) -> None:
    for candidate, summary in summaries.items():
        sensitivity = summary['action_sensitivity']
        recovery = summary['final_prediction_recovery']
        reliability = summary['oracle_reliability']
        regions = summary['region_reliability_means']
        context = summary['context_dependency']
        source_mask_errors = [
            value.get('probability_mse') for key, value in context.items() if key.endswith('source_mask_embedding')
        ]
        action_mask_errors = [
            value.get('probability_mse') for key, value in context.items() if key.endswith('action_mask_embedding')
        ]
        memory = summary['state'].get('tensor_memory_mb')
        sink.write({
            'candidate_state': candidate,
            'gamma_relative_delta_norm': sensitivity.get('gamma', {}).get('relative_delta_norm'),
            'blur_relative_delta_norm': sensitivity.get('blur', {}).get('relative_delta_norm'),
            'gamma_recovery_probability_mse': recovery.get('gamma', {}).get('probability_mse'),
            'blur_recovery_probability_mse': recovery.get('blur', {}).get('probability_mse'),
            'gamma_recovery_mask_dice': recovery.get('gamma', {}).get('predicted_mask_dice_vs_real'),
            'blur_recovery_mask_dice': recovery.get('blur', {}).get('predicted_mask_dice_vs_real'),
            'foreground_gamma_auroc': reliability.get('gamma_world_stability', {}).get('foreground', {}).get('auroc'),
            'foreground_gamma_spearman': reliability.get('gamma_world_stability', {}).get('foreground', {}).get('spearman'),
            'foreground_blur_auroc': reliability.get('blur_world_stability', {}).get('foreground', {}).get('auroc'),
            'foreground_blur_spearman': reliability.get('blur_world_stability', {}).get('foreground', {}).get('spearman'),
            'foreground_pairwise_auroc': reliability.get('pairwise_world_stability', {}).get('foreground', {}).get('auroc'),
            'foreground_pairwise_spearman': reliability.get('pairwise_world_stability', {}).get('foreground', {}).get('spearman'),
            'foreground_tp_reliability_mean': regions.get('pairwise_world_stability', {}).get('TP'),
            'foreground_fp_reliability_mean': regions.get('pairwise_world_stability', {}).get('FP'),
            'foreground_fn_reliability_mean': regions.get('pairwise_world_stability', {}).get('FN'),
            'foreground_tn_reliability_mean': regions.get('pairwise_world_stability', {}).get('TN'),
            'source_mask_hybrid_probability_mse': None if not source_mask_errors else float(np.mean([x for x in source_mask_errors if x is not None])),
            'action_mask_probability_mse': None if not action_mask_errors else float(np.mean([x for x in action_mask_errors if x is not None])),
            'state_memory_mb': memory,
        })


if __name__ == '__main__':
    run(parse_args())
