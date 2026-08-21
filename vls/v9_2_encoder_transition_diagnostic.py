"""Real-augmentation encoder transition diagnostic.

This diagnostic runs the frozen VoxTell encoder on source, real-gamma, and
real-blur patches.  It records feature transitions only; it does not load or
modify a World Predictor, decoder, loss, or segmentation head.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vls.config import ProjectPaths
from vls.data import iter_cases, read_image
from vls.v2_experiment import padded_visual_action_and_slicers, resolve_device, select_patch_slicers
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT
from vls.v7_0d_protocol_sanity import set_seed
from vls.voxtell_states import VoxTellStateInterface
from vls.v9_0_world_state_selection_audit import (
    ACTION_PROTOCOL,
    CsvSink,
    memory_status,
    pad_selected_patches,
    write_progress,
)


PROFILE_FIELDS = (
    'case_id', 'patch_index', 'patch_kind', 'action', 'layer',
    'feature_shape', 'delta_norm', 'relative_delta',
    'cosine_similarity', 'mse',
)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description='V9.2 real-augmentation VoxTell encoder transition diagnostic')
    parser.add_argument('--model-dir', default=str(paths.voxtell_model_dir))
    parser.add_argument('--voxtell-root', default=str(paths.voxtell_root))
    parser.add_argument('--data-root', default=str(paths.data_root))
    parser.add_argument('--split-json', default=str(paths.split_json))
    parser.add_argument('--output-dir', default='outputs/v9_2_encoder_transition_diagnostic')
    parser.add_argument('--patches-per-case', type=int, default=4)
    parser.add_argument('--foreground-patches-per-case', type=int, default=2)
    parser.add_argument('--foreground-candidate-patches', type=int, default=16)
    parser.add_argument('--foreground-threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case-limit', type=int, default=0, help='debug tiny-run limit; 0 uses all 30 train cases')
    parser.add_argument('--patch-limit', type=int, default=0, help='debug tiny-run limit; 0 uses all selected patches')
    return parser.parse_args()


def feature_names(feature_count: int) -> list[str]:
    if feature_count < 2:
        raise AssertionError(f'Expected multiple encoder outputs, got {feature_count}')
    names = [f'encoder_stage_{index}' for index in range(1, feature_count)]
    names.append('encoder_bottleneck_deepest')
    return names


@torch.inference_mode()
def encode_patch(interface: VoxTellStateInterface, patch: torch.Tensor) -> list[torch.Tensor]:
    network = interface.network.to(interface.device).eval()
    patch = patch.to(interface.device)
    context = (
        torch.autocast(interface.device.type, enabled=True)
        if interface.device.type == 'cuda' else nullcontext()
    )
    with context:
        features = network.encoder(patch)
    if not isinstance(features, (list, tuple)):
        raise TypeError(f'VoxTell encoder must return a list/tuple of skip features, got {type(features)}')
    return list(features)


def transition_row(
    source: torch.Tensor,
    transformed: torch.Tensor,
    case_id: str,
    patch_index: int,
    patch_kind: str,
    action: str,
    layer: str,
) -> dict[str, Any]:
    source = source.float()
    transformed = transformed.float()
    delta = transformed - source
    source_norm = torch.linalg.vector_norm(source.reshape(-1))
    delta_norm = torch.linalg.vector_norm(delta.reshape(-1))
    denominator = source_norm * torch.linalg.vector_norm(transformed.reshape(-1))
    cosine_valid = float(denominator.detach().cpu()) > 1e-12
    cosine = None if not cosine_valid else float(
        (torch.sum(source * transformed) / denominator).detach().cpu()
    )
    relative_valid = float(source_norm.detach().cpu()) > 1e-12
    relative = None if not relative_valid else float((delta_norm / source_norm).detach().cpu())
    return {
        'case_id': case_id,
        'patch_index': patch_index,
        'patch_kind': patch_kind,
        'action': action,
        'layer': layer,
        'feature_shape': 'x'.join(str(size) for size in source.shape),
        'delta_norm': float(delta_norm.detach().cpu()),
        'relative_delta': relative,
        'cosine_similarity': cosine,
        'mse': float(torch.mean(delta.square()).detach().cpu()),
    }


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def build_summary(
    args: argparse.Namespace,
    train_names: list[str],
    rows: list[dict[str, Any]],
    layer_metadata: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    layers = list(layer_metadata)
    by_layer_action: dict[str, dict[str, dict[str, Any]]] = {}
    for layer in layers:
        by_layer_action[layer] = {}
        for action, _ in ACTION_PROTOCOL:
            group = [row for row in rows if row['layer'] == layer and row['action'] == action]
            by_layer_action[layer][action] = {
                'count': len(group),
                'mean_delta_norm': mean_field(group, 'delta_norm'),
                'mean_relative_delta': mean_field(group, 'relative_delta'),
                'mean_cosine_similarity': mean_field(group, 'cosine_similarity'),
                'mean_mse': mean_field(group, 'mse'),
            }
    relative_by_action = {
        action: {
            layer: by_layer_action[layer][action]['mean_relative_delta']
            for layer in layers
        }
        for action, _ in ACTION_PROTOCOL
    }
    dominant_layer = {
        action: (
            max(
                ((layer, value) for layer, value in values.items() if value is not None),
                key=lambda item: item[1],
                default=(None, None),
            )[0]
        )
        for action, values in relative_by_action.items()
    }
    deepest_layer = layers[-1]
    shallow_layers = layers[: min(2, len(layers) - 1)]
    deep_mean = {
        action: relative_by_action[action].get(deepest_layer)
        for action, _ in ACTION_PROTOCOL
    }
    shallow_mean = {
        action: mean_field(
            [
                {'value': relative_by_action[action].get(layer)}
                for layer in shallow_layers
                if relative_by_action[action].get(layer) is not None
            ],
            'value',
        )
        for action, _ in ACTION_PROTOCOL
    }
    action_judgments = {}
    for action, _ in ACTION_PROTOCOL:
        if deep_mean[action] is None or shallow_mean[action] is None:
            action_judgments[action] = 'insufficient_valid_feature_norms'
        elif deep_mean[action] >= shallow_mean[action]:
            action_judgments[action] = 'deep_feature_change_is_at_least_as_large_as_shallow_mean; prioritize_WP_transition_learning_delta_or_residual_supervision'
        else:
            action_judgments[action] = 'shallow_feature_change_exceeds_deepest_mean; consider_multi_level_WP_or_shallow_feature_prediction'
    return {
        'stage': 'V9.2 Encoder Transition Diagnostic',
        'models_trained': False,
        'world_predictor_loaded': False,
        'decoder_modified': False,
        'segmentation_head_added': False,
        'gt_used': False,
        'full_train_split_used': args.case_limit == 0 and args.patch_limit == 0,
        'train_cases': train_names,
        'train_case_count': len(train_names),
        'action_protocol': {'gamma': 0.30, 'blur': 1.5},
        'patch_protocol': {
            'patches_per_case': args.patches_per_case,
            'foreground_patches_per_case': args.foreground_patches_per_case,
            'foreground_candidate_patches': args.foreground_candidate_patches,
            'foreground_threshold': args.foreground_threshold,
            'selection_uses_gt': False,
            'short_volume_fill': 'deterministic repeat_fill',
        },
        'encoder_layer_mapping': layer_metadata,
        'layers_sorted_shallow_to_deep': layers,
        'per_layer_action_summary': by_layer_action,
        'mean_relative_delta_by_layer': relative_by_action,
        'dominant_relative_delta_layer': dominant_layer,
        'deep_vs_shallow_mean_relative_delta': {
            action: {'deepest': deep_mean[action], 'shallow_mean': shallow_mean[action]}
            for action, _ in ACTION_PROTOCOL
        },
        'next_step_judgment': action_judgments,
        'interpretation_rule': {
            'A': 'deepest feature relative delta is at least the shallow-layer mean: prioritize WP transition learning/delta-residual supervision',
            'B': 'shallow-layer relative delta exceeds deepest mean: consider multi-level WP or shallow feature prediction',
        },
        'outputs': {
            'profile': str(output_dir / 'encoder_transition_profile.csv'),
            'summary': str(output_dir / 'encoder_transition_summary.json'),
            'progress': str(output_dir / 'progress.json'),
        },
    }


def run(args: argparse.Namespace) -> None:
    if ACTION_PROTOCOL != (('gamma', 0.30), ('blur', 1.5)):
        raise AssertionError('V9.2 action protocol must be gamma=0.30 and blur=1.5')
    if args.patches_per_case != 4 or args.foreground_patches_per_case != 2 or args.foreground_candidate_patches != 16:
        raise AssertionError('V9.2 patch protocol must match V9.1/V9.0: 4/2/16')
    if args.foreground_threshold != 0.5:
        raise AssertionError('V9.2 foreground threshold must remain 0.5')
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
        raise RuntimeError(f'V9.2 requires CUDA, resolved {device}')
    train_cases = iter_cases(paths, split='train')
    if len(train_cases) != 30:
        raise AssertionError(f'V9.2 requires the complete 30-case train split, got {len(train_cases)}')
    train_names = [case.case for case in train_cases]
    if args.case_limit:
        train_cases = train_cases[:args.case_limit]
        train_names = [case.case for case in train_cases]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_sink = CsvSink(output_dir / 'encoder_transition_profile.csv', PROFILE_FIELDS)
    progress_path = output_dir / 'progress.json'
    completed_cases: list[str] = []
    patch_rows = 0
    rows: list[dict[str, Any]] = []
    layer_metadata: dict[str, dict[str, Any]] = {}
    write_progress(progress_path, completed_cases, patch_rows, device, 'initializing')
    try:
        print('[V9.2] loading frozen VoxTell encoder only', flush=True)
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

        for case_index, case in enumerate(train_cases, start=1):
            print(f'[V9.2] case {case_index}/{len(train_cases)} start {case.case}', flush=True)
            image, _ = read_image(case)
            original_padded, slicers, patch_kinds = select_patch_slicers(
                interface, image, prompt_embedding,
                args.patches_per_case, args.foreground_patches_per_case,
                args.foreground_candidate_patches, args.foreground_threshold,
            )
            slicers, patch_kinds = pad_selected_patches(slicers, patch_kinds, args.patches_per_case)
            if args.patch_limit:
                slicers, patch_kinds = slicers[:args.patch_limit], patch_kinds[:args.patch_limit]
            action_padded = {
                action: padded_visual_action_and_slicers(interface.predictor, image, action, strength)[0]
                for action, strength in ACTION_PROTOCOL
            }
            for patch_index, slicer in enumerate(slicers):
                patch = torch.clone(original_padded[slicer][None], memory_format=torch.contiguous_format)
                source_features = encode_patch(interface, patch)
                if not layer_metadata:
                    names = feature_names(len(source_features))
                    for index, (name, feature) in enumerate(zip(names, source_features, strict=True)):
                        layer_metadata[name] = {
                            'encoder_index': index,
                            'feature_shape': 'x'.join(str(size) for size in feature.shape),
                            'channels': int(feature.shape[1]),
                            'spatial_resolution': 'x'.join(str(size) for size in feature.shape[-3:]),
                            'role': 'bottleneck/deepest' if index == len(names) - 1 else f'encoder stage {index + 1}',
                        }
                for action, _ in ACTION_PROTOCOL:
                    print(f'[V9.2] case {case_index}/{len(train_cases)} patch {patch_index + 1}/{len(slicers)} action={action}', flush=True)
                    action_patch = torch.clone(action_padded[action][slicer][None], memory_format=torch.contiguous_format)
                    transformed_features = encode_patch(interface, action_patch)
                    if len(transformed_features) != len(source_features):
                        raise AssertionError(f'Encoder feature count changed for {case.case} action={action}')
                    names = list(layer_metadata)
                    for name, source_feature, transformed_feature in zip(names, source_features, transformed_features, strict=True):
                        row = transition_row(
                            source_feature, transformed_feature, case.case, patch_index,
                            patch_kinds[patch_index], action, name,
                        )
                        profile_sink.write(row)
                        rows.append(row)
                    del action_patch, transformed_features
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                del patch, source_features
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                patch_rows += 1
            completed_cases.append(case.case)
            write_progress(progress_path, completed_cases, patch_rows, device, 'running')
            print(f'[V9.2] case {case_index}/{len(train_cases)} complete memory={memory_status(device)}', flush=True)
            del action_padded, original_padded, image
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        summary = build_summary(args, train_names, rows, layer_metadata, output_dir)
        (output_dir / 'encoder_transition_summary.json').write_text(json.dumps(summary, indent=2))
        write_progress(progress_path, completed_cases, patch_rows, device, 'complete')
        print('[V9.2] complete', flush=True)
    finally:
        profile_sink.close()


if __name__ == '__main__':
    run(parse_args())
