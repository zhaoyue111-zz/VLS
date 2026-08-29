"""V9.3 multi-level residual World Predictor training.

The encoder and native VoxTell decoder remain frozen and unchanged.  A bank
of the existing VisualWorldPredictor3D residual blocks predicts residuals for
encoder stage 2 through the deepest feature.  Stage 1 is intentionally not
predicted; it remains the only source-side auxiliary skip required by the
native decoder at the highest resolution.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vls.config import DEFAULT_PROMPTS, ProjectPaths
from vls.data import iter_cases, read_image
from vls.v2_experiment import (
    padded_image_and_slicers,
    padded_visual_action_and_slicers,
    resolve_device,
    select_patch_slicers,
    visual_action,
)
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT
from vls.v3_language_experiment import flatten_prompt_embedding
from vls.v7_0d_protocol_sanity import set_seed
from vls.voxtell_states import VoxTellStateInterface
from vls.world_model import VisualWorldPredictor3D, normalized_mse
from vls.v9_0_world_state_selection_audit import pad_selected_patches
from vls.v9_2_encoder_transition_diagnostic import encode_patch


GAMMA_STRENGTH = 0.30
BLUR_SIGMA = 1.5
ACTION_PROTOCOL = (('gamma', GAMMA_STRENGTH), ('blur', BLUR_SIGMA))
LEVEL_NAMES = (
    'encoder_stage_2',
    'encoder_stage_3',
    'encoder_stage_4',
    'encoder_stage_5',
    'encoder_bottleneck_deepest',
)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths()
    parser = argparse.ArgumentParser(description='V9.3 hierarchical residual World Predictor')
    parser.add_argument('--model-dir', default=str(paths.voxtell_model_dir))
    parser.add_argument('--voxtell-root', default=str(paths.voxtell_root))
    parser.add_argument('--data-root', default=str(paths.data_root))
    parser.add_argument('--split-json', default=str(paths.split_json))
    parser.add_argument('--output-dir', default='outputs/v9_3_hierarchical_residual_world_predictor')
    parser.add_argument('--patches-per-case', type=int, default=4)
    parser.add_argument('--foreground-patches-per-case', type=int, default=2)
    parser.add_argument('--foreground-candidate-patches', type=int, default=16)
    parser.add_argument('--foreground-threshold', type=float, default=0.5)
    parser.add_argument('--hidden-channels', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case-limit', type=int, default=0, help='debug limit; 0 uses all 30 train cases')
    parser.add_argument('--patch-limit', type=int, default=0, help='debug limit; 0 uses all selected patches')
    parser.add_argument('--sanity-only', action='store_true', help='run synthetic forward/loss/shape sanity only')
    return parser.parse_args()


def layer_name_for_encoder_index(index: int, feature_count: int) -> str:
    if index == feature_count - 1:
        return 'encoder_bottleneck_deepest'
    return f'encoder_stage_{index + 1}'


def selected_level_names(feature_count: int) -> tuple[str, ...]:
    if feature_count < 6:
        raise AssertionError(f'V9.3 requires six encoder outputs, got {feature_count}')
    names = tuple(layer_name_for_encoder_index(index, feature_count) for index in range(1, feature_count))
    if names != LEVEL_NAMES:
        raise AssertionError(f'Unexpected encoder hierarchy: {names}; expected {LEVEL_NAMES}')
    return names


class HierarchicalResidualWorldPredictor(nn.Module):
    """Existing VisualWorldPredictor3D residual architecture at each level."""

    def __init__(
        self,
        level_channels: dict[str, int],
        hidden_channels: int,
        text_delta_dim: int,
    ) -> None:
        super().__init__()
        self.level_names = tuple(level_channels)
        self.level_channels = dict(level_channels)
        self.predictors = nn.ModuleDict({
            name: VisualWorldPredictor3D(
                in_channels=channels,
                hidden_channels=hidden_channels,
                action_dim=3,
                num_blocks=2,
                use_action=True,
                text_delta_dim=text_delta_dim,
                use_language=True,
                allow_unconditioned=True,
            )
            for name, channels in level_channels.items()
        })

    @staticmethod
    def _forward_with_action_and_language(
        predictor: VisualWorldPredictor3D,
        state: torch.Tensor,
        action: torch.Tensor,
        text_delta: torch.Tensor,
    ) -> torch.Tensor:
        x = predictor.input_projection(state.float())
        action_bias = predictor.action_mlp(action.float()).type_as(x)
        language_bias = predictor.language_action_encoder(text_delta.float()).type_as(x)
        scale, bias = (action_bias + language_bias).chunk(2, dim=1)
        x = x * (1.0 + scale[:, :, None, None, None]) + bias[:, :, None, None, None]
        residual = predictor.output_projection(predictor.blocks(x))
        predicted = state.float() + residual
        if tuple(predicted.shape) != tuple(state.shape):
            raise AssertionError(f'Residual predictor changed feature shape: {state.shape} -> {predicted.shape}')
        return predicted

    def forward(
        self,
        features: dict[str, torch.Tensor],
        action: torch.Tensor,
        text_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tuple(features) != self.level_names:
            raise AssertionError(f'Feature level order mismatch: {tuple(features)} vs {self.level_names}')
        predictions = {}
        for name in self.level_names:
            feature = features[name]
            prediction = self._forward_with_action_and_language(
                self.predictors[name], feature, action, text_delta,
            )
            if tuple(prediction.shape) != tuple(feature.shape):
                raise AssertionError(f'{name} residual shape mismatch')
            predictions[name] = prediction
        return predictions


def hierarchical_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    if tuple(predictions) != tuple(targets):
        raise AssertionError('Prediction and target level sets differ')
    losses = []
    for name in predictions:
        if tuple(predictions[name].shape) != tuple(targets[name].shape):
            raise AssertionError(
                f'{name} prediction/target shape mismatch: '
                f'{tuple(predictions[name].shape)} vs {tuple(targets[name].shape)}'
            )
        losses.append(normalized_mse(predictions[name], targets[name]))
    return torch.stack(losses).mean()


def level_features(features: list[torch.Tensor], names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    if len(features) != len(names) + 1:
        raise AssertionError(f'Expected {len(names) + 1} encoder outputs, got {len(features)}')
    result = {
        name: features[index]
        for index, name in enumerate(names, start=1)
    }
    return result


def assert_decoder_feature_contract(
    source_skips: list[torch.Tensor],
    predicted_features: dict[str, torch.Tensor],
    level_names: tuple[str, ...],
) -> list[torch.Tensor]:
    if len(source_skips) != len(level_names) + 1:
        raise AssertionError('Source skip count does not match hierarchical level contract')
    predicted_skips = list(source_skips)
    for index, name in enumerate(level_names, start=1):
        if tuple(predicted_features[name].shape) != tuple(source_skips[index].shape):
            raise AssertionError(f'Decoder skip shape mismatch at {name}')
        predicted_skips[index] = predicted_features[name]
    for index, (source, predicted) in enumerate(zip(source_skips, predicted_skips, strict=True)):
        if tuple(source.shape) != tuple(predicted.shape):
            raise AssertionError(f'Decoder received invalid skip shape at encoder index {index}')
    if predicted_skips[0] is not source_skips[0]:
        raise AssertionError('Stage 1 must remain the only source-side auxiliary skip')
    return predicted_skips


@torch.inference_mode()
def native_decoder_from_predicted_skips(
    interface: VoxTellStateInterface,
    source_context: dict[str, Any],
    predicted_features: dict[str, torch.Tensor],
    level_names: tuple[str, ...],
) -> torch.Tensor:
    audit = source_context['decoder_audit']
    source_skips = audit['skips']
    predicted_skips = assert_decoder_feature_contract(source_skips, predicted_features, level_names)
    decoder = audit['decoder']
    mask_embeddings = audit['mask_embeddings']
    autocast_context = (
        torch.autocast(interface.device.type, enabled=True)
        if interface.device.type == 'cuda' else torch.autocast('cpu', enabled=False)
    )
    with autocast_context:
        output = decoder(predicted_skips, mask_embeddings)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return output


def serialize_slicer(slicer: tuple) -> list[list[int | None]]:
    return [[item.start, item.stop, item.step] for item in slicer]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    interface: VoxTellStateInterface,
    cases: list[Any],
    prompt_embedding: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        image, _ = read_image(case)
        _, slicers, patch_kinds = select_patch_slicers(
            interface, image, prompt_embedding,
            args.patches_per_case, args.foreground_patches_per_case,
            args.foreground_candidate_patches, args.foreground_threshold,
        )
        slicers, patch_kinds = pad_selected_patches(slicers, patch_kinds, args.patches_per_case)
        if args.patch_limit:
            slicers, patch_kinds = slicers[:args.patch_limit], patch_kinds[:args.patch_limit]
        for patch_index, (slicer, patch_kind) in enumerate(zip(slicers, patch_kinds, strict=True)):
            records.append({
                'case': case.case,
                'patch_index': patch_index,
                'patch_kind': patch_kind,
                'slicer': serialize_slicer(slicer),
            })
    return records


def deserialize_slicer(value: list[list[int | None]]) -> tuple[slice, ...]:
    return tuple(slice(start, stop, step) for start, stop, step in value)


def grouped_manifest(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record['case']].append(record)
    return grouped


def train_epoch(
    interface: VoxTellStateInterface,
    model: HierarchicalResidualWorldPredictor,
    optimizer: torch.optim.Optimizer,
    cases_by_name: dict[str, Any],
    records: list[dict[str, Any]],
    level_names: tuple[str, ...],
    text_delta: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    grouped = grouped_manifest(records)
    losses = []
    for case_name, case_records in grouped.items():
        image, _ = read_image(cases_by_name[case_name])
        source_padded, _ = padded_image_and_slicers(interface.predictor, image)
        action_padded = {
            action: padded_visual_action_and_slicers(interface.predictor, image, action, strength)[0]
            for action, strength in ACTION_PROTOCOL
        }
        for record in case_records:
            slicer = deserialize_slicer(record['slicer'])
            source_patch = torch.clone(source_padded[slicer][None], memory_format=torch.contiguous_format)
            source_features_all = encode_patch(interface, source_patch)
            # ``encode_patch`` runs the frozen encoder under inference_mode.
            # Clone the selected features into ordinary tensors before the
            # trainable WP sees them, so autograd can safely save inputs for
            # WP parameter gradients without retaining an encoder graph.
            source_features = {
                name: source_features_all[index].detach().clone()
                for index, name in enumerate(level_names, start=1)
            }
            for action, strength in ACTION_PROTOCOL:
                action_patch = torch.clone(action_padded[action][slicer][None], memory_format=torch.contiguous_format)
                target_features_all = encode_patch(interface, action_patch)
                targets = {
                    name: target_features_all[index]
                    for index, name in enumerate(level_names, start=1)
                }
                optimizer.zero_grad(set_to_none=True)
                predictions = model(source_features, visual_action(action, strength, device), text_delta)
                loss = hierarchical_loss(predictions, targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                del action_patch, target_features_all, targets, predictions, loss
            del source_patch, source_features_all, source_features
        del action_padded, source_padded, image
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return float(np.mean(losses)) if losses else float('nan')


def transition_fidelity_rows(
    source: dict[str, torch.Tensor],
    real: dict[str, torch.Tensor],
    predicted: dict[str, torch.Tensor],
    case: str,
    patch_index: int,
    patch_kind: str,
    action: str,
) -> list[dict[str, Any]]:
    rows = []
    for name in source:
        source_feature = source[name].float()
        real_feature = real[name].float()
        predicted_feature = predicted[name].float()
        true_delta = real_feature - source_feature
        predicted_delta = predicted_feature - source_feature
        true_norm = torch.linalg.vector_norm(true_delta.reshape(-1))
        predicted_norm = torch.linalg.vector_norm(predicted_delta.reshape(-1))
        source_norm = torch.linalg.vector_norm(source_feature.reshape(-1))
        denom = torch.linalg.vector_norm(predicted_delta.reshape(-1)) * torch.linalg.vector_norm(true_delta.reshape(-1))
        cosine = None if float(denom.detach().cpu()) <= 1e-12 else float((torch.sum(predicted_delta * true_delta) / denom).detach().cpu())
        source_norm_value = float(source_norm.detach().cpu())
        true_norm_value = float(true_norm.detach().cpu())
        predicted_norm_value = float(predicted_norm.detach().cpu())
        raw_mse = float(torch.mean((predicted_feature - real_feature).square()).detach().cpu())
        rows.append({
            # Preserve the V9.1/V9.2-compatible names as aliases alongside
            # the explicit V9.3 predicted-vs-real transition fields.
            'case_id': case,
            'case': case,
            'patch_index': patch_index,
            'patch_kind': patch_kind,
            'action': action,
            'layer': name,
            'feature_shape': 'x'.join(str(size) for size in source_feature.shape),
            'mse': raw_mse,
            'normalized_mse': float(normalized_mse(predicted_feature, real_feature).detach().cpu()),
            'delta_norm': predicted_norm_value,
            'relative_delta': None if source_norm_value <= 1e-12 else predicted_norm_value / source_norm_value,
            'predicted_delta_norm': predicted_norm_value,
            'real_delta_norm': true_norm_value,
            'magnitude_ratio': None if true_norm_value <= 1e-12 else predicted_norm_value / true_norm_value,
            'cosine_similarity': cosine,
        })
    return rows


@torch.inference_mode()
def evaluate_records(
    interface: VoxTellStateInterface,
    model: HierarchicalResidualWorldPredictor,
    cases_by_name: dict[str, Any],
    records: list[dict[str, Any]],
    level_names: tuple[str, ...],
    text_delta: torch.Tensor,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    grouped = grouped_manifest(records)
    losses = []
    fidelity_rows = []
    decoder_rows = []
    for case_name, case_records in grouped.items():
        image, _ = read_image(cases_by_name[case_name])
        source_padded, _ = padded_image_and_slicers(interface.predictor, image)
        action_padded = {
            action: padded_visual_action_and_slicers(interface.predictor, image, action, strength)[0]
            for action, strength in ACTION_PROTOCOL
        }
        for record in case_records:
            patch_index = int(record['patch_index'])
            slicer = deserialize_slicer(record['slicer'])
            patch_kind = record['patch_kind']
            source_patch = torch.clone(source_padded[slicer][None], memory_format=torch.contiguous_format)
            source_context = interface.forward_with_audit_context(source_patch, [SOURCE_PROMPT])
            source_skips = source_context['decoder_audit']['skips']
            source_features = level_features(source_skips, level_names)
            for action, strength in ACTION_PROTOCOL:
                action_patch = torch.clone(action_padded[action][slicer][None], memory_format=torch.contiguous_format)
                action_context = interface.forward_with_audit_context(action_patch, [SOURCE_PROMPT])
                target_features = level_features(action_context['decoder_audit']['skips'], level_names)
                action_tensor = visual_action(action, strength, device)
                predicted = model(source_features, action_tensor, text_delta)
                loss = hierarchical_loss(predicted, target_features)
                losses.append(float(loss.detach().cpu()))
                fidelity_rows.extend(transition_fidelity_rows(
                    source_features, target_features, predicted,
                    case_name, patch_index, patch_kind, action,
                ))
                predicted_skips = assert_decoder_feature_contract(source_skips, predicted, level_names)
                predicted_logits = native_decoder_from_predicted_skips(
                    interface, source_context, predicted, level_names,
                )
                real_logits = action_context['final_prediction'][:, :1]
                predicted_probability = torch.sigmoid(predicted_logits)
                real_probability = torch.sigmoid(real_logits)
                predicted_mask = predicted_probability > 0.5
                real_mask = real_probability > 0.5
                intersection = (predicted_mask & real_mask).sum().float()
                union = (predicted_mask | real_mask).sum().float()
                dice_den = predicted_mask.sum().float() + real_mask.sum().float()
                decoder_rows.append({
                    'case': case_name,
                    'patch_index': patch_index,
                    'patch_kind': patch_kind,
                    'action': action,
                    'probability_mse': float(torch.mean((predicted_probability - real_probability).square()).detach().cpu()),
                    'probability_mae': float(torch.mean(torch.abs(predicted_probability - real_probability)).detach().cpu()),
                    'logits_mse': float(torch.mean((predicted_logits.float() - real_logits.float()).square()).detach().cpu()),
                    'mask_dice_vs_real': float(((2 * intersection) / dice_den.clamp_min(1.0)).detach().cpu()),
                    'mask_iou_vs_real': float((intersection / union.clamp_min(1.0)).detach().cpu()),
                    'predicted_skip_count': len(predicted_skips),
                })
                del action_patch, action_context, target_features, predicted, loss, predicted_logits, real_logits
            del source_patch, source_context, source_features
        del action_padded, source_padded, image
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return float(np.mean(losses)), fidelity_rows, decoder_rows


def tiny_sanity_check(device: torch.device) -> None:
    torch.manual_seed(2026)
    names = LEVEL_NAMES
    channels = {name: 4 + index for index, name in enumerate(names)}
    # Keep at least two spatial elements at the deepest level because the
    # existing residual block contains InstanceNorm3d in train mode.
    shapes = [16, 8, 4, 2, 2]
    model = HierarchicalResidualWorldPredictor(channels, hidden_channels=4, text_delta_dim=3).to(device)
    source = {
        name: torch.randn(1, channels[name], size, size, size, device=device)
        for name, size in zip(names, shapes, strict=True)
    }
    target = {name: torch.randn_like(value) for name, value in source.items()}
    action = torch.randn(1, 3, device=device)
    text_delta = torch.randn(1, 3, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    prediction = model(source, action, text_delta)
    loss = hierarchical_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert all(tuple(prediction[name].shape) == tuple(source[name].shape) for name in names)
    source_skips = [torch.randn(1, 2, 32, 32, 32, device=device)] + [source[name] for name in names]
    predicted_skips = assert_decoder_feature_contract(source_skips, prediction, names)
    assert len(predicted_skips) == 6
    print(json.dumps({'sanity': 'passed', 'loss': float(loss.detach()), 'levels': list(names)}))


def run(args: argparse.Namespace) -> None:
    if args.sanity_only:
        if args.device == 'cuda' and torch.cuda.is_available():
            sanity_device = torch.device(f'cuda:{args.gpu}')
        else:
            sanity_device = torch.device('cpu')
        tiny_sanity_check(sanity_device)
        return
    if ACTION_PROTOCOL != (('gamma', 0.30), ('blur', 1.5)):
        raise AssertionError('V9.3 action protocol must remain gamma=0.30 and blur=1.5')
    if args.patches_per_case != 4 or args.foreground_patches_per_case != 2 or args.foreground_candidate_patches != 16:
        raise AssertionError('V9.3 patch protocol must match V9.0/V9.1: 4/2/16')
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
        raise RuntimeError(f'V9.3 requires CUDA, resolved {device}')
    train_cases = iter_cases(paths, split='train')
    test_cases = iter_cases(paths, split='test')
    if len(train_cases) != 30:
        raise AssertionError(f'V9.3 requires the complete 30-case train split, got {len(train_cases)}')
    train_names = [case.case for case in train_cases]
    if set(train_names) & {case.case for case in test_cases}:
        raise AssertionError('V9.3 train/test overlap')
    if args.case_limit:
        train_cases = train_cases[:args.case_limit]
        train_names = [case.case for case in train_cases]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interface = VoxTellStateInterface.from_model_dir(
        paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root,
    )
    prompt_embedding = interface.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    text_embedding = interface.embed_text_prompts(DEFAULT_PROMPTS + ['the liver'])
    text_delta = (flatten_prompt_embedding(text_embedding, 1) - flatten_prompt_embedding(text_embedding, 0)).to(device)[None]
    train_manifest = build_manifest(interface, train_cases, prompt_embedding, args)
    test_manifest = build_manifest(interface, test_cases, prompt_embedding, args)
    (output_dir / 'train_patch_manifest.json').write_text(json.dumps(train_manifest, indent=2))
    (output_dir / 'test_patch_manifest.json').write_text(json.dumps(test_manifest, indent=2))
    cases_by_name = {case.case: case for case in [*train_cases, *test_cases]}
    with torch.inference_mode():
        sample_image, _ = read_image(train_cases[0])
        sample_padded, _ = padded_image_and_slicers(interface.predictor, sample_image)
        sample_slicer = deserialize_slicer(train_manifest[0]['slicer'])
        sample_patch = torch.clone(sample_padded[sample_slicer][None])
        sample_features = encode_patch(interface, sample_patch)
    level_names = selected_level_names(len(sample_features))
    level_channels = {name: int(sample_features[index].shape[1]) for index, name in enumerate(level_names, start=1)}
    model = HierarchicalResidualWorldPredictor(level_channels, args.hidden_channels, int(text_delta.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    training_rows = []
    validation_rows = []
    best_loss = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(interface, model, optimizer, cases_by_name, train_manifest, level_names, text_delta, device)
        validation_loss, fidelity_rows, decoder_rows = evaluate_records(
            interface, model, cases_by_name, test_manifest, level_names, text_delta, device,
        )
        training_rows.append({'epoch': epoch, 'split': 'train', 'mean_normalized_mse': train_loss, 'train_case_count': len(train_cases)})
        validation_rows.append({'epoch': epoch, 'split': 'test', 'mean_normalized_mse': validation_loss, 'test_labels_used': False})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        write_csv(output_dir / 'training_curve.csv', training_rows)
        write_csv(output_dir / 'validation_curve.csv', validation_rows)
        write_csv(output_dir / 'encoder_transition_fidelity.csv', fidelity_rows)
        write_csv(output_dir / 'decoder_transition_fidelity.csv', decoder_rows)
    if best_state is None:
        raise AssertionError('No hierarchical checkpoint was selected')
    checkpoint = {
        'stage': 'V9.3 hierarchical residual World Predictor',
        'selected_epoch': int(np.argmin([row['mean_normalized_mse'] for row in validation_rows]) + 1),
        'selected_encoder_levels': list(level_names),
        'encoder_level_indices': {name: index for index, name in enumerate(level_names, start=1)},
        'level_channels': level_channels,
        'hidden_channels': args.hidden_channels,
        'text_delta_dim': int(text_delta.shape[-1]),
        'action_protocol': {'gamma': GAMMA_STRENGTH, 'blur': BLUR_SIGMA},
        'architecture': 'existing VisualWorldPredictor3D residual architecture replicated per encoder level; action and language conditioning summed before scale/bias injection',
        'decoder_contract': 'predicted encoder stage2-stage5 and bottleneck skips; source stage1 auxiliary skip only',
        'state_dict': best_state,
        'train_case_count': len(train_cases),
        'test_case_count': len(test_cases),
    }
    torch.save(checkpoint, output_dir / 'best_hierarchical_world_predictor.pt')
    summary = {
        'stage': 'V9.3 hierarchical residual World Predictor',
        'models_trained': True,
        'encoder_modified': False,
        'decoder_structure_modified': False,
        'segmentation_head_modified': False,
        'loss': 'mean over per-level normalized_mse(predicted_feature, real_feature)',
        'optimizer': {'name': 'AdamW', 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay},
        'train_cases': train_names,
        'train_case_count': len(train_names),
        'test_case_count': len(test_cases),
        'full_train_split_used': args.case_limit == 0 and args.patch_limit == 0,
        'selected_encoder_levels': list(level_names),
        'level_channels': level_channels,
        'stage1_prediction': False,
        'decoder_input_contract': 'stage1 source skip auxiliary; stage2-stage5 and bottleneck/deepest replaced by predicted residual features',
        'action_protocol': {'gamma': GAMMA_STRENGTH, 'blur': BLUR_SIGMA},
        'checkpoint': str(output_dir / 'best_hierarchical_world_predictor.pt'),
        'training_curve': str(output_dir / 'training_curve.csv'),
        'validation_curve': str(output_dir / 'validation_curve.csv'),
        'encoder_transition_fidelity': str(output_dir / 'encoder_transition_fidelity.csv'),
        'decoder_transition_fidelity': str(output_dir / 'decoder_transition_fidelity.csv'),
        'manifests': {'train': str(output_dir / 'train_patch_manifest.json'), 'test': str(output_dir / 'test_patch_manifest.json')},
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'checkpoint': str(output_dir / 'best_hierarchical_world_predictor.pt'), 'best_validation_loss': best_loss}, indent=2))


if __name__ == '__main__':
    run(parse_args())
