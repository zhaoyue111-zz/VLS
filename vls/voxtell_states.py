from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from einops import rearrange, repeat
from nnunetv2.utilities.helpers import dummy_context

from vls.config import ProjectPaths


def ensure_voxtell_on_path(voxtell_root: Path | str = ProjectPaths().voxtell_root) -> None:
    root = str(Path(voxtell_root))
    if root not in sys.path:
        sys.path.insert(0, root)


class VoxTellStateInterface:
    """Non-invasive wrapper that exposes VoxTell decoder states without editing VoxTell."""

    def __init__(self, predictor: Any):
        self.predictor = predictor
        self.network = predictor.network
        self.device = predictor.device

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path | str = ProjectPaths().voxtell_model_dir,
        device: torch.device | None = None,
        voxtell_root: Path | str = ProjectPaths().voxtell_root,
    ) -> "VoxTellStateInterface":
        ensure_voxtell_on_path(voxtell_root)
        from voxtell.inference.predictor_multiclass import VoxTellPredictor

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predictor = VoxTellPredictor(model_dir=str(model_dir), device=device)
        return cls(predictor)

    @torch.inference_mode()
    def embed_text_prompts(self, prompts: str | list[str]) -> torch.Tensor:
        return self.predictor.embed_text_prompts(prompts)

    @torch.inference_mode()
    def forward_with_states(
        self,
        image: torch.Tensor,
        prompt_or_embedding: str | list[str] | torch.Tensor,
    ) -> dict[str, Any]:
        if isinstance(prompt_or_embedding, torch.Tensor):
            text_embedding = prompt_or_embedding
        else:
            text_embedding = self.embed_text_prompts(prompt_or_embedding)

        network = self.network.to(self.device).eval()
        image = image.to(self.device)
        text_embedding = text_embedding.to(self.device)

        context = torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with context:
            return self._network_forward_with_states(network, image, text_embedding)

    def _network_forward_with_states(
        self,
        network: torch.nn.Module,
        img: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> dict[str, Any]:
        skips = network.encoder(img)
        selected_feature = skips[network.selected_decoder_layer]

        bottleneck_embed = rearrange(selected_feature, "b c d h w -> b h w d c")
        bottleneck_embed = network.project_bottleneck_embed(bottleneck_embed)
        bottleneck_embed = rearrange(bottleneck_embed, "b h w d c -> (h w d) b c")

        text_embedding = text_embedding.squeeze(2)
        text_embed = repeat(text_embedding, "b n dim -> n b dim")
        text_embed = network.project_text_embed(text_embed)

        mask_embedding, _ = network.transformer_decoder(
            tgt=text_embed,
            memory=bottleneck_embed,
            pos=network.pos_embed,
            memory_key_padding_mask=None,
        )
        mask_embedding = repeat(mask_embedding, "n b dim -> b n dim")

        mask_embeddings = [
            projection(mask_embedding)
            for projection in network.project_to_decoder_channels
        ]

        per_prompt_outputs = []
        per_prompt_states = []
        per_prompt_intermediate = []
        num_prompts = text_embedding.shape[1]
        for prompt_idx in range(num_prompts):
            prompt_embeds = [m[:, prompt_idx : prompt_idx + 1] for m in mask_embeddings]
            out = self._decoder_forward_with_states(network.decoder, skips, prompt_embeds)
            per_prompt_outputs.append(out["final_prediction"])
            per_prompt_states.append(out["decoder_states"])
            per_prompt_intermediate.append(out["intermediate_predictions"])

        return {
            "final_prediction": torch.cat(per_prompt_outputs, dim=1),
            "decoder_states": _stack_prompt_state_dicts(per_prompt_states),
            "intermediate_predictions": _cat_prompt_prediction_dicts(per_prompt_intermediate),
            "selected_encoder_feature": selected_feature,
            "text_embedding": text_embedding,
        }

    @staticmethod
    def _decoder_forward_with_states(
        decoder: torch.nn.Module,
        skips: list[torch.Tensor],
        mask_embeddings: list[torch.Tensor],
    ) -> dict[str, Any]:
        lres_input = skips[-1]
        seg_outputs: list[torch.Tensor] = []
        states: dict[str, torch.Tensor] = {}
        intermediate: dict[str, torch.Tensor] = {}
        mask_embeddings = mask_embeddings[::-1]

        for stage_idx in range(len(decoder.stages)):
            x = decoder.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = decoder.stages[stage_idx](x)

            stage_key = f"decoder_stage_{stage_idx}_low_to_high"
            if stage_idx == (len(decoder.stages) - 1):
                seg_pred = torch.einsum("b c h w d, b n c -> b n h w d", x, mask_embeddings[-1])
                seg_outputs.append(seg_pred)
                states[stage_key] = x
                intermediate[stage_key] = seg_pred
            elif stage_idx >= len(decoder.stages) - len(mask_embeddings):
                mask_embedding = mask_embeddings.pop(0)
                batch_size, _, _ = mask_embedding.shape
                mask_embedding_reshaped = mask_embedding.view(batch_size, decoder.num_heads, -1)
                fusion_features = torch.einsum(
                    "b c h w d, b n c -> b n h w d",
                    x,
                    mask_embedding_reshaped,
                )
                x = torch.cat((x, fusion_features), dim=1)
                seg_pred = decoder.seg_layers[stage_idx](x)
                seg_outputs.append(seg_pred)
                states[stage_key] = x
                intermediate[stage_key] = seg_pred
            else:
                states[stage_key] = x

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return {
            "final_prediction": seg_outputs[0],
            "decoder_states": states,
            "intermediate_predictions": intermediate,
        }


def _stack_prompt_state_dicts(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items], dim=1) for key in keys}


def _cat_prompt_prediction_dicts(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {key: torch.cat([item[key] for item in items], dim=1) for key in keys}
