from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualConvBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class VisualWorldPredictor3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        action_dim: int = 2,
        num_blocks: int = 2,
        use_action: bool = True,
        text_delta_dim: int | None = None,
        use_language: bool = False,
        allow_unconditioned: bool = False,
    ) -> None:
        super().__init__()
        self.use_action = use_action
        self.use_language = use_language
        self.allow_unconditioned = allow_unconditioned
        self.input_projection = nn.Conv3d(in_channels, hidden_channels, 1)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_channels * 2),
            nn.GELU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )
        if text_delta_dim is not None:
            self.language_action_encoder = nn.Linear(text_delta_dim, hidden_channels * 2)
            nn.init.zeros_(self.language_action_encoder.weight)
            nn.init.zeros_(self.language_action_encoder.bias)
        self.blocks = nn.Sequential(*[ResidualConvBlock3D(hidden_channels) for _ in range(num_blocks)])
        self.output_projection = nn.Conv3d(hidden_channels, in_channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        text_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_projection(state)
        if text_delta is not None:
            if not hasattr(self, "language_action_encoder"):
                raise ValueError("text_delta_dim is required for language conditioning")
            scale_bias = self.language_action_encoder(text_delta).type_as(x)
        elif action is not None:
            if not self.use_action:
                raise ValueError("action conditioning is disabled")
            scale_bias = self.action_mlp(action).type_as(x)
        elif not self.allow_unconditioned and (self.use_action or self.use_language):
            raise ValueError("action or text_delta is required when conditioning is enabled")
        else:
            scale_bias = None
        if scale_bias is not None:
            scale, bias = scale_bias.chunk(2, dim=1)
            x = x * (1.0 + scale[:, :, None, None, None]) + bias[:, :, None, None, None]
        delta = self.output_projection(self.blocks(x))
        return state + delta


class LanguageWorldPredictor3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        text_delta_dim: int,
        hidden_channels: int = 64,
        action_dim: int = 3,
        num_blocks: int = 2,
        use_language: bool = True,
    ) -> None:
        super().__init__()
        self.use_language = use_language
        self.input_projection = nn.Conv3d(in_channels, hidden_channels, 1)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_channels * 2),
            nn.GELU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )
        self.language_action_encoder = nn.Linear(text_delta_dim, hidden_channels * 2)
        self.blocks = nn.Sequential(*[ResidualConvBlock3D(hidden_channels) for _ in range(num_blocks)])
        self.output_projection = nn.Conv3d(hidden_channels, in_channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, state: torch.Tensor, text_delta: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_projection(state)
        if self.use_language:
            if text_delta is None:
                raise ValueError("text_delta is required when use_language=True")
            scale_bias = self.language_action_encoder(text_delta).type_as(x)
            scale, bias = scale_bias.chunk(2, dim=1)
            x = x * (1.0 + scale[:, :, None, None, None]) + bias[:, :, None, None, None]
        delta = self.output_projection(self.blocks(x))
        return state + delta


def gamma_action(strength: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([[1.0, float(strength)]], dtype=torch.float32, device=device)


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target_scale = target.detach().float().pow(2).mean().clamp_min(eps)
    return F.mse_loss(prediction.float(), target.float()) / target_scale
