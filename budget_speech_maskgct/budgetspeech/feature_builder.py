"""Lightweight construction of perceptual speech cue features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpeechCueFeatureBuilder(nn.Module):
    """Build frame-level features for the budget allocator.

    The module is intentionally small. It can be replaced by richer aligner or
    prosody encoders during full experiments.
    """

    def __init__(
        self,
        num_phone_symbols: int = 1024,
        phone_dim: int = 64,
        speaker_dim: int = 256,
        speaker_proj_dim: int = 16,
    ) -> None:
        super().__init__()
        self.phone_emb = nn.Embedding(num_phone_symbols, phone_dim, padding_idx=0)
        self.speaker_proj = nn.Linear(speaker_dim, speaker_proj_dim)
        self.output_dim = phone_dim + speaker_proj_dim + 8

    def forward(
        self,
        phone_ids: Tensor,
        duration: Tensor | None = None,
        f0: Tensor | None = None,
        energy: Tensor | None = None,
        voicing: Tensor | None = None,
        boundary: Tensor | None = None,
        speaker_embedding: Tensor | None = None,
    ) -> Tensor:
        """Return features with shape [B, T, D]."""
        if phone_ids.dim() != 2:
            raise ValueError("phone_ids must have shape [B, T]")
        bsz, frames = phone_ids.shape
        device = phone_ids.device

        phone_feat = self.phone_emb(phone_ids.long())
        scalar_feats = [
            self._norm_or_zero(duration, bsz, frames, device),
            self._norm_or_zero(f0, bsz, frames, device),
            self._norm_or_zero(energy, bsz, frames, device),
            self._bool_or_zero(voicing, bsz, frames, device),
            self._bool_or_zero(boundary, bsz, frames, device),
        ]

        if f0 is None:
            f0_delta = torch.zeros(bsz, frames, 1, device=device)
        else:
            f0_delta = torch.zeros_like(f0, dtype=torch.float32)
            f0_delta[:, 1:] = (f0[:, 1:] - f0[:, :-1]).abs()
            f0_delta = self._normalize(f0_delta).unsqueeze(-1)
        scalar_feats.append(f0_delta)

        if energy is None:
            energy_delta = torch.zeros(bsz, frames, 1, device=device)
        else:
            energy_delta = torch.zeros_like(energy, dtype=torch.float32)
            energy_delta[:, 1:] = (energy[:, 1:] - energy[:, :-1]).abs()
            energy_delta = self._normalize(energy_delta).unsqueeze(-1)
        scalar_feats.append(energy_delta)

        silence = (phone_ids == 0).to(torch.float32).unsqueeze(-1)
        scalar_feats.append(silence)

        if speaker_embedding is None:
            speaker_feat = torch.zeros(
                bsz, frames, self.speaker_proj.out_features, device=device
            )
        else:
            speaker_feat = self.speaker_proj(speaker_embedding.float())
            speaker_feat = speaker_feat[:, None, :].expand(-1, frames, -1)

        return torch.cat([phone_feat, speaker_feat, *scalar_feats], dim=-1)

    @staticmethod
    def _normalize(x: Tensor) -> Tensor:
        x = x.float()
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        return (x - mean) / std

    def _norm_or_zero(self, x: Tensor | None, bsz: int, frames: int, device) -> Tensor:
        if x is None:
            return torch.zeros(bsz, frames, 1, device=device)
        return self._normalize(x).unsqueeze(-1)

    @staticmethod
    def _bool_or_zero(x: Tensor | None, bsz: int, frames: int, device) -> Tensor:
        if x is None:
            return torch.zeros(bsz, frames, 1, device=device)
        return x.float().unsqueeze(-1)


def heuristic_features_from_semantic(
    semantic_code: Tensor,
    feature_dim: int = 96,
) -> Tensor:
    """Create deterministic placeholder features from semantic tokens.

    This is useful for wiring inference before a trained cue encoder is ready.
    It should be replaced by real phone/prosody/speaker features in experiments.
    """
    if semantic_code.dim() != 2:
        raise ValueError("semantic_code must have shape [B, T]")

    code = semantic_code.float()
    norm = (code - code.mean(dim=1, keepdim=True)) / code.std(
        dim=1, keepdim=True
    ).clamp_min(1e-5)
    delta = torch.zeros_like(norm)
    delta[:, 1:] = (norm[:, 1:] - norm[:, :-1]).abs()
    base = torch.stack([norm, delta], dim=-1)

    repeats = (feature_dim + base.shape[-1] - 1) // base.shape[-1]
    return base.repeat(1, 1, repeats)[..., :feature_dim]
