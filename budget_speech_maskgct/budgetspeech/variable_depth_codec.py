"""Variable-depth RVQ utilities for MaskGCT acoustic codec tokens."""

from __future__ import annotations

import torch
from torch import Tensor


def nested_depth_to_mask(token_depth: Tensor, num_quantizers: int) -> Tensor:
    """Convert integer depth [B, T] to nested RVQ mask [B, T, J]."""
    if token_depth.dim() != 2:
        raise ValueError("token_depth must have shape [B, T]")
    device = token_depth.device
    idx = torch.arange(num_quantizers, device=device).view(1, 1, -1)
    return (idx < token_depth.unsqueeze(-1)).to(torch.float32)


def mask_codec_tokens(
    codec_tokens: Tensor,
    token_depth: Tensor,
    null_token_id: int = 0,
) -> Tensor:
    """Replace inactive quantizer tokens with a null token.

    Args:
        codec_tokens: [B, T, J].
        token_depth: [B, T], active nested depth per frame.
    """
    if codec_tokens.dim() != 3:
        raise ValueError("codec_tokens must have shape [B, T, J]")
    mask = nested_depth_to_mask(token_depth, codec_tokens.shape[-1]).bool()
    nulls = torch.full_like(codec_tokens, fill_value=null_token_id)
    return torch.where(mask, codec_tokens, nulls)


def variable_depth_vq2emb(
    codec_decoder,
    codec_tokens: Tensor,
    token_depth: Tensor,
    token_layout: str = "BTJ",
) -> Tensor:
    """Convert codec tokens to quantized embeddings with per-frame depth.

    MaskGCT's codec decoder supports a global `n_quantizers`; this function
    adds a per-frame depth by summing each quantizer embedding only where the
    predicted nested depth activates that quantizer.

    Args:
        codec_decoder: MaskGCT CodecDecoder instance.
        codec_tokens: [B, T, J] if `token_layout="BTJ"`, or [J, B, T].
        token_depth: [B, T].
        token_layout: Layout of `codec_tokens`.

    Returns:
        Quantized embedding [B, D, T], ready for `codec_decoder(...)`.
    """
    if token_layout == "BTJ":
        tokens_qbt = codec_tokens.permute(2, 0, 1)
    elif token_layout == "JBT":
        tokens_qbt = codec_tokens
    else:
        raise ValueError("token_layout must be 'BTJ' or 'JBT'")

    if token_depth.dim() != 2:
        raise ValueError("token_depth must have shape [B, T]")

    num_quantizers = tokens_qbt.shape[0]
    token_depth = token_depth.clamp(0, num_quantizers)
    quantized_out = None

    for idx, quantizer in enumerate(codec_decoder.quantizer.quantizers):
        if idx >= num_quantizers:
            break
        emb = quantizer.vq2emb(tokens_qbt[idx])
        active = (token_depth > idx).to(emb.dtype).unsqueeze(1)
        emb = emb * active
        quantized_out = emb if quantized_out is None else quantized_out + emb

    if quantized_out is None:
        raise ValueError("No quantizer embeddings were produced")
    return quantized_out


def decode_variable_depth(
    codec_decoder,
    codec_tokens: Tensor,
    token_depth: Tensor,
    token_layout: str = "BTJ",
) -> Tensor:
    """Decode per-frame variable-depth codec tokens to waveform."""
    vq_emb = variable_depth_vq2emb(
        codec_decoder=codec_decoder,
        codec_tokens=codec_tokens,
        token_depth=token_depth,
        token_layout=token_layout,
    )
    return codec_decoder(vq_emb)


def average_token_depth(token_depth: Tensor, frame_mask: Tensor | None = None) -> Tensor:
    """Compute masked average token depth."""
    if frame_mask is None:
        return token_depth.float().mean()
    denom = frame_mask.float().sum().clamp_min(1.0)
    return (token_depth.float() * frame_mask.float()).sum() / denom
