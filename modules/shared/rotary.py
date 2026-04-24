from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Supports partial rotary: when cos.shape[-1] < q.shape[-1] only the leading
    `rotary_dim` channels are rotated, the tail passes through unchanged.
    Degenerates to full-rotary qwen3 behavior when cos.shape[-1] == q.shape[-1].
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    if q_pass.shape[-1] > 0:
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Parameterized rotary embedding.

    Supports:
      - Default RoPE (qwen3-style, 2D position_ids)
      - Partial rotary via ``rope_parameters["partial_rotary_factor"]``
      - Interleaved mRoPE via ``rope_parameters["mrope_interleaved"]=True`` and
        ``rope_parameters["mrope_section"]`` (qwen3.5-style).

    The caller passes ``position_ids`` as either a 2D ``[B, T]`` tensor or a
    3D ``[3, B, T]`` tensor (T, H, W for mrope). When mrope is enabled the
    input is auto-expanded to 3D if 2D is provided.
    """

    inv_freq: torch.Tensor

    def __init__(self, config, device=None) -> None:
        super().__init__()
        self.config = config
        rope_params = config.rope_parameters or {}
        self.rope_type = rope_params.get("rope_type", "default")
        self.mrope_interleaved = bool(rope_params.get("mrope_interleaved", False))
        self.mrope_section = rope_params.get("mrope_section", None)
        self.partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
        self.attention_scaling = 1.0

        base = rope_params.get("rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * self.partial_rotary_factor)
        # Force even
        dim = dim - (dim % 2)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.max_seq_len_cached = getattr(config, "max_position_embeddings", 0)
        self.original_max_seq_len = self.max_seq_len_cached

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mrope_interleaved:
            return self._forward_mrope(x, position_ids)
        return self._forward_default(x, position_ids)

    def _forward_default(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, T]
        if position_ids.ndim != 2:
            raise ValueError(f"Default RoPE expects 2D position_ids, got shape {tuple(position_ids.shape)}")
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _forward_mrope(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Auto-expand 2D → 3D for mrope.
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.shape[0] != 3:
            raise ValueError(
                f"mRoPE expects position_ids leading dim 3 (T,H,W), got {tuple(position_ids.shape)}"
            )
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self._apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section) -> torch.Tensor:
        """Interleaved mRoPE (ported from Qwen3_5MoeTextRotaryEmbedding).

        freqs: [3, bs, seq_len, head_dim//2]
        Returns: [bs, seq_len, head_dim//2] with T/H/W frequencies interleaved.
        """
        if mrope_section is None:
            # No section given; just take T and return 2D form
            return freqs[0]
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
