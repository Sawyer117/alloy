from __future__ import annotations

from typing import Callable

import torch
from torch import nn

# ROPE_INIT_FUNCTIONS maps "linear" / "dynamic" / "yarn" / "longrope" / "llama3"
# to their inv_freq-computing functions, and dynamic_rope_update is the
# context-aware rescaling wrapper HF uses on every rotary.forward. Both live
# in transformers' modeling_rope_utils — a framework-level utility module,
# not under transformers.models.*, so we import rather than copy.
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update


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
      - All rope_types registered in ``ROPE_INIT_FUNCTIONS``: linear, dynamic,
        yarn, longrope, llama3 — dispatched by ``rope_parameters["rope_type"]``
      - Partial rotary via ``rope_parameters["partial_rotary_factor"]``
      - Interleaved mRoPE via ``rope_parameters["mrope_interleaved"]=True`` and
        ``rope_parameters["mrope_section"]`` (qwen3.5-style)

    The caller passes ``position_ids`` as either a 2D ``[B, T]`` tensor or a
    3D ``[3, B, T]`` tensor (T, H, W for mrope). When mrope is enabled the
    input is auto-expanded to 3D if 2D is provided.

    Class structure mirrors per-model HF rotary classes (``Qwen3RotaryEmbedding``
    etc.): the ``rope_type == "default"`` path is computed locally; everything
    else delegates to ``ROPE_INIT_FUNCTIONS[rope_type]``.
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

        self.max_seq_len_cached = getattr(config, "max_position_embeddings", 0)
        self.original_max_seq_len = self.max_seq_len_cached

        rope_init_fn: Callable = self._compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def _compute_default_rope_parameters(config, device=None, seq_len=None):
        """Default RoPE (qwen3 / qwen3.5 style), supporting partial rotary.

        Signature matches ROPE_INIT_FUNCTIONS entries so we can treat it
        uniformly in ``__init__``. ``seq_len`` is unused for this type.
        """
        del seq_len
        rope_params = config.rope_parameters or {}
        base = rope_params.get("rope_theta", 10000.0)
        partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        dim = dim - (dim % 2)  # force even
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update  # runtime inv_freq rescaling for dynamic / yarn / etc.
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


# --------------------------------------------------------------------------- #
# Source-coupled aliases
# --------------------------------------------------------------------------- #
# The :class:`RotaryEmbedding` implementation above is genuinely parametric
# — its body covers qwen3 (full rotary) and qwen3.5 / qwen3-next (partial +
# interleaved mRoPE) under the same dispatch. The aliases below exist so
# call sites can name the source family for self-documentation, mirroring
# HF's per-model class convention. Future model families with structurally
# different rotary (e.g. DeepSeek-V4's two-set ``rope_parameters`` for main
# vs compressor RoPE) get their own non-aliased class.
class Qwen3RotaryEmbedding(RotaryEmbedding):
    """Qwen3 family rotary: full rotary, no partial / mrope.

    Pure documentation subclass — body is the parent's parametric
    implementation. Use this name at call sites in qwen3-flavoured
    modules so the reader knows which family they're in.
    """


class Qwen35RotaryEmbedding(RotaryEmbedding):
    """Qwen3.5 / Qwen3-Next family rotary: partial rotary (default 0.25)
    + optional interleaved mRoPE.

    Pure documentation subclass — body is the parent's parametric
    implementation, which already dispatches on the relevant
    ``rope_parameters`` flags.
    """


# --------------------------------------------------------------------------- #
# Interleaved RoPE (DeepSeek-V4 family)
# --------------------------------------------------------------------------- #
# The qwen3 / qwen3.5 path above uses *paired* RoPE: the head dim is split
# in half, the two halves are rotated against each other via
# `cat([first_half, second_half])` and `rotate_half` returns
# `cat([-second_half, first_half])`. DeepSeek-V4 instead uses *interleaved*
# RoPE: consecutive channel pairs (0,1), (2,3), (4,5), ... are rotated
# together. This needs a different `rotate_half` body and a different
# `apply_rotary_pos_emb` (single-tensor signature; cos/sin come in
# half-size and are expanded with `repeat_interleave(2)`).
#
# Helpers ported from
# ``references/dsv4/modeling_deepseek_v4.py:335-359``
# (rotate_half + apply_rotary_pos_emb).


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive channel pairs (DSV4 interleaved RoPE).

    Pairs ``(x_0, x_1), (x_2, x_3), ...`` are mapped to
    ``(-x_1, x_0), (-x_3, x_2), ...``. Used by
    :func:`apply_rotary_pos_emb_interleaved`.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply interleaved RoPE to the trailing rope slice of ``x``.

    ``cos`` / ``sin`` arrive at half-size (one entry per interleaved pair,
    from :class:`DeepseekV4RotaryEmbedding`); we expand to the full rope
    dim with ``repeat_interleave(2)``, then rotate the last
    ``2 * cos.shape[-1]`` channels of ``x`` with the standard
    ``x*cos + rotate_half(x)*sin`` formula in fp32. The leading ``nope``
    channels (head layout is ``[nope | rope]``) pass through unchanged.

    Single-tensor signature (vs the qwen3 helper's ``(q, k)`` pair) —
    DSV4 attention applies rotary separately to query and key.
    """
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (rotate_half_interleaved(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class DeepseekV4RotaryEmbedding(nn.Module):
    """DeepSeek-V4 rotary embedding (interleaved, multi-rope-type).

    DSV4 carries TWO sets of rope parameters in ``config.rope_parameters``,
    keyed by rope-type label rather than architecture layer type::

        config.rope_parameters = {
            "main":     {"rope_type": "default", "rope_theta": ..., "partial_rotary_factor": ...},
            "compress": {"rope_type": "default", "rope_theta": ..., ...},
        }

    The forward takes an extra ``layer_type`` argument (``"main"`` or
    ``"compress"``) that picks which inv_freq buffer to use. ``"main"`` is
    used by the standard attention path; ``"compress"`` is used inside
    HCA / CSA compressors where the RoPE base is different (typically a
    larger ``rope_theta`` to handle the sparse/compressed positions).

    No ``cat([freqs, freqs])`` duplication — the half-size cos/sin
    returned here is expanded by :func:`apply_rotary_pos_emb_interleaved`
    via ``repeat_interleave(2)``.

    Ported from ``references/dsv4/modeling_deepseek_v4.py:75-168``
    (DeepseekV4RotaryEmbedding).
    """

    inv_freq: torch.Tensor

    def __init__(self, config, device=None) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len_cached = getattr(config, "max_position_embeddings", 0)
        self.original_max_seq_len = self.max_seq_len_cached

        rope_params = config.rope_parameters or {}
        # Only sub-dicts are real per-rope-type entries — the top-level
        # ``rope_type`` key (left over by ``convert_rope_params_to_dict``
        # in some configs) is a flat-shape leftover, not a layer.
        self.layer_types = [k for k, v in rope_params.items() if isinstance(v, dict)]
        self.rope_type: dict[str, str] = {}
        for layer_type in self.layer_types:
            sub_params = rope_params[layer_type]
            self.rope_type[layer_type] = sub_params.get("rope_type", "default")
            rope_init_fn: Callable = self._compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(config, layer_type=layer_type, device=device)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def _compute_default_rope_parameters(
        config,
        device=None,
        seq_len=None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Default RoPE per-layer-type: read base + partial factor from
        ``config.rope_parameters[layer_type]``, build inv_freq."""
        del seq_len
        sub_params = config.rope_parameters[layer_type]
        base = sub_params["rope_theta"]
        partial_rotary_factor = float(sub_params.get("partial_rotary_factor", 1.0))
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_type is None:
            raise ValueError(
                "DeepseekV4RotaryEmbedding.forward requires layer_type "
                f"({list(self.layer_types)!r}); pass 'main' or 'compress'."
            )
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos() * attention_scaling
            sin = freqs.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
