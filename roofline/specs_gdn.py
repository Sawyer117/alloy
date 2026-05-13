"""Gated DeltaNet (GDN) roofline spec — Qwen3.5 / m-a-p style linear attention.

Models the alloy port :class:`~alloy.modules.attention.qwen3_5_gdn.Qwen35GatedDeltaNet`
under both kernel paths:

  * **chunk mode** (``query_len > chunk_size``): chunk-form parallel kernel
    used in long-prefill / training. Each chunk does:
      - K · K^T (lower-tri),  + solve_tril → A = (I + tril(...))⁻¹
      - u = A · (v · β),  w = A · (k · β · g)
      - cross-chunk recurrence: h_new = decay · h + k^T · u
      - intra-chunk causal attention: o = (q · k^T · L_mask) · v_new + q · h
    Per token per head:  ``2·C·(3·K + 2·V) + 4·K·V``
  * **fused_recurrent mode** (``query_len <= chunk_size``): scalar recurrent
    used for decode and short prefill. Per token per head:  ``7·K·V``
    (state decay + state-correction read + state update + state read).

We model the **algorithm at the matmul level**, deliberately backend-agnostic.
Both fla's Triton kernels and alloy's pure-PyTorch fallback compute the same
underlying matmuls; they differ in how those matmuls are tiled and fused. The
inversion / cumsum / L2norm / RMSNormGated overhead (~5% combined for typical
sizes) is absorbed into the formulas as small constants — exact constants
depend on backend choice and aren't worth chasing.

Linear projections (the **dominant** cost for GDN — typically ~80% of total
FLOPs/token, vs ~20% for the linear attention itself):

  * ``in_proj_qkv``: hidden → ``2·key_dim + value_dim``    (fused QKV)
  * ``in_proj_z``:   hidden → ``value_dim``                (gate)
  * ``in_proj_b``:   hidden → ``num_v_heads``              (β, tiny)
  * ``in_proj_a``:   hidden → ``num_v_heads``              (α, tiny)
  * ``out_proj``:    ``value_dim`` → hidden

Plus a depthwise 1D causal conv1d of kernel ``conv_size`` (default 4) over
the concatenated Q/K/V channels: ``conv_dim = 2·key_dim + value_dim``.

Recurrent state shape per layer per batch: ``[H_v, K, V]`` — much smaller
than a transformer KV cache at long context, which is the headline GDN
advantage. Persists across forwards via cache.

Source-coupled config field names match the alloy port:

  * ``hidden_size``
  * ``linear_num_key_heads``    (= num_k_heads)
  * ``linear_num_value_heads``  (= num_v_heads, can exceed num_k_heads = GVA)
  * ``linear_key_head_dim``     (= K, head_k_dim)
  * ``linear_value_head_dim``   (= V, head_v_dim, typically K · expand_v)
  * ``linear_conv_kernel_dim``  (= conv_size, default 4)
"""
from __future__ import annotations

import torch

from .specs import RooflineSpec, dtype_size, register_spec


# Chunk size used by fla's chunk_gated_delta_rule kernel (hardcoded to 64
# in fla/ops/gated_delta_rule/chunk.py and the alloy port). Mode dispatch
# below uses this as the boundary: query_len > 64 -> chunk, else recurrent.
GDN_CHUNK_SIZE = 64


def _read_dims(config) -> dict:
    """Pull GDN dims off an alloy-style config (source-coupled field names)."""
    return {
        "hidden": config.hidden_size,
        "num_k_heads": config.linear_num_key_heads,
        "num_v_heads": config.linear_num_value_heads,
        "head_k_dim": config.linear_key_head_dim,
        "head_v_dim": config.linear_value_head_dim,
        "conv_size": config.linear_conv_kernel_dim,
    }


def _linear_proj_flops(n_tokens: int, d: dict) -> int:
    """Sum of all 5 linear projections (matmul flops only).

    in_proj_qkv emits ``2·key_dim + value_dim`` outputs in one matmul; same
    total flops as separate q_proj / k_proj / v_proj. The per-head bias /
    A_log / dt_bias parameters are tiny and ignored on the flop side.
    """
    H = d["hidden"]
    K_d = d["num_k_heads"] * d["head_k_dim"]
    V_d = d["num_v_heads"] * d["head_v_dim"]
    H_v = d["num_v_heads"]
    return (
        2 * n_tokens * H * (2 * K_d + V_d)   # in_proj_qkv
        + 2 * n_tokens * H * V_d              # in_proj_z (gate)
        + 2 * n_tokens * H * H_v              # in_proj_b
        + 2 * n_tokens * H * H_v              # in_proj_a
        + 2 * n_tokens * V_d * H              # out_proj
    )


def _linear_proj_weight_bytes(d: dict, es: int) -> int:
    H = d["hidden"]
    K_d = d["num_k_heads"] * d["head_k_dim"]
    V_d = d["num_v_heads"] * d["head_v_dim"]
    H_v = d["num_v_heads"]
    return (
        H * (2 * K_d + V_d) * es      # in_proj_qkv
        + H * V_d * es                 # in_proj_z
        + H * H_v * es                 # in_proj_b
        + H * H_v * es                 # in_proj_a
        + V_d * H * es                 # out_proj
    )


def _conv_flops(n_tokens: int, d: dict) -> int:
    """Depthwise 1D causal conv over concatenated Q/K/V channels.

    conv_dim = 2·key_dim + value_dim; depthwise so flops = conv_size · conv_dim
    mul-adds per token.
    """
    K_d = d["num_k_heads"] * d["head_k_dim"]
    V_d = d["num_v_heads"] * d["head_v_dim"]
    conv_dim = 2 * K_d + V_d
    return 2 * n_tokens * d["conv_size"] * conv_dim


def _conv_weight_bytes(d: dict, es: int) -> int:
    K_d = d["num_k_heads"] * d["head_k_dim"]
    V_d = d["num_v_heads"] * d["head_v_dim"]
    conv_dim = 2 * K_d + V_d
    return d["conv_size"] * conv_dim * es


def _state_bytes(batch: int, d: dict, es: int) -> int:
    """Recurrent state h: [B, H_v, K, V]."""
    return batch * d["num_v_heads"] * d["head_k_dim"] * d["head_v_dim"] * es


def _conv_state_bytes(batch: int, d: dict, es: int) -> int:
    """Short conv ring buffer: [B, conv_size-1, conv_dim]. Tiny."""
    K_d = d["num_k_heads"] * d["head_k_dim"]
    V_d = d["num_v_heads"] * d["head_v_dim"]
    conv_dim = 2 * K_d + V_d
    return batch * (d["conv_size"] - 1) * conv_dim * es


def _chunk_attn_flops(n_tokens: int, d: dict) -> int:
    """Chunk-mode GDN attention (training/long-prefill path).

    Per token per head:
      KK^T (lower-tri):              2·C²·K  (per chunk) → 2·C·K  (per token)
      solve_tril for u:              2·C²·V          → 2·C·V
      solve_tril for w:              2·C²·K          → 2·C·K
      cross-chunk h update:          2·K·C·V         → 2·K·V
      intra-chunk q·k^T:             2·C²·K          → 2·C·K
      intra-chunk score · v_new:     2·C²·V          → 2·C·V
      cross-chunk o = q · h:         2·C·K·V         → 2·K·V

      sum: 6·C·K + 4·C·V + 4·K·V = 2·C·(3K + 2V) + 4·K·V

    Inversion / forward-substitution (the explicit ``for i in range(1,C)``
    loop in naive PyTorch, or block-recursive solve in Triton) is small —
    ~O(C²/log C) flops per chunk per head — and absorbed into the constants
    above as <5% noise. Backend-specific.
    """
    K = d["head_k_dim"]
    V = d["head_v_dim"]
    H_v = d["num_v_heads"]
    C = GDN_CHUNK_SIZE
    return n_tokens * H_v * (2 * C * (3 * K + 2 * V) + 4 * K * V)


def _recurrent_attn_flops(n_tokens: int, d: dict) -> int:
    """Fused recurrent (decode / short prefill) GDN attention.

    Per token per head (from naive_recurrent_gated_delta_rule):
      h *= g.exp()                # K·V mul        (state decay)
      v -= sum(h * k[:,None], -2) # 2·K·V          (state-correction read)
      v *= beta                   # V mul          (small, ignored)
      h += k ⊗ v                  # 2·K·V          (state update via outer product)
      o = q · h                   # 2·K·V          (state read)
                                  ────────────────
                                  total ≈ 7·K·V per token per head

    L2norm of q/k (when ``use_qk_l2norm_in_kernel=True``) adds ~8·K per
    head per token; <1% of total, ignored.
    """
    K = d["head_k_dim"]
    V = d["head_v_dim"]
    H_v = d["num_v_heads"]
    return n_tokens * H_v * 7 * K * V


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


class Qwen35GDNSpec(RooflineSpec):
    """Spec for ``qwen3_5_gdn`` (Qwen3.5 / m-a-p Gated DeltaNet linear attn).

    Mode dispatch is driven purely by ``query_len`` per the alloy port's
    boundary::

        mode = 'fused_recurrent' if query_len <= GDN_CHUNK_SIZE else 'chunk'

    ``kv_cache_len`` does **not** change which mode is used — only whether
    the recurrent state is read at the start (cached) or initialized to
    zeros (cold start). Bytes scale accordingly.

    The state itself is the same shape regardless of cache length, since
    GDN's recurrent state is a fixed-size summary — this is the headline
    long-context advantage over softmax attention's O(T) KV cache.

    Variables: B = batch, Q = query_len, N = B * Q, H = hidden_size,
    Nk = num_k_heads, Nv = num_v_heads, hkd = head_k_dim, hvd = head_v_dim,
    qkv_dim = sum of {q,k,v}-proj output dims, K = conv_kernel_size,
    C = GDN_CHUNK_SIZE (=64), es = dtype_size(dtype). State size per batch
    element = Nv * hkd * hvd (kept as fp32) plus the conv ring buffer
    [(K-1) * (Nk*hkd + 2*Nv*hvd) * es].

    ## FLOPs (mode-independent base + mode-dependent attention)
        linear_proj_flops(N, d)         # q/k/v/o/gate projections (see _linear_proj_flops)
      + conv_flops(N, d)                # depthwise conv1d on q/k/v concatenated
      + attn_flops                      # one of:
            chunk_attn_flops(N, d)      #   if Q > C  (chunk path, ~O(N * C * dim))
            recurrent_attn_flops(N, d)  #   else      (token-by-token recurrence, O(N * dim))

    ## Bytes (fused, recurrent state amortized)
        linear_proj_weight_bytes(d)     # Q/K/V/Gate/Out projection weights
      + conv_weight_bytes(d)            # conv1d weights + bias
      + head_v_dim * es                 # RMSNormGated weight
      + 2 * Nv * 4                      # A_log + dt_bias (fp32, tiny)
      + state_io                        # state_bytes (always written) + (read if P > 0)
      + conv_state_io                   # (K-1)-entry ring buffer, similar pattern
      + N * H * es                      # activation read
      + N * H * es                      # activation write

    The state-and-conv-buffer terms are the only cache-aware bytes; both are
    fixed size (do NOT grow with kv_cache_len). That's the math reason GDN
    wins long-context decode — softmax attention's KV scales O(P), GDN
    stays O(1).
    """

    def _check_in_shape(self, in_shape: tuple[int, ...]) -> tuple[int, int]:
        if len(in_shape) != 3:
            raise ValueError(
                f"Qwen35GDNSpec expects 3D in_shape (batch, query_len, hidden); "
                f"got {in_shape}."
            )
        return in_shape[0], in_shape[1]

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        batch, query_len = self._check_in_shape(in_shape)
        d = _read_dims(config)
        n_tokens = batch * query_len

        # Linear projections + conv1d are mode-independent — same compute
        # regardless of how the chunk/recurrent path is structured.
        flops = _linear_proj_flops(n_tokens, d)
        flops += _conv_flops(n_tokens, d)

        # GDN attention: dispatch by query_len
        if query_len > GDN_CHUNK_SIZE:
            flops += _chunk_attn_flops(n_tokens, d)
        else:
            flops += _recurrent_attn_flops(n_tokens, d)

        return flops

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        kv_cache_len = int(kwargs.get("kv_cache_len", 0))
        batch, query_len = self._check_in_shape(in_shape)
        d = _read_dims(config)
        es = dtype_size(dtype)
        n_tokens = batch * query_len

        # Weight bytes
        weights = _linear_proj_weight_bytes(d, es)
        weights += _conv_weight_bytes(d, es)
        # Norm gain + per-v-head scalars (A_log, dt_bias) — small but real.
        weights += d["head_v_dim"] * es              # Qwen35RMSNormGated weight
        weights += 2 * d["num_v_heads"] * 4          # A_log + dt_bias (fp32)

        # Recurrent state I/O — the only cache-aware byte term.
        # The state is always written (use_cache assumption, populates next
        # forward's prefix). It's read only when there's an existing prefix.
        state_size = _state_bytes(batch, d, es)
        state_io = state_size  # write
        if kv_cache_len > 0:
            state_io += state_size  # read

        # Short conv ring buffer (kernel_size - 1 history entries). Tiny.
        conv_state_size = _conv_state_bytes(batch, d, es)
        conv_state_io = conv_state_size  # always written
        if kv_cache_len > 0:
            conv_state_io += conv_state_size

        # Activations under fusion: input read once, output written once.
        # All intermediate tensors (Q/K/V proj outputs, conv outputs, gate,
        # scores, etc.) stay in SRAM under the fused-kernel ideal.
        act_in = n_tokens * d["hidden"] * es
        act_out = n_tokens * d["hidden"] * es

        return weights + state_io + conv_state_io + act_in + act_out


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


register_spec("qwen3_5_gdn", Qwen35GDNSpec())


__all__ = [
    "Qwen35GDNSpec",
    "GDN_CHUNK_SIZE",
]
