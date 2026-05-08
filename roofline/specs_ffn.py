"""FFN-family roofline specs: SwiGLU MLP + DSV4 / Qwen3.5 MoE blocks.

Covers all FFN names currently in :data:`alloy.modules.registry.FFN_REGISTRY`:

  * ``qwen3_mlp``      — dense SwiGLU MLP (gate + up + down, bias=False)
  * ``dsv4_moe``       — DSV4 sparse MoE with topk router + shared expert
  * ``dsv4_hash_moe``  — DSV4 sparse MoE with hash router + shared expert
  * ``qwen3_5_moe``    — Qwen3.5 sparse MoE with topk router + gated shared expert

All specs assume optimal operator fusion: the SwiGLU intermediate ``[N, 2I]``
and ``[N, I]`` activations stay in SRAM (don't spill to HBM), so HBM traffic
is just (weight reads + input read + output write). This is the roofline
upper bound; real implementations land at or below.

FLOPs of activation / clamp / score functions are approximate (~5 flops per
element). They're tiny relative to matmul terms (1-1000x smaller for typical
hidden_size / intermediate_size); precision is not load-bearing.

For MoE expert weight bytes we use ``min(num_experts, n_tokens * top_k)``
under the user-stated rule "we don't model routing distribution" — this is
the optimal-routing upper bound (every expert weight loaded at most once
per forward) and matches both prefill (N*top_k >> E, all experts loaded)
and decode (N=1, only top_k experts loaded) regimes naturally.
"""
from __future__ import annotations

from math import prod

import torch

from .specs import RooflineSpec, dtype_size, register_spec


# --------------------------------------------------------------------------- #
# SwiGLU helpers — used by SwiGLUMLPSpec and inside both MoE specs.
# --------------------------------------------------------------------------- #


# Approximate flops per element for SwiGLU's activation + gating mul:
#   silu(gate) = gate * sigmoid(gate)  ~ 4 flops (sigmoid + mul)
#   times up                           ~ 1 flop
# Total ~5 flops/elem on the [N, I] intermediate tensor. Some impls also
# clamp gate/up (DSV4 with swiglu_limit), adding 2 flops/elem; ignored
# here as below the noise floor of matmul terms.
_SWIGLU_ACT_FLOPS_PER_ELEM = 5


def _swiglu_flops(n_tokens: int, hidden: int, intermediate: int, bias: bool = False) -> int:
    """FLOPs for one SwiGLU MLP forward on ``n_tokens`` tokens.

    Matmul terms:
      gate + up (combined as gate_up [H -> 2I]):  4 * N * H * I
      down [I -> H]:                              2 * N * H * I
      total matmul:                               6 * N * H * I

    Activation + gating mul: ~5 flops per [N, I] elem.
    Bias (when present):
      gate bias adds: N * I,  up bias adds: N * I,  down bias adds: N * H.
    """
    matmul = 6 * n_tokens * hidden * intermediate
    activation = _SWIGLU_ACT_FLOPS_PER_ELEM * n_tokens * intermediate
    bias_flops = (2 * n_tokens * intermediate + n_tokens * hidden) if bias else 0
    return matmul + activation + bias_flops


def _swiglu_weight_bytes(hidden: int, intermediate: int, es: int, bias: bool = False) -> int:
    """Weight bytes for one SwiGLU MLP (gate + up + down, optional bias)."""
    weights = 3 * hidden * intermediate * es
    if bias:
        weights += (2 * intermediate + hidden) * es
    return weights


# --------------------------------------------------------------------------- #
# SwiGLUMLPSpec — dense SwiGLU MLP (qwen3_mlp and friends).
# --------------------------------------------------------------------------- #


class SwiGLUMLPSpec(RooflineSpec):
    """Spec for a dense SwiGLU MLP: ``down(silu(gate(x)) * up(x))``.

    Optimal-fusion bytes assume the entire MLP runs as one fused kernel:
    the ``[N, 2I]`` and ``[N, I]`` intermediate activations stay in SRAM,
    only the ``[N, H]`` input read and output write hit HBM.

    Config field names are configurable so the same spec class works for
    several alloy modules whose source-coupled config fields differ:

      * qwen3_mlp / DSV4 shared expert: ``intermediate_size``, ``mlp_bias``
      * Qwen3.5 shared expert: ``shared_expert_intermediate_size``,
        ``mlp_bias`` (defaults False if absent)
    """

    def __init__(
        self,
        intermediate_attr: str = "intermediate_size",
        bias_attr: str = "mlp_bias",
    ) -> None:
        self.intermediate_attr = intermediate_attr
        self.bias_attr = bias_attr

    def _read_config(self, config) -> tuple[int, bool]:
        intermediate = getattr(config, self.intermediate_attr)
        bias = bool(getattr(config, self.bias_attr, False))
        return intermediate, bias

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        n_tokens = prod(in_shape[:-1])
        hidden = in_shape[-1]
        intermediate, bias = self._read_config(config)
        return _swiglu_flops(n_tokens, hidden, intermediate, bias)

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        n_tokens = prod(in_shape[:-1])
        hidden = in_shape[-1]
        intermediate, bias = self._read_config(config)
        es = dtype_size(dtype)
        weights = _swiglu_weight_bytes(hidden, intermediate, es, bias)
        act_in = n_tokens * hidden * es
        act_out = n_tokens * hidden * es
        return weights + act_in + act_out


# --------------------------------------------------------------------------- #
# DSV4MoESpec — shared between dsv4_moe (topk) and dsv4_hash_moe (hash).
# --------------------------------------------------------------------------- #


class DSV4MoESpec(RooflineSpec):
    """Spec for DSV4 sparse MoE block.

    Both ``dsv4_moe`` (topk router) and ``dsv4_hash_moe`` (hash router) share
    identical routed-expert and shared-expert math; the only difference is
    the router buffer:

      * topk: small ``e_score_correction_bias`` of shape ``[num_experts]``
      * hash: large ``tid2eid`` lookup table of shape
        ``[vocab_size, top_k]`` in int64 (8 bytes/elem). Each token reads
        one row (``top_k * 8`` bytes), so total hash-table read is
        ``n_tokens * top_k * 8``.

    DSV4-coupled config field names: routed experts use ``n_routed_experts``
    (NOT ``num_experts``) and ``intermediate_size`` for both routed and
    shared expert dims (DSV4 reference uses the same MLP class for both).
    """

    def __init__(self, is_hash: bool = False) -> None:
        self.is_hash = is_hash

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        n_tokens = prod(in_shape[:-1])
        hidden = in_shape[-1]
        intermediate = config.intermediate_size
        n_experts = config.n_routed_experts
        top_k = config.num_experts_per_tok
        bias = bool(getattr(config, "mlp_bias", False))

        # Router score matmul: [N, H] @ [H, E]
        router_matmul = 2 * n_tokens * hidden * n_experts
        # Score function (sqrtsoftplus / sigmoid family): ~5 flops/elem.
        # Approximate; matmul dominates.
        router_score = 5 * n_tokens * n_experts
        # Top-k normalization: sum + divide per (token, k)
        router_norm = 2 * n_tokens * top_k
        # Selection cost (topk sort or hash lookup): tiny, ignored.

        # Routed experts: N * top_k SwiGLU forwards (one token each)
        per_expert = _swiglu_flops(1, hidden, intermediate, bias)
        routed = n_tokens * top_k * per_expert

        # Shared expert: SwiGLU on all N tokens
        shared = _swiglu_flops(n_tokens, hidden, intermediate, bias)

        # Final add (routed + shared): N * H
        add = n_tokens * hidden

        return router_matmul + router_score + router_norm + routed + shared + add

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        n_tokens = prod(in_shape[:-1])
        hidden = in_shape[-1]
        intermediate = config.intermediate_size
        n_experts = config.n_routed_experts
        top_k = config.num_experts_per_tok
        bias = bool(getattr(config, "mlp_bias", False))
        es = dtype_size(dtype)

        # Router weights: [E, H]
        router_w = n_experts * hidden * es
        if self.is_hash:
            # tid2eid lookup: each token reads one [top_k] int64 row (8 B/elt).
            router_w += n_tokens * top_k * 8
        else:
            # e_score_correction_bias: [E] in compute dtype (negligible)
            router_w += n_experts * es

        # Routed expert weights: under any-routing upper bound, at most
        # min(E, N*top_k) unique experts get hit and have their weights loaded.
        unique_experts = min(n_experts, n_tokens * top_k)
        per_expert_w = _swiglu_weight_bytes(hidden, intermediate, es, bias)
        routed_w = unique_experts * per_expert_w

        # Shared expert weights: always loaded
        shared_w = _swiglu_weight_bytes(hidden, intermediate, es, bias)

        # Activations under fusion: input read once, output written once.
        # The internal expert dispatch (gather/scatter into [N, H]) is assumed
        # to live in SRAM in the fused-kernel ideal.
        act_in = n_tokens * hidden * es
        act_out = n_tokens * hidden * es

        return router_w + routed_w + shared_w + act_in + act_out


# --------------------------------------------------------------------------- #
# Qwen35MoESpec — Qwen3.5 sparse MoE with gated shared expert.
# --------------------------------------------------------------------------- #


class Qwen35MoESpec(RooflineSpec):
    """Spec for Qwen3.5 sparse MoE block (qwen3_5_moe).

    Differences from DSV4 MoE:

      * Different intermediate dims: routed experts use
        ``moe_intermediate_size``; shared expert uses
        ``shared_expert_intermediate_size``.
      * Router is plain topk softmax (no e_score_correction_bias).
      * Has a learned **shared_expert_gate** = ``nn.Linear(hidden, 1)``
        that produces a sigmoid scalar per token, multiplying the shared
        expert output before the residual add.
      * No bias on any linear (Qwen3.5 standard).

    Config field names: ``num_experts`` (Qwen-coupled, distinct from DSV4's
    ``n_routed_experts``), ``num_experts_per_tok``, ``moe_intermediate_size``,
    ``shared_expert_intermediate_size``.
    """

    def flops(self, in_shape: tuple[int, ...], config, **kwargs) -> int:
        n_tokens = prod(in_shape[:-1])
        hidden = in_shape[-1]
        moe_inter = config.moe_intermediate_size
        shared_inter = config.shared_expert_intermediate_size
        n_experts = config.num_experts
        top_k = config.num_experts_per_tok

        # Router: matmul + softmax + topk norm
        router_matmul = 2 * n_tokens * hidden * n_experts
        router_softmax = 5 * n_tokens * n_experts
        router_norm = 2 * n_tokens * top_k

        # Routed experts: N * top_k SwiGLU forwards at moe_intermediate_size
        per_expert = _swiglu_flops(1, hidden, moe_inter, bias=False)
        routed = n_tokens * top_k * per_expert

        # Shared expert: SwiGLU on all N tokens at shared_expert_intermediate_size
        shared = _swiglu_flops(n_tokens, hidden, shared_inter, bias=False)

        # Shared expert gate: Linear(H -> 1) + sigmoid + scalar mul on [N, H]
        # Linear: 2 * N * H * 1 = 2 * N * H
        # sigmoid: 3 * N (exp + add + div)
        # scalar broadcast mul on [N, H]: N * H
        shared_gate = 2 * n_tokens * hidden + 3 * n_tokens + n_tokens * hidden

        # Final add (routed + gated_shared): N * H
        add = n_tokens * hidden

        return (router_matmul + router_softmax + router_norm
                + routed + shared + shared_gate + add)

    def bytes(self, in_shape: tuple[int, ...], config, dtype: torch.dtype, **kwargs) -> int:
        n_tokens = prod(in_shape[:-1])
        hidden = in_shape[-1]
        moe_inter = config.moe_intermediate_size
        shared_inter = config.shared_expert_intermediate_size
        n_experts = config.num_experts
        top_k = config.num_experts_per_tok
        es = dtype_size(dtype)

        # Router weights: [E, H]
        router_w = n_experts * hidden * es

        # Routed expert weights: min(E, N*top_k) experts loaded
        unique_experts = min(n_experts, n_tokens * top_k)
        per_expert_w = _swiglu_weight_bytes(hidden, moe_inter, es, bias=False)
        routed_w = unique_experts * per_expert_w

        # Shared expert weights (always loaded), plus shared_expert_gate weight [H]
        shared_w = _swiglu_weight_bytes(hidden, shared_inter, es, bias=False)
        gate_w = hidden * es  # nn.Linear(hidden, 1) weight is just [1, hidden]

        # Activations under fusion: input read once, output written once.
        act_in = n_tokens * hidden * es
        act_out = n_tokens * hidden * es

        return router_w + routed_w + shared_w + gate_w + act_in + act_out


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
# Names must match register_ffn keys in alloy.modules.ffn.*. Importing this
# module triggers registration via the alloy.roofline package __init__.

register_spec("qwen3_mlp", SwiGLUMLPSpec())
register_spec("dsv4_moe", DSV4MoESpec(is_hash=False))
register_spec("dsv4_hash_moe", DSV4MoESpec(is_hash=True))
register_spec("qwen3_5_moe", Qwen35MoESpec())


__all__ = [
    "SwiGLUMLPSpec",
    "DSV4MoESpec",
    "Qwen35MoESpec",
]
