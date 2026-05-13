"""Model-level roofline analysis for DFlash speculative decoding.

Builds on ``specs_dflash.DFlashAttentionSpec`` to model:

  * One DFlash draft forward (fc + hidden_norm + N draft layers +
    final norm). Embedding + lm_head live on the **target** so they're
    NOT charged here.
  * One speculation round = draft forward + target verification
    mini-prefill on the block.
  * Steady-state throughput = ``(avg_acc + 1) / (t_draft + t_target_verify)``
    at a chosen past-cache length.

Workload axes that matter:

    block_size   draft & target Q per round (default 16)
    ctx_len      target_hidden length this round (= acc + 1 steady state;
                                                   = prompt_len at round 1)
    past_cache   draft + target KV cache length (= confirmed prefix end)
    avg_acc      expected accepted tokens per round (from MTP table or
                                                     measurement; default 1.44)
    L_anchor     len(target_layer_ids); reads from
                 ``config.dflash_config["target_layer_ids"]`` when present

The "round 1 is heavy" caveat (because ctx_len = prompt_len there) is
exposed as a separate ``round1=True`` flag; default analysis is
steady-state (round 2..N average).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch

from .analyze import (
    ModuleStat,
    RooflineReport,
    roofline_mini_prefill,
)
from .hardware import Hardware, get_hardware
from .specs import LinearSpec, RMSNormSpec, dtype_size
from .specs_dflash import DFlashAttentionSpec
from .specs_ffn import SwiGLUMLPSpec


_DRAFT_ATTN_SPEC = DFlashAttentionSpec()
_DRAFT_MLP_SPEC = SwiGLUMLPSpec()
_NORM_SPEC = RMSNormSpec()


# --------------------------------------------------------------------------- #
# Single DFlash draft forward
# --------------------------------------------------------------------------- #


def roofline_dflash_draft_forward(
    config,
    *,
    batch: int,
    block_size: int,
    ctx_len: int,
    past_cache_len: int = 0,
    hardware="H100",
    dtype: torch.dtype = torch.bfloat16,
) -> RooflineReport:
    """Roofline analysis of ONE DFlash draft forward.

    Composition (matches ``dflash/dflash/model.py:DFlashDraftModel.forward``):
      fc Linear(H * L_anchor, H) on ctx_len tokens         (once)
      hidden_norm RMSNorm on [ctx_len, H]                  (once)
      num_hidden_layers blocks of:
        input_layernorm RMSNorm on [q_len, H]              (per layer)
        DFlashAttentionSpec at (ctx_len, q_len, past_cache_len)
        post_attention_layernorm RMSNorm on [q_len, H]     (per layer)
        SwiGLUMLPSpec on [q_len, H]                        (per layer)
      final RMSNorm on [q_len, H]                          (once)

    Embedding lookup and lm_head are TARGET's; charged separately in the
    speculation-round wrapper.
    """
    hardware = get_hardware(hardware)
    es = dtype_size(dtype)
    q_len = block_size
    hidden = config.hidden_size
    n_layers = config.num_hidden_layers
    nq = batch * q_len
    nctx = batch * ctx_len

    # L_anchor: read from dflash_config when present; fall back to
    # num_target_layers / num_hidden_layers heuristic.
    target_layer_ids = None
    dflash_cfg = getattr(config, "dflash_config", None)
    if dflash_cfg is not None:
        target_layer_ids = dflash_cfg.get("target_layer_ids") if isinstance(dflash_cfg, dict) else None
    if target_layer_ids is None:
        # Default per build_target_layer_ids logic in dflash/model.py
        target_layer_ids = list(range(n_layers))  # best-effort fallback
    l_anchor = len(target_layer_ids)

    report = RooflineReport(
        config=config,
        batch=batch,
        query_len=q_len,
        kv_cache_len=past_cache_len,
        dtype=dtype,
        hardware=hardware,
    )

    # ---- fc + hidden_norm (once per forward, on ctx_len tokens) ----
    fc_spec = LinearSpec(in_features=l_anchor * hidden, out_features=hidden, bias=False)
    fc_in_shape = (batch, ctx_len, l_anchor * hidden)
    report.modules.append(ModuleStat(
        kind="dflash_pre", name="fc",
        layer_idx=None,
        flops=fc_spec.flops(fc_in_shape),
        bytes=fc_spec.bytes(fc_in_shape, dtype=dtype),
    ))
    report.modules.append(ModuleStat(
        kind="dflash_pre", name="hidden_norm",
        layer_idx=None,
        flops=_NORM_SPEC.flops((batch, ctx_len, hidden)),
        bytes=_NORM_SPEC.bytes((batch, ctx_len, hidden), dtype=dtype),
    ))

    # ---- N x (input_norm + attn + post_attn_norm + mlp) ----
    layer_q_shape = (batch, q_len, hidden)
    attn_kwargs = {"kv_cache_len": past_cache_len, "ctx_len": ctx_len}
    for i in range(n_layers):
        report.modules.append(ModuleStat(
            kind="norm", name="input_layernorm", layer_idx=i,
            flops=_NORM_SPEC.flops(layer_q_shape),
            bytes=_NORM_SPEC.bytes(layer_q_shape, dtype=dtype),
        ))
        report.modules.append(ModuleStat(
            kind="mixer", name="dflash_attention", layer_idx=i,
            flops=_DRAFT_ATTN_SPEC.flops(layer_q_shape, config, **attn_kwargs),
            bytes=_DRAFT_ATTN_SPEC.bytes(layer_q_shape, config, dtype, **attn_kwargs),
        ))
        report.modules.append(ModuleStat(
            kind="norm", name="post_attention_layernorm", layer_idx=i,
            flops=_NORM_SPEC.flops(layer_q_shape),
            bytes=_NORM_SPEC.bytes(layer_q_shape, dtype=dtype),
        ))
        report.modules.append(ModuleStat(
            kind="ffn", name="qwen3_mlp", layer_idx=i,
            flops=_DRAFT_MLP_SPEC.flops(layer_q_shape, config),
            bytes=_DRAFT_MLP_SPEC.bytes(layer_q_shape, config, dtype),
        ))

    # ---- Final norm ----
    report.modules.append(ModuleStat(
        kind="norm", name="final_norm", layer_idx=None,
        flops=_NORM_SPEC.flops(layer_q_shape),
        bytes=_NORM_SPEC.bytes(layer_q_shape, dtype=dtype),
    ))

    return report


# --------------------------------------------------------------------------- #
# One speculation round = draft forward + target verify
# --------------------------------------------------------------------------- #


def roofline_dflash_speculation_round(
    target_config,
    draft_config,
    *,
    batch: int,
    block_size: int = 16,
    past_cache_len: int = 0,
    ctx_len: Optional[int] = None,
    target_hardware="H100",
    draft_hardware: Optional[object] = None,
    dtype: torch.dtype = torch.bfloat16,
    round1: bool = False,
    prompt_len: Optional[int] = None,
) -> dict:
    """Roofline analysis of one DFlash speculation round.

    Returns a dict with keys::

        draft  : RooflineReport          # one draft forward
        target : RooflineReport          # one target mini-prefill (Q=block_size)
        t_round         : float          # max(draft.time, 0) + target.time on each hw
                                          # — assumes serial draft-then-verify
        tokens_per_round: float          # avg_acc + 1; uses block_size as
                                          # the optimistic upper bound (full acceptance)
                                          # NB: this function doesn't know acc_len;
                                          # caller supplies via the throughput wrapper.

    Args:
        target_config / draft_config : SimpleNamespace-style configs (use the
            builders in alloy.examples.roofline.{dsv4_pro, dflash.*})
        batch, block_size, past_cache_len : workload axes
        ctx_len : target_hidden length. Default is ``prompt_len`` if
            ``round1=True``, else ``block_size`` (a conservative steady-state
            stand-in for ``avg_acc + 1`` — call sites that know the true
            acceptance length should pass it explicitly).
        target_hardware / draft_hardware : hardware for each side. If
            ``draft_hardware`` is None, defaults to ``target_hardware`` (the
            common production scenario where main + draft co-locate).
        dtype : compute dtype for both.
        round1 : if True, ``ctx_len`` defaults to ``prompt_len`` (the heavy
            first round where the draft sees the whole prompt as context).
        prompt_len : required when ``round1=True``.
    """
    if draft_hardware is None:
        draft_hardware = target_hardware
    if ctx_len is None:
        if round1:
            if prompt_len is None:
                raise ValueError("round1=True requires prompt_len")
            ctx_len = prompt_len
        else:
            ctx_len = block_size  # conservative; replace via wrapper

    draft_report = roofline_dflash_draft_forward(
        draft_config,
        batch=batch,
        block_size=block_size,
        ctx_len=ctx_len,
        past_cache_len=past_cache_len,
        hardware=draft_hardware,
        dtype=dtype,
    )
    target_report = roofline_mini_prefill(
        target_config,
        batch=batch,
        chunk_len=block_size,
        kv_cache_len=past_cache_len,
        hardware=target_hardware,
        dtype=dtype,
    )

    return {
        "draft": draft_report,
        "target": target_report,
        "t_round": draft_report.roofline_time_s + target_report.roofline_time_s,
    }


# --------------------------------------------------------------------------- #
# Steady-state throughput
# --------------------------------------------------------------------------- #


def roofline_dflash_steady_throughput(
    target_config,
    draft_config,
    *,
    batch: int,
    past_cache_len: int,
    block_size: int = 16,
    avg_accept_len: float = 1.44,
    target_hardware="H100",
    draft_hardware: Optional[object] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """End-to-end speculation throughput at a representative context length.

    Uses ``ctx_len = ceil(avg_accept_len + 1)`` for the draft (steady-state
    approximation — at round k the target_hidden has ``acc_{k-1} + 1`` tokens
    which equals avg_accept + 1 on average).

    Returns a dict with::

        draft           : RooflineReport
        target          : RooflineReport
        t_round         : float           seconds per speculation round
        tokens_per_round: float           = avg_accept_len + 1
        effective_tps   : float           = B * tokens_per_round / t_round
        ar_baseline_tps : float           single-token target decode throughput
                                           at the same past_cache_len, for ratio
        speedup_vs_ar   : float           effective_tps / ar_baseline_tps

    Compare ``speedup_vs_ar > 1`` is the headline number: 'how much faster
    is DFlash speculation than pure autoregressive decode here?'
    """
    from .analyze import roofline_decode  # late import to avoid cycle perception

    ctx_len = max(1, int(round(avg_accept_len + 1)))
    round_data = roofline_dflash_speculation_round(
        target_config, draft_config,
        batch=batch, block_size=block_size,
        past_cache_len=past_cache_len, ctx_len=ctx_len,
        target_hardware=target_hardware, draft_hardware=draft_hardware,
        dtype=dtype,
    )
    tokens_per_round = avg_accept_len + 1.0
    effective_tps = batch * tokens_per_round / round_data["t_round"]

    # Baseline: autoregressive decode on the target alone at the same context.
    ar_report = roofline_decode(
        target_config, batch=batch, kv_cache_len=past_cache_len,
        dtype=dtype, hardware=target_hardware,
    )
    ar_tps = ar_report.tokens_per_sec  # B * 1 / t_ar  by definition

    return {
        **round_data,
        "tokens_per_round": tokens_per_round,
        "effective_tps": effective_tps,
        "ar_baseline_tps": ar_tps,
        "ar_baseline_t_round": ar_report.roofline_time_s,
        "speedup_vs_ar": effective_tps / ar_tps if ar_tps > 0 else 0.0,
    }


__all__ = [
    "roofline_dflash_draft_forward",
    "roofline_dflash_speculation_round",
    "roofline_dflash_steady_throughput",
]
