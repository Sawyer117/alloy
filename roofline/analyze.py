"""Top-level roofline analysis: walk config, dispatch to specs, aggregate.

The core entry point is :func:`roofline`, which models ONE forward pass with::

    query_len     — number of new tokens to process this forward
    kv_cache_len  — number of tokens already cached from previous forwards
                    (0 = cold prefill)

Three convenience wrappers cover the common LLM-serving modes:

  * :func:`roofline_prefill` — cold start, full prompt at once
  * :func:`roofline_mini_prefill` — one chunk of chunked prefill
  * :func:`roofline_decode` — generate a single token given an existing cache

All three reduce to the same underlying call; the wrappers just spell out
the (query_len, kv_cache_len) combination clearly. The report's header
auto-detects the mode label from those two values.

No model construction. Pure analytical, config-driven. Works for any model
size — memory cost is O(num_layers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from .hardware import Hardware, get_hardware
from .specs import RMSNormSpec, dtype_size, get_spec


# --------------------------------------------------------------------------- #
# Auto-scaled formatting helpers (used by RooflineReport.__str__)
# --------------------------------------------------------------------------- #


def _scale_flops(n: float) -> str:
    if n == 0:
        return "0 FLOPs"
    for prefix, threshold in (
        ("TFLOPs", 1e12), ("GFLOPs", 1e9), ("MFLOPs", 1e6), ("KFLOPs", 1e3),
    ):
        if abs(n) >= threshold:
            return f"{n / threshold:.3f} {prefix}"
    return f"{int(n):,} FLOPs"


def _scale_bytes(n: float) -> str:
    if n == 0:
        return "0 B"
    for prefix, threshold in (
        ("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3),
    ):
        if abs(n) >= threshold:
            return f"{n / threshold:.3f} {prefix}"
    return f"{int(n)} B"


def _scale_time(t: float) -> str:
    if t == 0:
        return "0 ns"
    if t >= 1.0:
        return f"{t:.3f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.3f} ms"
    if t >= 1e-6:
        return f"{t * 1e6:.3f} us"
    return f"{t * 1e9:.3f} ns"


def _dtype_short(dtype: torch.dtype) -> str:
    s = str(dtype).replace("torch.", "")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
            "float64": "fp64", "int8": "i8"}.get(s, s)


# --------------------------------------------------------------------------- #
# Per-module stat + aggregated report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModuleStat:
    """One module's contribution to the report.

    ``kind`` describes the role: ``"mixer"``, ``"ffn"``, ``"embedding"``,
    ``"norm"``, ``"lm_head"``, or ``"unknown"`` (no spec registered).
    """

    kind: str
    name: str
    layer_idx: Optional[int]
    flops: int
    bytes: int

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes if self.bytes > 0 else 0.0


@dataclass
class RooflineReport:
    """Aggregated per-module breakdown + totals + bottleneck analysis.

    Carries both ``query_len`` (new tokens this forward) and ``kv_cache_len``
    (already-cached tokens) so the mode label can be inferred and the
    underlying forward pass is fully described by the report.
    """

    config: object              # AlloyConfig — typed loosely to avoid hard import cycle
    batch: int
    query_len: int
    kv_cache_len: int
    dtype: torch.dtype
    hardware: Hardware
    modules: list[ModuleStat] = field(default_factory=list)

    @property
    def total_flops(self) -> int:
        return sum(m.flops for m in self.modules)

    @property
    def total_bytes(self) -> int:
        return sum(m.bytes for m in self.modules)

    @property
    def arithmetic_intensity(self) -> float:
        b = self.total_bytes
        return self.total_flops / b if b > 0 else 0.0

    @property
    def compute_time_s(self) -> float:
        peak = self.hardware.get_peak_flops(self.dtype)
        return self.total_flops / peak if peak > 0 else 0.0

    @property
    def memory_time_s(self) -> float:
        bw = self.hardware.hbm_bandwidth
        return self.total_bytes / bw if bw > 0 else 0.0

    @property
    def bottleneck(self) -> str:
        return "compute" if self.compute_time_s >= self.memory_time_s else "memory"

    @property
    def roofline_time_s(self) -> float:
        return max(self.compute_time_s, self.memory_time_s)

    @property
    def mode(self) -> str:
        """Inferred mode label from (query_len, kv_cache_len)."""
        if self.kv_cache_len == 0:
            return "prefill"
        if self.query_len == 1:
            return "decode"
        return "mini-prefill"

    def _mode_label(self) -> str:
        if self.mode == "prefill":
            return f"prefill (B={self.batch}, Q={self.query_len})"
        if self.mode == "decode":
            return f"decode (B={self.batch}, cache={self.kv_cache_len})"
        return f"mini-prefill (B={self.batch}, Q={self.query_len}, cache={self.kv_cache_len})"

    def __str__(self) -> str:
        sep = "-" * 92
        header_cols = [
            f"{'idx':>4}",
            f"{'kind':<10}",
            f"{'name':<32}",
            f"{'flops':>14}",
            f"{'bytes':>12}",
            f"{'AI':>8}",
        ]
        lines = [
            f"Roofline | {self.hardware.name} | {_dtype_short(self.dtype)} | {self._mode_label()}",
            sep,
            "  ".join(header_cols),
            sep,
        ]
        for m in self.modules:
            idx = "-" if m.layer_idx is None else str(m.layer_idx)
            lines.append("  ".join([
                f"{idx:>4}",
                f"{m.kind:<10}",
                f"{m.name:<32}",
                f"{_scale_flops(m.flops):>14}",
                f"{_scale_bytes(m.bytes):>12}",
                f"{m.arithmetic_intensity:>8.1f}",
            ]))
        lines.append(sep)
        lines.append(
            f"TOTAL: {_scale_flops(self.total_flops)} / "
            f"{_scale_bytes(self.total_bytes)} / "
            f"AI = {self.arithmetic_intensity:.1f}"
        )
        lines.append(
            f"{self.hardware.name}: "
            f"compute={_scale_time(self.compute_time_s)}, "
            f"memory={_scale_time(self.memory_time_s)} "
            f"-> bottleneck={self.bottleneck} "
            f"({_scale_time(self.roofline_time_s)} / forward)"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Core entry point
# --------------------------------------------------------------------------- #


def roofline(
    config,
    batch: int,
    query_len: int,
    dtype: torch.dtype = torch.bfloat16,
    hardware: Union[str, Hardware] = "A100",
    *,
    kv_cache_len: int = 0,
    strict: bool = False,
) -> RooflineReport:
    """Roofline analysis for ONE forward pass at ``(query_len, kv_cache_len)``.

    Models the cost of processing ``query_len`` new tokens given a KV cache of
    ``kv_cache_len`` already-cached tokens. ``kv_cache_len=0`` is cold prefill.

    Args:
        config: An ``AlloyConfig`` (or anything with ``layer_types``,
            ``ffn_types``, ``hidden_size``, ``vocab_size``).
        batch: Batch size.
        query_len: Number of new tokens this forward pass.
        dtype: Compute dtype. Affects bytes (weights + activations); FLOPs
            are dtype-independent.
        hardware: Preset name (``"A100"`` / ``"H100"`` / ``"Ascend910B1"`` /
            ``"Ascend910C"``) or a :class:`Hardware` instance.
        kv_cache_len: Tokens already in KV cache from previous forwards.
            Default 0 (cold prefill).
        strict: If True, raise on any unknown mixer/ffn name. Default False —
            warn, contribute zero, continue.

    Returns:
        :class:`RooflineReport` with per-module breakdown and aggregate
        metrics. Mode label (``prefill`` / ``mini-prefill`` / ``decode``)
        is inferred from ``(query_len, kv_cache_len)`` for the report header.
    """
    hw = get_hardware(hardware)
    report = RooflineReport(
        config=config, batch=batch, query_len=query_len, kv_cache_len=kv_cache_len,
        dtype=dtype, hardware=hw,
    )

    es = dtype_size(dtype)
    hidden = config.hidden_size
    vocab = config.vocab_size
    n_query_tokens = batch * query_len
    spec_kwargs = {"kv_cache_len": kv_cache_len}

    # ---- Embedding: lookup of B*Q rows from [vocab, hidden] table. -------
    # Only NEW tokens are embedded each forward; cached tokens were embedded
    # in their own forward and persist in the residual stream of the cache.
    report.modules.append(ModuleStat(
        kind="embedding", name="embed_tokens", layer_idx=None,
        flops=0, bytes=2 * n_query_tokens * hidden * es,
    ))

    # ---- Decoder layers: each (mixer, ffn) pair, dispatched by name. -----
    # Shape between layers is [B, query_len, hidden_size]; the cache state is
    # passed via spec_kwargs (kv_cache_len), not via shape.
    in_shape: tuple[int, ...] = (batch, query_len, hidden)
    layer_types = list(config.layer_types)
    ffn_types = list(config.ffn_types)
    if len(layer_types) != len(ffn_types):
        raise ValueError(
            f"config.layer_types ({len(layer_types)}) and ffn_types "
            f"({len(ffn_types)}) length mismatch."
        )

    for i, (mixer_name, ffn_name) in enumerate(zip(layer_types, ffn_types)):
        # Mixer
        spec = get_spec(mixer_name, strict=strict)
        if spec is None:
            report.modules.append(ModuleStat(
                kind="unknown", name=mixer_name, layer_idx=i, flops=0, bytes=0,
            ))
        else:
            report.modules.append(ModuleStat(
                kind="mixer", name=mixer_name, layer_idx=i,
                flops=spec.flops(in_shape, config, **spec_kwargs),
                bytes=spec.bytes(in_shape, config, dtype, **spec_kwargs),
            ))
            in_shape = spec.out_shape(in_shape, config, **spec_kwargs)

        # FFN
        spec = get_spec(ffn_name, strict=strict)
        if spec is None:
            report.modules.append(ModuleStat(
                kind="unknown", name=ffn_name, layer_idx=i, flops=0, bytes=0,
            ))
        else:
            report.modules.append(ModuleStat(
                kind="ffn", name=ffn_name, layer_idx=i,
                flops=spec.flops(in_shape, config, **spec_kwargs),
                bytes=spec.bytes(in_shape, config, dtype, **spec_kwargs),
            ))
            in_shape = spec.out_shape(in_shape, config, **spec_kwargs)

    # ---- Final RMSNorm: applied to the Q new tokens before lm_head. ------
    norm_spec = RMSNormSpec()
    final_norm_shape = (batch, query_len, hidden)
    report.modules.append(ModuleStat(
        kind="norm", name="final_norm", layer_idx=None,
        flops=norm_spec.flops(final_norm_shape),
        bytes=norm_spec.bytes(final_norm_shape, dtype=dtype),
    ))

    # ---- LM head: linear [B,Q,hidden] -> [B,Q,vocab]. --------------------
    # Tied embeddings: the weight matrix is shared with embed_tokens but
    # still must be read from HBM for this matmul (HBM bandwidth doesn't
    # benefit from weight sharing — only memory storage does).
    lm_head_flops = 2 * n_query_tokens * hidden * vocab
    lm_head_bytes = (hidden * vocab + n_query_tokens * hidden + n_query_tokens * vocab) * es
    report.modules.append(ModuleStat(
        kind="lm_head", name="lm_head", layer_idx=None,
        flops=lm_head_flops, bytes=lm_head_bytes,
    ))

    return report


# --------------------------------------------------------------------------- #
# Convenience wrappers — common LLM-serving modes
# --------------------------------------------------------------------------- #


def roofline_prefill(
    config,
    batch: int,
    seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
    hardware: Union[str, Hardware] = "A100",
    *,
    strict: bool = False,
) -> RooflineReport:
    """Cold prefill: process ``seq_len`` tokens with no existing KV cache."""
    return roofline(
        config, batch=batch, query_len=seq_len, kv_cache_len=0,
        dtype=dtype, hardware=hardware, strict=strict,
    )


def roofline_decode(
    config,
    batch: int,
    kv_cache_len: int,
    dtype: torch.dtype = torch.bfloat16,
    hardware: Union[str, Hardware] = "A100",
    *,
    strict: bool = False,
) -> RooflineReport:
    """Single-token decode given ``kv_cache_len`` tokens already cached."""
    return roofline(
        config, batch=batch, query_len=1, kv_cache_len=kv_cache_len,
        dtype=dtype, hardware=hardware, strict=strict,
    )


def roofline_mini_prefill(
    config,
    batch: int,
    chunk_len: int,
    kv_cache_len: int,
    dtype: torch.dtype = torch.bfloat16,
    hardware: Union[str, Hardware] = "A100",
    *,
    strict: bool = False,
) -> RooflineReport:
    """One chunk of chunked prefill: ``chunk_len`` new tokens with
    ``kv_cache_len`` tokens already cached from earlier chunks."""
    return roofline(
        config, batch=batch, query_len=chunk_len, kv_cache_len=kv_cache_len,
        dtype=dtype, hardware=hardware, strict=strict,
    )


__all__ = [
    "ModuleStat",
    "RooflineReport",
    "roofline",
    "roofline_prefill",
    "roofline_decode",
    "roofline_mini_prefill",
]
