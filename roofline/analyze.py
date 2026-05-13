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
from .specs_mhc import MhcHyperConnectionSpec, MhcHyperHeadSpec


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
    def tokens_per_sec(self) -> float:
        """Throughput: query tokens processed this forward / forward time."""
        n = self.batch * self.query_len
        return n / self.roofline_time_s if self.roofline_time_s > 0 else 0.0

    @property
    def time_per_token_s(self) -> float:
        """Inverse of :attr:`tokens_per_sec` — forward time per token."""
        n = self.batch * self.query_len
        return self.roofline_time_s / n if n > 0 else 0.0

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

    # ----- Per-module helpers (used by Level 2 / 3 formatters) ------------- #

    def _module_time_s(self, m: "ModuleStat") -> float:
        """Theoretical time for one module on this report's hardware/dtype:
        ``max(m.flops/peak_flops, m.bytes/hbm_bandwidth)``. Sum over modules
        is the no-cross-module-overlap pessimistic bound; the global
        :attr:`roofline_time_s` is the with-overlap optimistic bound."""
        peak = self.hardware.peak_flops.get(self.dtype, 0.0)
        bw = self.hardware.hbm_bandwidth
        if peak <= 0 or bw <= 0:
            return 0.0
        return max(m.flops / peak, m.bytes / bw)

    def _module_bound(self, m: "ModuleStat") -> str:
        """Per-module bottleneck letter: 'C' (compute), 'M' (memory), '?'.
        Determined by comparing m.flops/peak vs m.bytes/bw on this hardware.
        """
        peak = self.hardware.peak_flops.get(self.dtype, 0.0)
        bw = self.hardware.hbm_bandwidth
        if peak <= 0 or bw <= 0:
            return "?"
        return "C" if m.flops / peak >= m.bytes / bw else "M"

    # ----- Three-level formatters ------------------------------------------ #

    def format(self, level: int = 2) -> str:
        """Format the report at one of three verbosity levels.

        * level=1 — single-line summary (totals + tokens/sec + bottleneck)
        * level=2 — full per-module table with bound and %time columns
        * level=3 — same totals as level=2, but rows aggregated by
          ``(kind, name)`` so e.g. all 58 ``ffn:dsv4_moe`` layers collapse
          into one row showing combined contribution
        """
        if level == 1:
            return self._format_level1()
        if level == 2:
            return self._format_level2()
        if level == 3:
            return self._format_level3()
        raise ValueError(f"level must be 1, 2, or 3; got {level}")

    def _header_line(self) -> str:
        return f"Roofline | {self.hardware.name} | {_dtype_short(self.dtype)} | {self._mode_label()}"

    def _footer_lines(self) -> list[str]:
        return [
            f"TOTAL: {_scale_flops(self.total_flops)} / "
            f"{_scale_bytes(self.total_bytes)} / "
            f"AI = {self.arithmetic_intensity:.1f}",
            f"{self.hardware.name}: "
            f"compute={_scale_time(self.compute_time_s)}, "
            f"memory={_scale_time(self.memory_time_s)} "
            f"-> bottleneck={self.bottleneck} "
            f"({_scale_time(self.roofline_time_s)} / forward, "
            f"{self.tokens_per_sec:,.0f} tok/s)",
        ]

    def _format_level1(self) -> str:
        return (
            f"{self.hardware.name} | {_dtype_short(self.dtype)} | {self._mode_label()} | "
            f"{_scale_flops(self.total_flops)} / {_scale_bytes(self.total_bytes)} / "
            f"AI={self.arithmetic_intensity:.1f} -> "
            f"{_scale_time(self.roofline_time_s)} ({self.bottleneck}) "
            f"{self.tokens_per_sec:,.0f} tok/s"
        )

    def _format_level2(self) -> str:
        sep = "-" * 110
        per_module_times = [self._module_time_s(m) for m in self.modules]
        sum_t = sum(per_module_times) or 1.0  # avoid /0 in pct

        header = "  ".join([
            f"{'idx':>4}", f"{'kind':<10}", f"{'name':<32}",
            f"{'flops':>14}", f"{'bytes':>12}", f"{'AI':>6}",
            f"{'b':>1}", f"{'time':>10}", f"{'%t':>5}",
        ])
        lines = [self._header_line(), sep, header, sep]
        for m, t in zip(self.modules, per_module_times):
            idx = "-" if m.layer_idx is None else str(m.layer_idx)
            lines.append("  ".join([
                f"{idx:>4}",
                f"{m.kind:<10}",
                f"{m.name:<32}",
                f"{_scale_flops(m.flops):>14}",
                f"{_scale_bytes(m.bytes):>12}",
                f"{m.arithmetic_intensity:>6.1f}",
                f"{self._module_bound(m):>1}",
                f"{_scale_time(t):>10}",
                f"{(t / sum_t * 100):>5.1f}",
            ]))
        lines.append(sep)
        lines.extend(self._footer_lines())
        lines.append("b: C=compute-bound, M=memory-bound  "
                     "(%t = per-module time / sum of per-module times)")
        return "\n".join(lines)

    def _format_level3(self) -> str:
        # Aggregate by (kind, name). For top-level kinds where kind == purpose
        # (embedding/norm/lm_head/unknown) display "name" alone; for layer
        # kinds (mixer/ffn) prefix the kind so e.g. mixer:dsv4_moe is
        # distinguishable from ffn:dsv4_moe (would never collide today, but
        # keeps the convention readable when scanning).
        from collections import OrderedDict

        groups: "OrderedDict[str, dict]" = OrderedDict()
        for m, t in zip(self.modules, [self._module_time_s(m) for m in self.modules]):
            key = f"{m.kind}:{m.name}" if m.kind in ("mixer", "ffn", "mhc") else m.name
            g = groups.setdefault(key, {"count": 0, "flops": 0, "bytes": 0, "time": 0.0,
                                       "compute_bound_count": 0, "memory_bound_count": 0})
            g["count"] += 1
            g["flops"] += m.flops
            g["bytes"] += m.bytes
            g["time"] += t
            if self._module_bound(m) == "C":
                g["compute_bound_count"] += 1
            elif self._module_bound(m) == "M":
                g["memory_bound_count"] += 1

        sum_t = sum(g["time"] for g in groups.values()) or 1.0

        sep = "-" * 110
        header = "  ".join([
            f"{'component':<32}", f"{'count':>5}",
            f"{'flops':>14}", f"{'bytes':>12}", f"{'AI':>6}",
            f"{'b':>1}", f"{'time':>10}", f"{'%t':>5}",
        ])
        lines = [self._header_line(), sep, header, sep]
        for name, g in groups.items():
            ai = (g["flops"] / g["bytes"]) if g["bytes"] > 0 else 0.0
            # Group bound: majority across members (almost always uniform for
            # same-named modules; '?' when neither dominates).
            cb, mb = g["compute_bound_count"], g["memory_bound_count"]
            bnd = "C" if cb > mb else ("M" if mb > cb else "?")
            lines.append("  ".join([
                f"{name:<32}",
                f"{g['count']:>5}",
                f"{_scale_flops(g['flops']):>14}",
                f"{_scale_bytes(g['bytes']):>12}",
                f"{ai:>6.1f}",
                f"{bnd:>1}",
                f"{_scale_time(g['time']):>10}",
                f"{(g['time'] / sum_t * 100):>5.1f}",
            ]))
        lines.append(sep)
        lines.extend(self._footer_lines())
        lines.append("b: C=compute-bound, M=memory-bound  "
                     "(%t = group time / sum of all group times)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format(level=2)


def format_comparison(
    reports: "list[RooflineReport]",
    title: Optional[str] = None,
) -> str:
    """Compact cross-hardware comparison block — Level-1 view across devices.

    All ``reports`` should describe the same case (same flops/bytes/AI);
    only ``hardware`` should differ. Output: a shared metrics line plus one
    row per hardware showing time, throughput, and bottleneck.
    """
    if not reports:
        return ""
    rep0 = reports[0]
    lines: list[str] = []
    if title:
        lines.append(f"--- {title} ---")
    lines.append(
        f"  FLOPs={_scale_flops(rep0.total_flops)}  "
        f"bytes={_scale_bytes(rep0.total_bytes)}  "
        f"AI={rep0.arithmetic_intensity:.1f}"
    )
    name_w = max(len(r.hardware.name) for r in reports)
    for r in reports:
        lines.append(
            f"  {r.hardware.name:<{name_w}}  "
            f"{_scale_time(r.roofline_time_s):>10}  "
            f"{r.tokens_per_sec:>9,.0f} tok/s  "
            f"({r.bottleneck:7} bound)"
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

    # MHC residual machinery: when ``config.use_mhc=True``, every decoder
    # layer wraps both sublayers in a ``_HyperConnection`` (collapse hc_mult
    # streams to one, run sublayer, place back). The sublayer itself still
    # sees [B, S, hidden] — that's why ``in_shape`` here stays single-stream.
    # Cost is added per-layer as kind="mhc"; the final HyperHead collapse
    # is added once after the loop.
    use_mhc = bool(getattr(config, "use_mhc", False))
    hc_spec = MhcHyperConnectionSpec() if use_mhc else None

    for i, (mixer_name, ffn_name) in enumerate(zip(layer_types, ffn_types)):
        # MHC: per-layer pre-mixer HyperConnection
        if hc_spec is not None:
            report.modules.append(ModuleStat(
                kind="mhc", name="hc_attn", layer_idx=i,
                flops=hc_spec.flops(in_shape, config),
                bytes=hc_spec.bytes(in_shape, config, dtype),
            ))

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

        # MHC: per-layer pre-ffn HyperConnection
        if hc_spec is not None:
            report.modules.append(ModuleStat(
                kind="mhc", name="hc_ffn", layer_idx=i,
                flops=hc_spec.flops(in_shape, config),
                bytes=hc_spec.bytes(in_shape, config, dtype),
            ))

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

    # MHC: final HyperHead collapse — single instance, NOT per-layer.
    if use_mhc:
        head_spec = MhcHyperHeadSpec()
        head_in_shape = (batch, query_len, hidden)
        report.modules.append(ModuleStat(
            kind="mhc", name="hc_head", layer_idx=None,
            flops=head_spec.flops(head_in_shape, config),
            bytes=head_spec.bytes(head_in_shape, config, dtype),
        ))

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
    "format_comparison",
    "roofline",
    "roofline_prefill",
    "roofline_decode",
    "roofline_mini_prefill",
]
