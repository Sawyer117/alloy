"""Count parameters in an alloy model from its config.

Builds the model on the meta device (no real allocation) and walks the
module tree to break the parameter total down by component:

  - ``embed_tokens`` / ``lm_head`` (lm_head reported as 0 when tied)
  - per-layer mixer + FFN totals, grouped by registry key
    (``qwen3_attention``, ``qwen3_5_gdn``, ``qwen3_mlp``, ``qwen3_5_moe``, ...)
  - the ``input_layernorm`` + ``post_attention_layernorm`` of every layer,
    plus the final ``model.norm``

The breakdown is the actual deliverable; the total is just ``sum(numel())``.
Use this to sanity-check a new ``examples/configs/*.json`` before training,
or to verify that two configs differ where you expect them to.

Python API
----------
::

    from alloy.tools.count_params import count_params

    counts = count_params("alloy/examples/configs/qwen3_340m_dense.json")
    print(counts.total)              # int
    print(counts.format_table())     # human-readable breakdown

Accepts: path to a config JSON, path to a directory containing
``config.json``, a config-dict, or an ``AlloyConfig`` instance.

CLI
---
::

    python -m alloy.tools.count_params alloy/examples/configs/qwen3_340m_dense.json
    python -m alloy.tools.count_params ./hf_models/alloy_qwen3_next_340m
    python -m alloy.tools.count_params --expect 340M --tolerance 5 <config>

``--expect`` accepts a number with optional ``K`` / ``M`` / ``B`` suffix
(e.g. ``340M``, ``35B``); the run exits non-zero if the actual total
deviates by more than ``--tolerance`` percent.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from alloy.configuration_alloy import AlloyConfig


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@dataclass
class _TypeBucket:
    count: int = 0       # number of layers of this type
    params: int = 0      # total params across those layers


@dataclass
class ParamCount:
    """Result of :func:`count_params`. ``total`` is what most callers want;
    the per-component fields exist so a reader can verify *where* the
    parameters live (catches things like a typo'd ``intermediate_size`` that
    silently halves the FFN block).
    """

    total: int
    embed_tokens: int
    lm_head: int                                    # 0 when tied to embed_tokens
    tied_embeddings: bool
    layer_norms: int                                # input + post-attn norms summed across layers
    final_norm: int                                 # model.norm
    mixers: dict[str, _TypeBucket] = field(default_factory=dict)
    ffns: dict[str, _TypeBucket] = field(default_factory=dict)
    other: int = 0                                  # uncounted leftover (sanity check; should be ~0)

    def format_table(self) -> str:
        """Render a fixed-width breakdown for terminal output."""
        lines: list[str] = []
        lines.append(f"Total parameters:  {_fmt_count(self.total)}")
        lines.append("")
        lines.append(f"  embed_tokens     {_fmt_count(self.embed_tokens)}")
        if self.tied_embeddings:
            lines.append(f"  lm_head          (tied to embed_tokens)")
        else:
            lines.append(f"  lm_head          {_fmt_count(self.lm_head)}")
        lines.append(f"  layer norms      {_fmt_count(self.layer_norms)}")
        lines.append(f"  final norm       {_fmt_count(self.final_norm)}")
        if self.mixers:
            lines.append("")
            lines.append("  mixers:")
            for kind, bucket in sorted(self.mixers.items()):
                lines.append(
                    f"    {kind:<26} x{bucket.count:<3}  {_fmt_count(bucket.params)}"
                )
        if self.ffns:
            lines.append("")
            lines.append("  FFNs:")
            for kind, bucket in sorted(self.ffns.items()):
                lines.append(
                    f"    {kind:<26} x{bucket.count:<3}  {_fmt_count(bucket.params)}"
                )
        if self.other:
            lines.append("")
            lines.append(f"  uncounted        {_fmt_count(self.other)}  "
                         f"(should be ~0; if not, count_params is missing a component)")
        return "\n".join(lines)


def count_params(source: str | Path | dict | AlloyConfig) -> ParamCount:
    """Count parameters of an alloy model defined by ``source``.

    Parameters
    ----------
    source
        - ``str`` / ``Path``: either a JSON config file, or a directory
          containing ``config.json``.
        - ``dict``: a config dict (as parsed from JSON).
        - ``AlloyConfig``: used directly.

    Returns
    -------
    ParamCount
        Total + per-component breakdown. See class docstring.

    Notes
    -----
    The model is constructed on the ``meta`` device — parameters have shape
    metadata but no storage. Cost is roughly the cost of running
    ``__init__`` on every nn.Module; it does not allocate weights.
    """
    cfg = _coerce_config(source)

    # Lazy import keeps the (heavy) modeling import out of users that only
    # want to read config json.
    from alloy.modeling_alloy import AlloyForCausalLM

    with torch.device("meta"):
        model = AlloyForCausalLM(cfg)

    total = sum(p.numel() for p in model.parameters())

    embed_w = model.model.embed_tokens.weight
    lm_head_w = model.lm_head.weight
    tied = lm_head_w is embed_w

    embed_tokens = embed_w.numel()
    lm_head = 0 if tied else lm_head_w.numel()

    final_norm = sum(p.numel() for p in model.model.norm.parameters())

    mixers: dict[str, _TypeBucket] = defaultdict(_TypeBucket)
    ffns: dict[str, _TypeBucket] = defaultdict(_TypeBucket)
    layer_norms_total = 0

    for layer in model.model.layers:
        mixers[layer.layer_type].count += 1
        mixers[layer.layer_type].params += sum(p.numel() for p in layer.mixer.parameters())

        ffns[layer.ffn_type].count += 1
        ffns[layer.ffn_type].params += sum(p.numel() for p in layer.mlp.parameters())

        layer_norms_total += sum(p.numel() for p in layer.input_layernorm.parameters())
        layer_norms_total += sum(p.numel() for p in layer.post_attention_layernorm.parameters())

    counted = (
        embed_tokens
        + lm_head
        + final_norm
        + layer_norms_total
        + sum(b.params for b in mixers.values())
        + sum(b.params for b in ffns.values())
    )
    other = total - counted

    return ParamCount(
        total=total,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        tied_embeddings=tied,
        layer_norms=layer_norms_total,
        final_norm=final_norm,
        mixers=dict(mixers),
        ffns=dict(ffns),
        other=other,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _coerce_config(source: str | Path | dict | AlloyConfig) -> AlloyConfig:
    if isinstance(source, AlloyConfig):
        return source
    if isinstance(source, dict):
        cfg_dict = dict(source)
        cfg_dict.pop("model_type", None)
        cfg_dict.pop("transformers_version", None)
        return AlloyConfig(**cfg_dict)

    path = Path(source)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("model_type", None)
    cfg_dict.pop("transformers_version", None)
    return AlloyConfig(**cfg_dict)


def _fmt_count(n: int) -> str:
    """Format a parameter count as a human-readable string with both raw and
    suffix forms, e.g. ``327,544,832 (327.54M)``. Aligned-padded for tables.
    """
    if n == 0:
        return f"{0:>15,}"
    abs_n = abs(n)
    if abs_n >= 1e9:
        suffix = f"{n / 1e9:.2f}B"
    elif abs_n >= 1e6:
        suffix = f"{n / 1e6:.2f}M"
    elif abs_n >= 1e3:
        suffix = f"{n / 1e3:.2f}K"
    else:
        suffix = f"{n}"
    return f"{n:>15,} ({suffix})"


_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9}


def _parse_count(s: str) -> int:
    """Parse a count string like ``340M`` / ``35B`` / ``1234`` into an int."""
    s = s.strip().replace(",", "").replace("_", "")
    if not s:
        raise ValueError("empty count string")
    suffix = s[-1].upper()
    if suffix in _SUFFIXES:
        return int(float(s[:-1]) * _SUFFIXES[suffix])
    return int(s)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source",
        help="Path to a JSON config, or a directory containing config.json.",
    )
    parser.add_argument(
        "--expect",
        default=None,
        help="Expected total parameter count (e.g. 340M, 35B, 327544832). "
             "If set, exit non-zero when the actual count is outside "
             "--tolerance percent of this value.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Percent tolerance for --expect (default: 5.0). Ignored without --expect.",
    )
    args = parser.parse_args()

    counts = count_params(args.source)
    print(f"Source: {args.source}")
    print()
    print(counts.format_table())

    if args.expect is not None:
        expected = _parse_count(args.expect)
        delta = counts.total - expected
        pct = abs(delta) / max(expected, 1) * 100
        print()
        print(f"Expected:          {_fmt_count(expected)}")
        print(f"Delta:             {_fmt_count(delta)}  ({pct:+.2f}% of expected)")
        if pct > args.tolerance:
            print(f"FAIL — deviation {pct:.2f}% exceeds tolerance {args.tolerance:.2f}%",
                  file=sys.stderr)
            return 1
        print(f"OK  — within tolerance {args.tolerance:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
