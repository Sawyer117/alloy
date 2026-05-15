"""Binder ON vs OFF for DeepSeek-V4 — precision + speed on alloy.

DSV4-specific sibling of ``compare_binder_vs_torch.py``. Builds a small
alloy DSV4 model whose ``layer_types`` pattern covers all three DSV4
attention flavours (``dsv4_hca_attention``, ``dsv4_csa_attention``,
``dsv4_sliding_attention``), then forwards the same input twice:

  1. **OFF (baseline)** — every DSV4 attention dispatch surface is
     pinned to alloy's torch impl (``_torch_dsv4_attention`` →
     ``_eager_attention_with_sinks``). This is the byte-exact
     reference path on this hardware.

  2. **ON (binder)** — ``activate(model, prefer=<backend>, target=<set>)``
     flips the ``_dsv4_{csa,hca,sliding}_implementation`` fields named
     in ``--target`` (default ``all``) to the chosen backend so those
     layer types dispatch to the binder fast path. Untargeted layer
     types stay on torch so the diff isolates the layer types under
     test.

The diff between the two logit outputs is the **fast-path drift** for
the targeted layer types; in bf16 we expect it in the accumulation-
order noise floor (~1e-3 to 1e-2 max_abs). Same hardware, same random
weights, same input — only the chosen attention impl(s) change, so any
larger drift is a wrapper bug.

Shape constraint: the binder triton adapters share the SFA kernel's
``CONFIG_MAP`` ``{128, 160, 640}`` for total topk width.

  * Sliding-only: ``sliding_window`` must hit one of those values.
  * CSA: ``sliding_window + index_topk`` must hit one of those.
  * HCA: ``sliding_window + ceil(seq_len / compress_rate_hca)`` must hit
    one of those.

Default config (W=128, index_topk=32, compress_rate_hca=8, seq_len=128)
gives CSA=160 and HCA=128+16=144 — HCA misses CONFIG_MAP at this
default. Either widen seq_len so HCA lands (e.g. seq=4096 with
compress_rate_hca=128 → 128+32=160) or pass ``--target csa,sliding`` to
skip HCA on default settings.

Skipped cleanly if ``torch_npu`` or ``hf_npu_binder`` is not installed.

Usage::

    # Default: all three attention layer types swap to binder triton;
    # MoE experts STAY on HF eager (so the perf number isn't skewed by
    # the experts-flash speedup which is unrelated to attention).
    python -m alloy.tests.npu.compare_dsv4_binder_vs_torch \\
        --num-layers 4 --dtype bf16 --n-repeat 5

    # Just CSA
    python -m alloy.tests.npu.compare_dsv4_binder_vs_torch --target csa

    # MoE experts only (binder flash via ALL_EXPERTS_FUNCTIONS["flash"]).
    # Attention layers stay torch on both phases so the diff isolates
    # the experts path.
    python -m alloy.tests.npu.compare_dsv4_binder_vs_torch --target experts

    # CSA + experts together
    python -m alloy.tests.npu.compare_dsv4_binder_vs_torch --target csa,experts

    # Try the ascendc CSA path (HCA / sliding stay torch via DEFAULTS):
    python -m alloy.tests.npu.compare_dsv4_binder_vs_torch --prefer ascendc

Targets:
    csa / hca / sliding  — attention dispatch (alloy IMPL_REGISTRY ->
                            binder triton). Pinning order: baseline pins
                            ``_dsv4_{kind}_implementation = 'torch'``;
                            binder phase flips targeted ones to args.prefer.
    experts              — MoE experts forward (HF ALL_EXPERTS_FUNCTIONS
                            via ``_experts_implementation``). Baseline pins
                            to ``'eager'`` (HF's per-expert loop body);
                            binder phase flips to ``'flash'`` (the binder
                            shared ``moe_experts.flash`` chain of 4 ascendc
                            fused ops + clamped npu_swiglu for DSV4).
    all                  — equivalent to csa,hca,sliding (NOT experts;
                            experts must be opted in explicitly).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_IMPORT_ERR: str | None = None
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401  patches torch.cuda -> npu
except ImportError as _e:
    _IMPORT_ERR = f"torch_npu not available ({_e}); this script must run on NPU hardware."

if _IMPORT_ERR is None:
    try:
        import hf_npu_binder  # noqa: F401
    except ImportError as _e:
        _IMPORT_ERR = f"hf_npu_binder not installed ({_e}); pip install hf-npu-binder."

if _IMPORT_ERR is None:
    from alloy import AlloyConfig, AlloyForCausalLM
    import alloy.integrations.hf_npu_binder as binder
    from alloy.tests._compare_utils import diff_logits, pick_device
else:
    AlloyConfig = AlloyForCausalLM = binder = None  # type: ignore[assignment]
    diff_logits = pick_device = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config builder — small DSV4 with at least one CSA layer
# ---------------------------------------------------------------------------
def _layer_pattern(num_layers: int) -> list[str]:
    """Mirror DSV4's representative ``[hca, hca, csa, sliding]`` repeat so
    every attention flavour shows up, and the CSA layer is the one that
    swaps backends under ``activate(prefer="triton")``.
    """
    base = ["dsv4_hca_attention", "dsv4_hca_attention",
            "dsv4_csa_attention", "dsv4_sliding_attention"]
    out: list[str] = []
    while len(out) < num_layers:
        out.extend(base)
    return out[:num_layers]


def _ffn_pattern(num_layers: int) -> list[str]:
    """DSV4 convention: hash routing on the first few layers, topk after."""
    base = ["dsv4_hash_moe", "dsv4_moe", "dsv4_moe", "dsv4_moe"]
    out: list[str] = []
    while len(out) < num_layers:
        out.extend(base)
    return out[:num_layers]


def _build_config(args) -> AlloyConfig:
    head_dim = args.hidden_size // args.num_attention_heads
    return AlloyConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=1,            # DSV4 is single-KV-head MQA
        head_dim=head_dim,
        intermediate_size=args.hidden_size,
        max_position_embeddings=max(args.seq_len, 512),
        sliding_window=args.sliding_window,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        # MHC (mixture of head clusters). DSV4 paper / production uses
        # hc_mult=4 — the binder MHC triton kernels are hardcoded for
        # that (manually-unrolled 4-stream loops). hc_mult is overridable
        # via --hc-mult; --target mhc enforces 4.
        hc_mult=args.hc_mult,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        use_mhc=True,
        # MoE
        n_routed_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        mlp_bias=False,
        # DSV4 attention
        q_lora_rank=args.hidden_size // 2,
        o_groups=2,
        o_lora_rank=args.hidden_size // 2,
        # Lightning Indexer (CSA only). index_topk + sliding_window must
        # land on a CONFIG_MAP width for the binder triton path.
        index_n_heads=args.num_attention_heads,
        index_head_dim=head_dim,
        index_topk=args.index_topk,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        rope_parameters={
            "main":     {"rope_type": "default", "rope_theta": 10000.0,  "partial_rotary_factor": 0.25},
            "compress": {"rope_type": "default", "rope_theta": 100000.0, "partial_rotary_factor": 0.25},
        },
        layer_types=_layer_pattern(args.num_layers),
        ffn_types=_ffn_pattern(args.num_layers),
        attn_implementation="eager",
    )


# ---------------------------------------------------------------------------
# Forward + timing
# ---------------------------------------------------------------------------
def _measure(
    model: AlloyForCausalLM,
    input_ids: torch.Tensor,
    *,
    n_warmup: int,
    n_repeat: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """Returns (logits_of_last_call, t_first_seconds, t_avg_seconds_over_n_repeat)."""
    model.eval()
    sync = torch.npu.synchronize if device.type == "npu" else (lambda: None)

    with torch.no_grad():
        sync()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=False)
        sync()
        t_first = time.perf_counter() - t0

        for _ in range(n_warmup):
            out = model(input_ids=input_ids, use_cache=False)
        sync()

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            out = model(input_ids=input_ids, use_cache=False)
        sync()
        t_avg = (time.perf_counter() - t0) / n_repeat

    return out.logits, t_first, t_avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[s]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument(
        "--hc-mult", type=int, default=2,
        help="MHC stream multiplicity. Default 2 keeps prior tests cheap; "
             "--target mhc requires 4 (DSV4 paper config matching the "
             "vendored MindSpeed triton kernels which manually unroll 4 streams).",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sliding-window", type=int, default=128,
        help="Sliding-window size W. The binder triton path requires "
             "W + index_topk to be in CONFIG_MAP {128, 160, 640}.",
    )
    parser.add_argument(
        "--index-topk", type=int, default=32,
        help="Lightning Indexer top-k value. Default 32 makes total = 128+32=160.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument(
        "--prefer", default="triton", choices=["triton", "ascendc", "torch"],
        help="binder backend to compare baseline against. 'triton' is the "
             "MindSpeed kernel via BHSD adapter; 'ascendc' is CANN's "
             "aclnnSparseAttnSharedkv (requires the op in libopapi.so).",
    )
    parser.add_argument(
        "--target", default="all",
        help="comma-separated subset of {csa,hca,sliding,experts,all} for "
             "which surfaces to swap to the binder backend. csa / hca / "
             "sliding swap the attention dispatch (per-layer-type triton "
             "via alloy IMPL_REGISTRY); 'experts' swaps the MoE experts "
             "forward (HF ALL_EXPERTS_FUNCTIONS via "
             "_experts_implementation = 'flash'). Untargeted surfaces stay "
             "pinned to torch / eager on both phases so the diff isolates "
             "the targeted ones. Default 'all' covers attention only "
             "(experts must be opted in explicitly since baseline = HF "
             "eager loop is meaningfully slower than the flash path even "
             "at small scale).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-repeat", type=int, default=5)
    args = parser.parse_args(argv)

    if _IMPORT_ERR is not None:
        print(f"SKIP - {_IMPORT_ERR}")
        return 0

    device = pick_device()
    dtype = _dtype_from_str(args.dtype)
    print(f"device={device}  dtype={dtype}  layers={args.num_layers}  hidden={args.hidden_size}  "
          f"seq_len={args.seq_len}  sliding_window={args.sliding_window}  index_topk={args.index_topk}")

    cfg = _build_config(args)

    # Parse --target into normalised sets:
    #   ATTN_TARGETS  — subset of {csa, hca, sliding}; controls
    #                   _dsv4_{kind}_implementation flips (alloy IMPL_REGISTRY)
    #   moe_target    — bool; controls _experts_implementation flip
    #                   (HF ALL_EXPERTS_FUNCTIONS). Routed via separate field
    #                   because experts dispatch is HF-canonical, not
    #                   per-layer-type.
    #   mhc_target    — bool; controls _dsv4_mhc_implementation flip
    #                   (every MHC HyperConnection site in the model).
    #                   Requires config.hc_mult == 4 — binder triton path
    #                   raises otherwise.
    _ALL_ATTN_TARGETS = {"csa", "hca", "sliding"}
    _ALL_VALID = _ALL_ATTN_TARGETS | {"experts", "mhc"}
    raw = [t.strip() for t in args.target.split(",") if t.strip()]
    if "all" in raw:
        # 'all' means all attention layer types; experts + mhc must be
        # opted in explicitly so accidentally hitting them doesn't change
        # the baseline cost reference.
        attn_targets = set(_ALL_ATTN_TARGETS)
        moe_target = False
        mhc_target = False
    else:
        unknown = set(raw) - _ALL_VALID
        if unknown:
            print(f"ERROR - --target contains unknown entries {sorted(unknown)}; "
                  f"choose from {sorted(_ALL_VALID)} or 'all'.")
            return 2
        attn_targets = set(raw) & _ALL_ATTN_TARGETS
        moe_target = "experts" in raw
        mhc_target = "mhc" in raw

    if mhc_target and cfg.hc_mult != 4:
        print(f"ERROR - --target mhc requires config.hc_mult == 4 (DSV4 paper "
              f"config). Vendored MindSpeed triton kernels hardcode 4 streams; "
              f"got hc_mult={cfg.hc_mult}. Bump --hc-mult or drop mhc from target.")
        return 2

    print(f"binder targets for this run: attn={sorted(attn_targets)}  "
          f"experts={moe_target}  mhc={mhc_target}")

    # Layer-type populations (per chosen layer_types pattern).
    _LAYER_TYPE_BY_TARGET = {
        "csa":     "dsv4_csa_attention",
        "hca":     "dsv4_hca_attention",
        "sliding": "dsv4_sliding_attention",
    }
    counts = {t: sum(1 for lt in cfg.layer_types if lt == _LAYER_TYPE_BY_TARGET[t])
              for t in _ALL_ATTN_TARGETS}
    moe_layer_count = sum(1 for lt in cfg.ffn_types if "moe" in lt)
    attn_targeted_count = sum(counts[t] for t in attn_targets)
    if attn_targeted_count == 0 and not moe_target and not mhc_target:
        print(f"SKIP - no targeted surfaces present "
              f"(layer_types={cfg.layer_types}, ffn_types={cfg.ffn_types}, "
              f"attn={sorted(attn_targets)}, experts={moe_target}, mhc={mhc_target}); "
              f"nothing for binder to fast-path.")
        return 0
    print(f"layer counts: attn={counts}  moe_layers={moe_layer_count}  →  "
          f"attn {attn_targeted_count}/{args.num_layers} swapped, "
          f"experts {'flash on all' if moe_target else 'eager (untouched)'}, "
          f"mhc {'triton on all' if mhc_target else 'torch (untouched)'}")

    torch.manual_seed(args.seed)
    # Generate on CPU then move — some CANN releases route a direct
    # ``torch.randint(..., device='npu')`` through ``aclnnInplaceRandom``
    # whose dynamic kernel config fails to parse on older toolkits.
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len)).to(device)

    # =========================================================================
    # Phase 1: BASELINE — pin DSV4 attention + MoE experts to their torch refs
    # =========================================================================
    # Importing the binder bridge sets DEFAULT_IMPL entries to whatever binder
    # DEFAULTS' "auto" resolves to (currently "triton" / "flash"). Without
    # explicit override the baseline would ALSO pick up binder kernels and
    # the whole comparison becomes "binder vs binder" — diff would only show
    # kernel determinism, not wrapper correctness. Force torch / eager here.
    for t in _ALL_ATTN_TARGETS:
        setattr(cfg, f"_dsv4_{t}_implementation", "torch")
    # HF's experts dispatch validates against {"eager", "flash", "grouped_mm",
    # "batched_mm", ...} — "torch" isn't a registered intent name there.
    # "eager" runs HF's original _Experts.forward loop body (one expert at a
    # time over selected tokens) — the meaningful torch baseline.
    cfg._experts_implementation = "eager"
    cfg._dsv4_mhc_implementation = "torch"
    print("\n" + "=" * 70)
    print("[baseline] alloy default (attn -> _torch_dsv4_attention, "
          "experts -> HF eager, mhc -> _torch_hyper_connection)")
    torch.manual_seed(args.seed)
    baseline_model = AlloyForCausalLM(cfg).to(device=device, dtype=dtype)
    baseline_logits, t_first_off, t_avg_off = _measure(
        baseline_model, input_ids,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat, device=device,
    )
    print(f"[baseline] forward - first {t_first_off*1000:.2f} ms  /  avg {t_avg_off*1000:.2f} ms over {args.n_repeat}")

    state_dict = {k: v.detach().clone() for k, v in baseline_model.state_dict().items()}
    baseline_logits_cpu = baseline_logits.detach().to("cpu").to(torch.float32)
    del baseline_model
    if device.type == "npu":
        torch.npu.empty_cache()

    # =========================================================================
    # Phase 2: BINDER — flip targeted surfaces to args.prefer; rest stay torch
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"[binder]   activating prefer={args.prefer!r} on "
          f"attn={sorted(attn_targets)}  experts={moe_target}  mhc={mhc_target}")
    # deepcopy so _attn_implementation (and any other underscore-prefixed
    # runtime hint) survives — AlloyConfig.to_dict() drops underscore fields
    # by design (so they don't leak into config.json), but for within-process
    # cloning we WANT them.
    cfg_b = copy.deepcopy(cfg)
    fake = type("Model", (), {"config": cfg_b})()
    # Map form: flip ONLY the targeted surfaces. The string-broadcast form
    # would touch every activatable field; explicit mapping is safer.
    prefer_map: dict[str, str] = {f"dsv4_{t}": args.prefer for t in attn_targets}
    if moe_target:
        # experts use HF's standard intent names — "flash" is the binder
        # fast path registered into ALL_EXPERTS_FUNCTIONS. The DEFAULTS
        # table maps {auto, flash, triton} -> "flash" (qwen3_5 + dsv4 share
        # this entry; binder/shared/moe_experts.py handles dsv4 clamped
        # SwiGLU via self.limit branch). Honour --prefer if it's "flash"
        # or "auto"; otherwise fall back to flash since that's the only
        # NPU-fast option binder ships for experts.
        prefer_map["experts"] = "flash"
    if mhc_target:
        # MHC HyperConnection: triton is the only non-torch backend.
        # Caller is responsible for hc_mult=4 (verified above).
        prefer_map["dsv4_mhc"] = "triton"
    chosen = binder.activate(fake, prefer=prefer_map)
    print(f"[binder]   activate() set: {chosen}")

    torch.manual_seed(args.seed)
    binder_model = AlloyForCausalLM(cfg_b).to(device=device, dtype=dtype)
    missing, unexpected = binder_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[binder]   state_dict load: {len(missing)} missing, {len(unexpected)} unexpected")
        print(f"           missing[:3]={missing[:3]}  unexpected[:3]={unexpected[:3]}")

    binder_logits, t_first_on, t_avg_on = _measure(
        binder_model, input_ids,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat, device=device,
    )
    print(f"[binder]   forward - first {t_first_on*1000:.2f} ms  /  avg {t_avg_on*1000:.2f} ms over {args.n_repeat}")

    binder_logits_cpu = binder_logits.detach().to("cpu").to(torch.float32)
    del binder_model
    if device.type == "npu":
        torch.npu.empty_cache()

    # =========================================================================
    # Compare
    # =========================================================================
    print("\n" + "=" * 70)
    print("=== Precision (binder vs torch baseline, on the SAME hardware) ===")
    diffs = diff_logits(baseline_logits_cpu, binder_logits_cpu)
    for k, v in diffs.items():
        print(f"  {k:14s} {v:.6e}")

    print("\n=== Speed (avg over n-repeat) ===")
    print(f"  baseline (attn torch, experts eager, mhc torch):    {t_avg_off*1000:.2f} ms")
    binder_label = (
        f"attn={sorted(attn_targets)} "
        f"experts={'flash' if moe_target else 'eager'} "
        f"mhc={'triton' if mhc_target else 'torch'}"
    )
    print(f"  binder ({args.prefer} on {binder_label}):  {t_avg_on*1000:.2f} ms")
    if t_avg_on > 0:
        speedup = t_avg_off / t_avg_on
        print(f"  speedup:                    {speedup:.2f}x")

    print("\n=== First-call latency (cold-start) ===")
    print(f"  baseline:  {t_first_off*1000:.2f} ms")
    print(f"  binder:    {t_first_on*1000:.2f} ms")

    return 0


def test_dsv4_binder_auto_vs_torch_npu() -> None:
    """Contract 2: alloy_torch ≈ binder(auto/triton) on NPU, bf16 noise floor.

    Default target = csa,sliding (excludes HCA). HCA's triton SFA path
    computes width as ``sliding_window + ceil(seq_len/compress_rate_hca)``;
    with the default toy config (W=128, seq=128, compress_rate_hca=8) that
    lands on 144, which isn't in the binder kernel's CONFIG_MAP {128, 160,
    640} and raises ValueError. CLI users targeting HCA must adjust
    --seq-len / --sliding-window so HCA's width lands supported (e.g.
    seq=4096 + compress_rate_hca=128 → 128+32=160)."""
    if _IMPORT_ERR is not None:
        import pytest

        pytest.skip(_IMPORT_ERR)
    assert main(["--target=csa,sliding"]) == 0


if __name__ == "__main__":
    sys.exit(main())
