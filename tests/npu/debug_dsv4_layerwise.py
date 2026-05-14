"""Per-layer drift between alloy(torch) and alloy(binder triton) for any
target subset of DSV4 attention layer types.

When ``compare_dsv4_binder_vs_torch.py`` reports large logit drift, this
is where to start. Builds two alloy DSV4 models with the same state_dict,
one routing the targeted ``dsv4_{kind}.attention`` keys -> torch and the
other -> triton, hooks every decoder layer's output, and reports the
per-layer ``max_abs`` / ``mean_abs`` / relative drift.

Reading the output:

  * Layers NOT in --target should be byte-identical (max_abs == 0; both
    runs are using the exact same torch callable). If they show drift,
    either the bridge re-wired something it shouldn't have, or the
    state_dict copy isn't byte-identical between runs.

  * The FIRST targeted layer ("first divergence") should show the actual
    kernel/wrapper drift. In a correct wire-up this is bf16 noise floor:
    max_abs ~ 1e-3 ~ 1e-2 on small configs. If it's 1e-1 ~ 10, the
    binder wrapper is buggy for this layer type.

  * Targeted-but-later layers and downstream non-targeted layers show
    drift propagated from the first divergence — magnitude monotone
    non-decreasing as the residual stream accumulates the error.

Usage::

    # Default: CSA only (same as old debug_dsv4_csa_layerwise.py)
    python -m tests.npu.debug_dsv4_layerwise

    # HCA-only debug at long sequence (where CONFIG_MAP=640 trips)
    python -m tests.npu.debug_dsv4_layerwise --target hca --seq-len 4096

    # Compare all 3 layer types together
    python -m tests.npu.debug_dsv4_layerwise --target all --seq-len 4096
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_IMPORT_ERR: str | None = None
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
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
    from alloy.tests._compare_utils import pick_device
else:
    AlloyConfig = AlloyForCausalLM = binder = None  # type: ignore[assignment]
    pick_device = None  # type: ignore[assignment]


_ALL_TARGETS = {"csa", "hca", "sliding"}
_LAYER_TYPE_BY_TARGET = {
    "csa":     "dsv4_csa_attention",
    "hca":     "dsv4_hca_attention",
    "sliding": "dsv4_sliding_attention",
}


# ---------------------------------------------------------------------------
# Config — same as compare_dsv4_binder_vs_torch
# ---------------------------------------------------------------------------
def _layer_pattern(num_layers: int) -> list[str]:
    base = ["dsv4_hca_attention", "dsv4_hca_attention",
            "dsv4_csa_attention", "dsv4_sliding_attention"]
    out: list[str] = []
    while len(out) < num_layers:
        out.extend(base)
    return out[:num_layers]


def _ffn_pattern(num_layers: int) -> list[str]:
    base = ["dsv4_hash_moe", "dsv4_moe", "dsv4_moe", "dsv4_moe"]
    out: list[str] = []
    while len(out) < num_layers:
        out.extend(base)
    return out[:num_layers]


def _build_config(args):
    head_dim = args.hidden_size // args.num_attention_heads
    return AlloyConfig(
        vocab_size=args.vocab_size, hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=1, head_dim=head_dim,
        intermediate_size=args.hidden_size,
        max_position_embeddings=max(args.seq_len, 512),
        sliding_window=args.sliding_window,
        rms_norm_eps=1e-6, attention_bias=False, attention_dropout=0.0,
        hc_mult=2, hc_sinkhorn_iters=2, hc_eps=1e-6, use_mhc=True,
        n_routed_experts=4, num_experts_per_tok=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        swiglu_limit=10.0, mlp_bias=False,
        q_lora_rank=args.hidden_size // 2,
        o_groups=2, o_lora_rank=args.hidden_size // 2,
        index_n_heads=args.num_attention_heads,
        index_head_dim=head_dim, index_topk=args.index_topk,
        compress_rates={"compressed_sparse_attention": 4,
                        "heavily_compressed_attention": 8},
        rope_parameters={
            "main":     {"rope_type": "default", "rope_theta": 10000.0,  "partial_rotary_factor": 0.25},
            "compress": {"rope_type": "default", "rope_theta": 100000.0, "partial_rotary_factor": 0.25},
        },
        layer_types=_layer_pattern(args.num_layers),
        ffn_types=_ffn_pattern(args.num_layers),
        attn_implementation="eager",
    )


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------
def _hook_layers(model) -> dict[str, list[torch.Tensor]]:
    """Register forward hooks on every decoder layer + final norm.

    Hooks keep the captured tensor on the original device (no .to("cpu")
    inside the hook) to avoid forcing a sync mid-forward that can collide
    with the triton kernel's stream. CPU+fp32 conversion happens AFTER
    the forward completes, in _diff_captures.
    """
    captures: dict[str, list[torch.Tensor]] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            captures.setdefault(name, []).append(t.detach())
        return hook

    decoder = model.model.layers
    for i, layer in enumerate(decoder):
        layer.register_forward_hook(make_hook(f"layer_{i}"))
    model.model.norm.register_forward_hook(make_hook("final_norm"))
    return captures


def _diff_captures(
    a: dict[str, list[torch.Tensor]],
    b: dict[str, list[torch.Tensor]],
    layer_types: list[str],
    targets: set[str],
) -> None:
    print()
    print("=" * 118)
    print(f"{'point':22s} {'layer_type':28s} {'targeted':>9s} "
          f"{'max_abs':>12s} {'mean_abs':>12s} {'max_rel':>12s} {'mean_rel':>12s}")
    print("-" * 118)
    keys = list(a.keys())
    for k in keys:
        ta_list, tb_list = a.get(k, []), b.get(k, [])
        if not ta_list or not tb_list:
            continue
        ta, tb = ta_list[-1], tb_list[-1]
        if ta.shape != tb.shape:
            print(f"{k:22s} {'shape mismatch':28s} {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        ta = ta.to("cpu").to(torch.float32)
        tb = tb.to("cpu").to(torch.float32)
        d = (ta - tb).abs()
        ref = ta.abs()
        max_ref = ref.max().item()
        mean_ref = ref.mean().item()
        max_rel = d.max().item() / max_ref if max_ref > 0 else 0.0
        mean_rel = d.mean().item() / mean_ref if mean_ref > 0 else 0.0

        layer_idx = int(k.split("_")[1]) if k.startswith("layer_") else -1
        lt = layer_types[layer_idx] if 0 <= layer_idx < len(layer_types) else ""

        # `targeted` = is this layer's type in the binder-swap set
        if lt:
            kind = lt.replace("dsv4_", "").replace("_attention", "")
            is_targeted = kind in targets
            tag = "YES" if is_targeted else "no"
        else:
            tag = ""

        print(f"{k:22s} {lt:28s} {tag:>9s} "
              f"{d.max().item():12.4e} {d.mean().item():12.4e} "
              f"{max_rel:12.4e} {mean_rel:12.4e}")
    print("=" * 118)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[s]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sliding-window", type=int, default=128)
    parser.add_argument("--index-topk", type=int, default=32)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target", default="csa",
        help="comma-separated subset of {csa,hca,sliding,all} for which layer "
             "types to swap to binder triton in model B. Untargeted layer "
             "types stay torch on BOTH models so they should produce "
             "byte-identical outputs.",
    )
    args = parser.parse_args()

    if _IMPORT_ERR is not None:
        print(f"SKIP - {_IMPORT_ERR}")
        return 0

    # Parse --target
    raw = [t.strip() for t in args.target.split(",") if t.strip()]
    if "all" in raw:
        targets = set(_ALL_TARGETS)
    else:
        unknown = set(raw) - _ALL_TARGETS
        if unknown:
            print(f"ERROR - --target contains unknown entries {sorted(unknown)}; "
                  f"choose from {sorted(_ALL_TARGETS)} or 'all'.")
            return 2
        targets = set(raw)

    device = pick_device()
    dtype = _dtype_from_str(args.dtype)
    cfg = _build_config(args)
    counts = {t: sum(1 for lt in cfg.layer_types if lt == _LAYER_TYPE_BY_TARGET[t])
              for t in _ALL_TARGETS}
    targeted_count = sum(counts[t] for t in targets)
    print(f"device={device}  dtype={dtype}  layers={args.num_layers}  "
          f"seq_len={args.seq_len}  sliding_window={args.sliding_window}  index_topk={args.index_topk}")
    print(f"layer_types = {cfg.layer_types}")
    print(f"binder targets = {sorted(targets)}  → {targeted_count}/{args.num_layers} layers swap to triton in model B")

    if targeted_count == 0:
        print(f"SKIP - no targeted layers in this pattern; nothing for binder to swap.")
        return 0

    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len)).to(device)

    # -------------------------------------------------------------------------
    # Build A: ALL torch. Master state_dict.
    # -------------------------------------------------------------------------
    # Pin every DSV4 attention surface to torch so binder import doesn't
    # silently re-wire any of them.
    for t in _ALL_TARGETS:
        setattr(cfg, f"_dsv4_{t}_implementation", "torch")
    print(f"\n[A] building alloy with ALL dsv4_{{csa,hca,sliding}}_implementation='torch'")
    torch.manual_seed(args.seed)
    model_a = AlloyForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    cap_a = _hook_layers(model_a)
    state_dict = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    for i, lt in enumerate(cfg.layer_types):
        fn = getattr(model_a.model.layers[i].self_attn, '_attn_fn', None)
        print(f"    layer_{i} ({lt}) _attn_fn = {fn}")

    print("[A] running forward...", flush=True)
    with torch.no_grad():
        _ = model_a(input_ids=input_ids, use_cache=False)
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    print(f"[A] forward done. captures: {sorted(cap_a.keys())}", flush=True)

    # -------------------------------------------------------------------------
    # Build B: targeted layer types -> binder triton; others -> torch.
    # -------------------------------------------------------------------------
    cfg_b = copy.deepcopy(cfg)
    fake = type("Model", (), {"config": cfg_b})()
    prefer_map = {f"dsv4_{t}": "triton" for t in targets}
    chosen = binder.activate(fake, prefer=prefer_map)
    print(f"\n[B] activate() set: {chosen}")
    # Untargeted layer types pinned to torch
    for t in _ALL_TARGETS - targets:
        setattr(cfg_b, f"_dsv4_{t}_implementation", "torch")
    torch.manual_seed(args.seed)
    model_b = AlloyForCausalLM(cfg_b).to(device=device, dtype=dtype).eval()
    missing, unexpected = model_b.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"    state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:    print(f"    missing[:3]={missing[:3]}")
        if unexpected: print(f"    unexpected[:3]={unexpected[:3]}")
    cap_b = _hook_layers(model_b)
    for i, lt in enumerate(cfg.layer_types):
        fn = getattr(model_b.model.layers[i].self_attn, '_attn_fn', None)
        print(f"    layer_{i} ({lt}) _attn_fn = {fn}")

    print("[B] running forward...", flush=True)
    try:
        with torch.no_grad():
            _ = model_b(input_ids=input_ids, use_cache=False)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        print(f"[B] forward done. captures: {sorted(cap_b.keys())}", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"\n[B] FORWARD RAISED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        # Still try to diff what we have (model_a captures are complete)
        print(f"[B] captures collected before failure: {sorted(cap_b.keys())}", flush=True)

    print("computing per-layer diffs...", flush=True)
    _diff_captures(cap_a, cap_b, cfg.layer_types, targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
