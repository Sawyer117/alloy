"""Per-layer drift between alloy(torch CSA) and alloy(binder triton CSA).

When compare_dsv4_binder_vs_torch.py reports large logit drift, this is
where to start. Builds two alloy DSV4 models with the same state_dict,
one routing dsv4_csa.attention -> torch and the other -> triton, hooks
every decoder layer's output, and reports the per-layer max_abs diff.

Expected behaviour if the binder triton wrapper is correct::

  layer_0 (hca)      max_abs ~ 0           (both run torch — should be byte-identical)
  layer_1 (hca)      max_abs ~ 0           (same)
  layer_2 (csa)      max_abs ~ 1e-3 ~ 1e-2 (bf16 fused-kernel noise; FIRST DIVERGENCE)
  layer_3 (sliding)  max_abs ~ same as layer_2 propagated downstream
  final logits       same envelope

If the wrapper is wrong, layer_2 drift will be 1e-1 ~ 10 (well above the
fused-kernel noise floor) and downstream layers amplify it.

If we see drift starting BEFORE layer_2 (the CSA layer), that means the
two phases aren't actually controlled — either the binder bridge import
is re-wiring HCA / sliding paths too, or the state_dict copy isn't
byte-identical between runs.

Usage::

    python -m alloy.tests.npu.debug_dsv4_csa_layerwise --dtype bf16
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


def _build_config(args) -> AlloyConfig:
    head_dim = args.hidden_size // args.num_attention_heads
    return AlloyConfig(
        vocab_size=args.vocab_size, hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads, num_key_value_heads=1,
        head_dim=head_dim, intermediate_size=args.hidden_size,
        max_position_embeddings=max(args.seq_len, 512),
        sliding_window=args.sliding_window,
        rms_norm_eps=1e-6, attention_bias=False, attention_dropout=0.0,
        hc_mult=2, hc_sinkhorn_iters=2, hc_eps=1e-6, use_mhc=True,
        n_routed_experts=4, num_experts_per_tok=2,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        swiglu_limit=10.0, mlp_bias=False,
        q_lora_rank=args.hidden_size // 2, o_groups=2,
        o_lora_rank=args.hidden_size // 2,
        index_n_heads=args.num_attention_heads, index_head_dim=head_dim,
        index_topk=args.index_topk,
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
    """Register forward hooks on every decoder layer; returns a dict that
    fills up with captured outputs as the model runs."""
    captures: dict[str, list[torch.Tensor]] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            # Layer output is either a Tensor or a tuple whose first element
            # is the hidden_states tensor.
            t = output[0] if isinstance(output, tuple) else output
            captures.setdefault(name, []).append(t.detach().to("cpu").to(torch.float32))
        return hook

    decoder = model.model.layers  # AlloyModel.layers
    for i, layer in enumerate(decoder):
        layer.register_forward_hook(make_hook(f"layer_{i}"))
    model.model.norm.register_forward_hook(make_hook("final_norm"))
    return captures


def _diff_captures(
    a: dict[str, list[torch.Tensor]],
    b: dict[str, list[torch.Tensor]],
    layer_types: list[str],
) -> None:
    print()
    print("=" * 102)
    print(f"{'point':22s} {'layer_type':28s} "
          f"{'max_abs':>12s} {'mean_abs':>12s} {'max_rel':>12s} {'mean_rel':>12s}")
    print("-" * 102)
    keys = list(a.keys())
    for k in keys:
        ta_list, tb_list = a.get(k, []), b.get(k, [])
        if not ta_list or not tb_list:
            continue
        ta, tb = ta_list[-1], tb_list[-1]  # compare the final forward call
        if ta.shape != tb.shape:
            print(f"{k:22s} {'shape mismatch':28s} {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        d = (ta - tb).abs()
        ref = ta.abs()
        # max_rel: drift relative to the largest reference value (how much
        # of the largest signal does the worst-position drift represent).
        # mean_rel: mean drift / mean signal magnitude (the bulk noise-floor
        # ratio). 1e-3 ~ 1e-2 in bf16 is the normal fused-kernel envelope.
        max_ref = ref.max().item()
        mean_ref = ref.mean().item()
        max_rel = d.max().item() / max_ref if max_ref > 0 else 0.0
        mean_rel = d.mean().item() / mean_ref if mean_ref > 0 else 0.0
        layer_idx = int(k.split("_")[1]) if k.startswith("layer_") else -1
        lt = layer_types[layer_idx] if 0 <= layer_idx < len(layer_types) else ""
        print(f"{k:22s} {lt:28s} "
              f"{d.max().item():12.4e} {d.mean().item():12.4e} "
              f"{max_rel:12.4e} {mean_rel:12.4e}")
    print("=" * 102)


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
    args = parser.parse_args()

    if _IMPORT_ERR is not None:
        print(f"SKIP - {_IMPORT_ERR}")
        return 0

    device = pick_device()
    dtype = _dtype_from_str(args.dtype)
    cfg = _build_config(args)
    csa_layers = sum(1 for lt in cfg.layer_types if lt == "dsv4_csa_attention")
    print(f"device={device}  dtype={dtype}  layers={args.num_layers}  "
          f"sliding_window={args.sliding_window}  index_topk={args.index_topk}")
    print(f"layer_types = {cfg.layer_types}")
    print(f"CSA layers in this run: {csa_layers}/{args.num_layers}  "
          f"(only these should diverge under a correct triton wrapper)")

    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len)).to(device)

    # -------------------------------------------------------------------------
    # Build A: torch-CSA. Build first so its state_dict is the shared
    # weight source for both runs.
    # -------------------------------------------------------------------------
    cfg.A_dsv4_csa_implementation = None  # touch nothing
    cfg._dsv4_csa_implementation = "torch"
    print("\n[A] building alloy with _dsv4_csa_implementation='torch'")
    torch.manual_seed(args.seed)
    model_a = AlloyForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    cap_a = _hook_layers(model_a)
    state_dict = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    print(f"    layer_0 self_attn._attn_fn = {getattr(model_a.model.layers[0].self_attn, '_attn_fn', None)}")
    print(f"    CSA layer self_attn._attn_fn = {getattr(model_a.model.layers[cfg.layer_types.index('dsv4_csa_attention')].self_attn, '_attn_fn', None)}")

    with torch.no_grad():
        _ = model_a(input_ids=input_ids, use_cache=False)

    # -------------------------------------------------------------------------
    # Build B: binder triton CSA, identical weights.
    # -------------------------------------------------------------------------
    # deepcopy so underscore-prefixed runtime hints (especially
    # _attn_implementation) carry over — to_dict() filters them by design,
    # which would otherwise let cfg_b's HCA/sliding layers fall back to
    # HF's auto-detected attn impl (often sdpa, no sinks) and diverge from
    # the cfg-built model from layer 0.
    cfg_b = copy.deepcopy(cfg)
    fake = type("Model", (), {"config": cfg_b})()
    chosen = binder.activate(fake, prefer={"dsv4_csa": "triton"})
    print(f"\n[B] activate() set: {chosen}")
    torch.manual_seed(args.seed)
    model_b = AlloyForCausalLM(cfg_b).to(device=device, dtype=dtype).eval()
    missing, unexpected = model_b.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"    state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:  print(f"    missing[:3]={missing[:3]}")
        if unexpected: print(f"    unexpected[:3]={unexpected[:3]}")
    cap_b = _hook_layers(model_b)
    print(f"    CSA layer self_attn._attn_fn = {getattr(model_b.model.layers[cfg.layer_types.index('dsv4_csa_attention')].self_attn, '_attn_fn', None)}")

    with torch.no_grad():
        _ = model_b(input_ids=input_ids, use_cache=False)

    _diff_captures(cap_a, cap_b, cfg.layer_types)
    return 0


if __name__ == "__main__":
    sys.exit(main())
