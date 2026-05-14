"""Standalone dtype-chain diagnostic for the DSV4 MoE flash path on NPU.

Why this script exists: ``compare_dsv4_binder_vs_torch.py --target experts``
crashes inside ``npu_grouped_matmul`` with ``x dtype FLOAT, weight dtype
BF16``, but static code reading suggests every tensor in the chain
should be bf16 (alloy does not put norms in ``_keep_in_fp32_modules_strict``;
all params get cast by ``model.to(bf16)``). The leak point isn't obvious
from code; we need runtime dtype prints to find it.

This file is the **non-invasive** way to do that: it imports alloy +
binder unchanged, attaches PyTorch forward hooks to every interesting
module, monkey-patches the binder ``ALL_EXPERTS_FUNCTIONS["flash"]``
entry to add an entry-point trace, runs one tiny forward, and dumps the
dtype + shape at each captured point. Production code stays clean.

Usage::

    # Show dtype at every interesting module's output, layer 0 only
    python -m tests.npu.diagnose_dsv4_dtype_chain

    # All layers
    python -m tests.npu.diagnose_dsv4_dtype_chain --all-layers

    # CPU run (no binder flash, just trace the torch eager path —
    # useful to confirm alloy's own dtype chain is clean BEFORE binder)
    python -m tests.npu.diagnose_dsv4_dtype_chain --no-binder --device cpu

Output convention:
    [hook <module_qualname>] dtype=<...> shape=<...>
    [flash entry]            dtype=<...>          # ← binder side
    [flash gate_up_proj]     dtype=<...>          # ← binder side
    ...

Read top-down: the first row whose ``dtype`` is fp32 instead of bf16 is
the leak source (or just upstream of it — depends on which module's
output was captured).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_IMPORT_ERR: str | None = None
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    _NPU_AVAILABLE = True
except ImportError:
    _NPU_AVAILABLE = False

try:
    import hf_npu_binder  # noqa: F401
    _BINDER_AVAILABLE = True
except ImportError as _e:
    _BINDER_AVAILABLE = False
    _IMPORT_ERR = f"hf_npu_binder not installed ({_e})"

from alloy import AlloyConfig, AlloyForCausalLM


# ---------------------------------------------------------------------------
# Forward-hook trace
# ---------------------------------------------------------------------------
def _fmt(x) -> str:
    """One-line summary of a tensor's dtype + shape (or a brief type tag)."""
    if isinstance(x, torch.Tensor):
        return f"dtype={x.dtype} shape={tuple(x.shape)}"
    if x is None:
        return "None"
    if isinstance(x, (tuple, list)):
        parts = [_fmt(t) for t in x]
        return f"[{', '.join(parts)}]"
    return f"<{type(x).__name__}>"


def _attach_hooks(model: torch.nn.Module, layer_filter: int | None) -> list:
    """Register forward hooks on every module whose qualname matches an
    interesting pattern. ``layer_filter=N`` restricts to layer N (keeps
    output short); ``None`` keeps all layers.

    Captured modules (chosen so the trace shows every dtype-flip
    candidate):
      - attn_hc / ffn_hc            ← HC outputs (collapsed should be bf16)
      - input_layernorm             ← norm output before attention
      - post_attention_layernorm    ← norm output before MLP (this is
                                     where fp32 would enter MoE flash)
      - self_attn / mixer           ← attention compute output
      - mlp                         ← FFN compute output
      - norm (model-level final)    ← bonus
    """
    interesting = (
        "attn_hc", "ffn_hc",
        "input_layernorm", "post_attention_layernorm",
        "self_attn", "mlp",
    )

    handles = []
    for qualname, module in model.named_modules():
        # qualname looks like "model.layers.0.attn_hc" / "model.layers.0.mlp"
        parts = qualname.split(".")
        if not parts:
            continue
        leaf = parts[-1]
        if leaf not in interesting and qualname != "model.norm":
            continue
        if layer_filter is not None:
            # only emit hooks for layers.<layer_filter>... or for model.norm
            if "layers" in parts:
                layer_idx_str = parts[parts.index("layers") + 1]
                try:
                    if int(layer_idx_str) != layer_filter:
                        continue
                except ValueError:
                    continue

        def make_hook(name: str):
            def hook(_mod, _inputs, output):
                print(f"[hook {name:55s}] {_fmt(output)}", flush=True)
            return hook

        handles.append(module.register_forward_hook(make_hook(qualname)))

    return handles


def _patch_binder_flash() -> "callable | None":
    """Re-register the ``"flash"`` entry in ``ALL_EXPERTS_FUNCTIONS`` with a
    trace wrapper. Returns the original callable so the caller can restore
    after the run. Returns ``None`` if binder isn't importable.
    """
    if not _BINDER_AVAILABLE:
        return None
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    original = ALL_EXPERTS_FUNCTIONS.get("flash")
    if original is None:
        print("WARN - ALL_EXPERTS_FUNCTIONS has no 'flash' entry; binder bridge "
              "not imported?", flush=True)
        return None

    def traced_flash(self, hidden_states, top_k_index, top_k_weights):
        print(f"[flash entry      hidden_states     ] {_fmt(hidden_states)}", flush=True)
        print(f"[flash entry      top_k_index       ] {_fmt(top_k_index)}", flush=True)
        print(f"[flash entry      top_k_weights     ] {_fmt(top_k_weights)}", flush=True)
        print(f"[flash entry      self.gate_up_proj ] {_fmt(getattr(self, 'gate_up_proj', None))}", flush=True)
        print(f"[flash entry      self.down_proj    ] {_fmt(getattr(self, 'down_proj', None))}", flush=True)
        print(f"[flash entry      self.is_transposed] {getattr(self, 'is_transposed', '?')}", flush=True)
        print(f"[flash entry      self.limit        ] {getattr(self, 'limit', '?')}", flush=True)
        return original(self, hidden_states, top_k_index, top_k_weights)

    ALL_EXPERTS_FUNCTIONS["flash"] = traced_flash
    return original


def _restore_binder_flash(original) -> None:
    if original is None:
        return
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    ALL_EXPERTS_FUNCTIONS["flash"] = original


# ---------------------------------------------------------------------------
# Config builder — same shape as compare_dsv4_binder_vs_torch
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
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=32,
                        help="Smaller is faster + less trace noise.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sliding-window", type=int, default=128)
    parser.add_argument("--index-topk", type=int, default=32)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-layers", action="store_true",
        help="Trace every layer (default: layer 0 only, to keep output short).",
    )
    parser.add_argument(
        "--no-binder", action="store_true",
        help="Run without activating the binder flash path — uses the alloy "
             "torch _Experts.forward loop instead. Useful for confirming the "
             "alloy-side dtype chain is clean before involving binder.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "npu", "cuda"],
        help="Device. 'auto' picks npu if torch_npu is importable, else cpu.",
    )
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[args.dtype]

    if args.device == "auto":
        device = torch.device("cuda" if _NPU_AVAILABLE else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device}  dtype={dtype}  layers={args.num_layers}  seq_len={args.seq_len}", flush=True)
    print(f"binder_path={'OFF' if args.no_binder else 'ON (experts -> flash)'}", flush=True)

    cfg = _build_config(args)
    # Pin attention surfaces to torch unconditionally — we're not exercising
    # those triton paths here, just want them deterministic.
    for t in ("csa", "hca", "sliding"):
        setattr(cfg, f"_dsv4_{t}_implementation", "torch")
    # Experts implementation: flash if binder, eager otherwise.
    if args.no_binder:
        cfg._experts_implementation = "eager"
    else:
        if not _BINDER_AVAILABLE:
            print(f"ERROR - --no-binder not set but binder import failed: {_IMPORT_ERR}", flush=True)
            return 2
        # Trigger the bridge — registers ALL_EXPERTS_FUNCTIONS["flash"] etc.
        import alloy.integrations.hf_npu_binder  # noqa: F401
        cfg._experts_implementation = "flash"

    torch.manual_seed(args.seed)
    model = AlloyForCausalLM(cfg).to(device=device, dtype=dtype).eval()

    # Patch + hook AFTER model construction so the model itself isn't
    # built with traced internals.
    original_flash = None if args.no_binder else _patch_binder_flash()
    layer_filter = None if args.all_layers else 0
    handles = _attach_hooks(model, layer_filter)

    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len)).to(device)
    print(f"\n=== forward begin (layer_filter={layer_filter}, hooks={len(handles)}) ===\n", flush=True)

    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=False)
        print("\n=== forward done — no crash ===", flush=True)
    except Exception as e:
        import traceback
        print(f"\n=== forward RAISED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        for h in handles:
            h.remove()
        _restore_binder_flash(original_flash)

    return 0


if __name__ == "__main__":
    sys.exit(main())
