"""Diagnostic — isolate which config setattr crashes HF's bf16 GDN forward.

Background
----------
When ``compare_training_step_vs_hf.py`` runs row 3 (``--prefer flash
--attn-impl eager --dtype bf16``), HF's GDN forward crashes inside conv1d:

    RuntimeError: Input type (npuFloatType) and weight type
                  (npuBFloat16Type) should be the same

Row 2 (same dtype, no binder) works on HF. The only difference between
row 2 and row 3 is two attribute assignments on ``hf_model.config``:

    hf_model.config._experts_implementation     = "flash"
    hf_model.config._qwen3_5_gdn_implementation = "flash"

This script imports the binder bridge (so ``ALL_EXPERTS_FUNCTIONS["flash"]``
is registered) and then runs four trials on the same fresh HF model.
The crash pattern tells us which attribute (or interaction) is the actual
trigger:

    trial 0  no setattr                              (= row 2 control)
    trial 1  only _experts_implementation = "flash"
    trial 2  only _qwen3_5_gdn_implementation = "flash"
    trial 3  both set                                (= row 3 reproduction)

Earlier rev forgot to import the bridge — that surfaced a different
crash (``KeyError: 'flash' is not registered``) which masked the real
issue. Now bridge is imported up-front to mirror compare_training_step's
setup exactly.

Usage::

    python -m alloy.tests.npu.diag_hf_setattr_isolation
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Skip-detect after import so --help works.
_SKIP: str | None = None
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError as _e:
    _SKIP = f"torch_npu not available ({_e})"

# Import the bridge UP-FRONT so ALL_EXPERTS_FUNCTIONS["flash"] is registered
# before any model construction or forward — exactly mirrors what
# compare_training_step_vs_hf.py does. Without this, trial 1/3 would crash
# with `KeyError: 'flash' not registered` rather than the original GDN
# dtype mismatch we're trying to isolate.
if _SKIP is None:
    try:
        import alloy.integrations.hf_npu_binder  # noqa: F401  -- side effect: registers "flash"
    except ImportError as _e:
        _SKIP = f"alloy.integrations.hf_npu_binder not importable ({_e})"


def _build_hf():
    """Mirror compare_training_step_vs_hf row 3 setup: 4-layer 3:1 GDN+attn,
    hidden=64, seq_len=32, bf16, eager attn.
    """
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    cfg = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
        max_position_embeddings=512,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        output_router_logits=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25},
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"

    torch.manual_seed(0)
    model = Qwen3_5MoeForCausalLM(cfg).to(device="npu", dtype=torch.bfloat16)
    return model, cfg


def _attach_layer_probes(model):
    """Attach forward hooks that report (layer_idx, dtype) on entry to each
    sub-step of GDN, so a crash narrows down to the exact op.
    """
    seen: list[tuple[str, torch.dtype]] = []

    def make(name):
        def hook(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            if isinstance(t, torch.Tensor):
                seen.append((name, t.dtype))
        return hook

    handles = []
    handles.append(model.model.embed_tokens.register_forward_hook(make("embed")))
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make(f"layer_{i}")))
        # GDN sub-probes (only on linear_attention layers)
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            handles.append(la.in_proj_qkv.register_forward_hook(make(f"layer_{i}.in_proj_qkv")))
            handles.append(la.conv1d.register_forward_hook(make(f"layer_{i}.conv1d")))
    return seen, handles


def _trial(label: str, attrs_to_set: dict[str, str], model, input_ids):
    """Set the requested attrs on model.config, run forward, report success/error."""
    # Reset any previous setattrs so trials don't bleed.
    for k in ("_experts_implementation", "_qwen3_5_gdn_implementation"):
        if hasattr(model.config, k):
            try:
                delattr(model.config, k)
            except AttributeError:
                pass

    for k, v in attrs_to_set.items():
        setattr(model.config, k, v)

    set_str = ", ".join(f"{k}={v!r}" for k, v in attrs_to_set.items()) if attrs_to_set else "(no setattr)"
    print(f"\n[{label}] config setattr: {set_str}")

    # Mirror compare_training_step_vs_hf._train_step exactly: train mode +
    # gradient-enabled forward + manual CE-with-shift loss + backward. The
    # earlier eval+no_grad rev failed to reproduce, so something in the
    # train/grad path is the trigger.
    seen, handles = _attach_layer_probes(model)
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, output_router_logits=False, use_cache=False)
        # match compare's loss: shifted CE in fp32
        import torch.nn.functional as F
        shift_logits = out.logits[..., :-1, :].contiguous().to(torch.float32)
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss.backward()
        finite_logits = torch.isfinite(out.logits).all().item()
        finite_loss = torch.isfinite(loss).all().item()
        print(f"[{label}] OK — logits {tuple(out.logits.shape)}  "
              f"loss={loss.item():.4f}  finite_logits={finite_logits}  finite_loss={finite_loss}")
        ok = True
    except Exception as e:  # noqa: BLE001 — we want every failure mode
        print(f"[{label}] CRASH — {type(e).__name__}: {e}")
        # Trim traceback to the most informative frames (the deepest one).
        tb = traceback.format_exc().splitlines()
        for line in tb[-15:]:
            print(f"          {line}")
        ok = False
    finally:
        for h in handles:
            h.remove()

    # Last few capture points + dtypes — shows where we got before the crash.
    print(f"[{label}] sub-step trace (got to):")
    for name, dt in seen[-8:]:
        print(f"          {name:32s} {dt}")
    return ok


def main() -> int:
    if _SKIP is not None:
        print(f"SKIP — {_SKIP}; this script must run on NPU hardware.")
        return 0

    model, cfg = _build_hf()
    print(f"Built HF Qwen3_5MoeForCausalLM bf16 on {next(model.parameters()).device}")
    print(f"  layer_types: {cfg.layer_types}")

    torch.manual_seed(1)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 32), device="npu")

    trials = [
        ("trial 0  control",                             {}),
        ("trial 1  experts only",                        {"_experts_implementation": "flash"}),
        ("trial 2  gdn only",                            {"_qwen3_5_gdn_implementation": "flash"}),
        ("trial 3  both (= compare_training_step row 3)", {
            "_experts_implementation": "flash",
            "_qwen3_5_gdn_implementation": "flash",
        }),
    ]

    results = []
    for label, attrs in trials:
        ok = _trial(label, attrs, model, input_ids)
        results.append((label, attrs, ok))

    # Summary
    print("\n" + "=" * 72)
    print("Summary:")
    for label, attrs, ok in results:
        status = "OK   " if ok else "CRASH"
        print(f"  {status}  {label}")

    # Diagnosis
    only_control_ok = results[0][2] and not any(r[2] for r in results[1:])
    if results[0][2] and not results[1][2] and results[2][2]:
        diag = "ROOT CAUSE: setting _experts_implementation alone triggers the GDN crash."
    elif results[0][2] and results[1][2] and not results[2][2]:
        diag = "ROOT CAUSE: setting _qwen3_5_gdn_implementation alone triggers the GDN crash."
    elif results[0][2] and results[1][2] and results[2][2] and not results[3][2]:
        diag = "ROOT CAUSE: only the COMBINATION of both fields triggers the crash (interaction)."
    elif only_control_ok:
        diag = "ROOT CAUSE: any non-default _<x>_implementation field triggers the crash."
    elif all(r[2] for r in results):
        diag = "No crash reproduced — env may differ from the original reporting."
    else:
        diag = "Unexpected pattern; inspect per-trial output above."

    print(f"\n{diag}")
    return 0 if results[0][2] else 1


if __name__ == "__main__":
    sys.exit(main())
