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

This script runs four trials on the same fresh HF model and reports which
trial(s) crash. The crash pattern tells us which attribute (or interaction)
is the actual trigger:

    trial 0  no setattr                              (= row 2 control)
    trial 1  only _experts_implementation = "flash"
    trial 2  only _qwen3_5_gdn_implementation = "flash"
    trial 3  both set                                (= row 3 reproduction)

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


def _build_hf():
    """Small qwen3.5-MoE TextConfig: 1 GDN layer + 1 attn layer is enough — the
    crash is at GDN conv1d, layer_0 hits first.
    """
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    cfg = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
        max_position_embeddings=512,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        layer_types=["linear_attention", "full_attention"],
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

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, output_router_logits=False, use_cache=False)
        finite = torch.isfinite(out.logits).all().item()
        print(f"[{label}] OK — logits {tuple(out.logits.shape)}  finite={finite}")
        return True
    except Exception as e:  # noqa: BLE001 — we want every failure mode
        print(f"[{label}] CRASH — {type(e).__name__}: {e}")
        # Trim traceback to the most informative frames (the deepest one).
        tb = traceback.format_exc().splitlines()
        # Print the last ~12 lines; that includes the deep frame + error.
        for line in tb[-15:]:
            print(f"          {line}")
        return False


def main() -> int:
    if _SKIP is not None:
        print(f"SKIP — {_SKIP}; this script must run on NPU hardware.")
        return 0

    model, cfg = _build_hf()
    print(f"Built HF Qwen3_5MoeForCausalLM bf16 on {next(model.parameters()).device}")
    print(f"  layer_types: {cfg.layer_types}")

    torch.manual_seed(1)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), device="npu")

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
