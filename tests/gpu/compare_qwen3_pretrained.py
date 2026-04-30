"""End-to-end equivalence test against real Qwen3-4B weights.

Flow (sequential to avoid holding two copies of weights simultaneously):

    Phase 1: load HF AutoModelForCausalLM("Qwen/Qwen3-4B") → forward on a fixed
             input → greedy-generate N tokens → save CPU logits + token ids →
             delete model from HBM, empty_cache.

    Phase 2: instantiate AlloyForCausalLM from the derived AlloyConfig →
             stream state_dict directly from the on-disk safetensors → run the
             same forward + generate → compare.

Determinism:
    * Fixed seed before any random op.
    * Greedy decoding (do_sample=False, num_beams=1) — bit-exact reproducible
      given identical weights + math.
    * attn_implementation="eager" on both sides so kernel choice doesn't
      introduce numerical drift.

Usage:
    python -m alloy.tests.gpu.compare_qwen3_pretrained \\
        --pretrained Qwen/Qwen3-4B --dtype bf16 --max-new-tokens 16
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Equivalence test contract: HF reference and alloy port should match on the
# torch reference path. Pin alloy to torch by disabling the hf-npu-binder
# auto-bridge BEFORE alloy is imported (no-op on a typical CUDA box without
# binder installed; defensive in case the env happens to have it).
os.environ["ALLOY_DISABLE_AUTO_BRIDGE"] = "1"

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alloy import AlloyForCausalLM
from alloy.tests._compare_utils import (
    compare_tokens,
    diff_logits,
    empty_cache,
    alloy_config_from_qwen3,
    build_skeleton,
    load_state_dict_from_disk,
    pick_device,
)

DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog. In a single sentence, explain why recursion is useful:"


def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[s]


def phase_reference(
    pretrained: str,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    print(f"[phase-1] Loading HF reference: {pretrained}")
    ref = AutoModelForCausalLM.from_pretrained(
        pretrained, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()

    print(f"[phase-1] model_type={ref.config.model_type}  dtype={dtype}  device={device}")

    ids = input_ids.to(device)
    with torch.no_grad():
        fwd_logits = ref(input_ids=ids, use_cache=False).logits.detach().to("cpu", torch.float32)

    torch.manual_seed(0)
    with torch.no_grad():
        gen = ref.generate(
            ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=ref.config.eos_token_id if isinstance(ref.config.eos_token_id, int) else None,
        )

    out = {
        "logits": fwd_logits,
        "generated_ids": gen.detach().to("cpu"),
        "hf_config": ref.config,
    }
    print(f"[phase-1] logits shape {tuple(fwd_logits.shape)}, generated shape {tuple(gen.shape)}")

    del ref
    empty_cache(device)
    print("[phase-1] released reference from memory")
    return out


def phase_ours(
    pretrained: str,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    hf_config,
) -> dict[str, Any]:
    print(f"[phase-2] Building AlloyForCausalLM from HF config")
    alloy_cfg = alloy_config_from_qwen3(hf_config)
    alloy_cfg._attn_implementation = "eager"

    with build_skeleton(dtype):
        ours = AlloyForCausalLM(alloy_cfg)

    # build_skeleton runs under no_init_weights, which skips HF's post_init
    # tie_weights() step. Call it here before load_state_dict so that, for
    # checkpoints with tie_word_embeddings=True (e.g. Qwen3-4B), lm_head.weight
    # is aliased to model.embed_tokens.weight. load_state_dict will then fill
    # both via the single "model.embed_tokens.weight" entry. (load_state_dict
    # will still report lm_head.weight as missing — harmless, the storage is
    # correct via the alias.)
    ours.tie_weights()

    print(f"[phase-2] streaming state_dict from {pretrained}")
    sd = load_state_dict_from_disk(
        pretrained,
        ignore_patterns=[r"^mtp\.", r"^model\.visual\.", r".*rotary_emb\.inv_freq$"],
        device="cpu",
    )
    res = ours.load_state_dict(sd, strict=False)
    print(f"[phase-2] load_state_dict: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"           missing (first 5): {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"           unexpected (first 5): {res.unexpected_keys[:5]}")
    del sd
    empty_cache(device)

    ours = ours.to(device).eval()

    ids = input_ids.to(device)
    with torch.no_grad():
        fwd_logits = ours(input_ids=ids, use_cache=False).logits.detach().to("cpu", torch.float32)

    torch.manual_seed(0)
    with torch.no_grad():
        gen = ours.generate(
            ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=alloy_cfg.eos_token_id if isinstance(alloy_cfg.eos_token_id, int) else None,
        )

    out = {"logits": fwd_logits, "generated_ids": gen.detach().to("cpu")}
    print(f"[phase-2] logits shape {tuple(fwd_logits.shape)}, generated shape {tuple(gen.shape)}")

    del ours
    empty_cache(device)
    print("[phase-2] released ours from memory")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="Qwen/Qwen3-4B",
                        help="HF hub id or local path containing config.json + safetensors")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for logits diff (bf16-friendly default).")
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = _dtype_from_str(args.dtype)

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.pretrained}")
    tok = AutoTokenizer.from_pretrained(args.pretrained)
    input_ids = tok(args.prompt, return_tensors="pt").input_ids

    ref_out = phase_reference(args.pretrained, input_ids, args.max_new_tokens, device, dtype)
    our_out = phase_ours(args.pretrained, input_ids, args.max_new_tokens, device, dtype, ref_out["hf_config"])

    print("\n=== Comparison ===")

    # 1. Token-level behavioral match (strongest signal)
    tok_cmp = compare_tokens(ref_out["generated_ids"], our_out["generated_ids"])
    print(f"Generated tokens match exactly: {tok_cmp['match']}")
    if not tok_cmp["match"]:
        print(f"  first divergence (row, col): {tok_cmp['first_divergence_row_col']}")
    print(f"  ref (decoded):  {tok.decode(tok_cmp['ref_tokens'][0], skip_special_tokens=True)!r}")
    print(f"  ours (decoded): {tok.decode(tok_cmp['our_tokens'][0], skip_special_tokens=True)!r}")

    # 2. Logits numerical diff (tighter diagnostic)
    logit_diff = diff_logits(ref_out["logits"], our_out["logits"])
    print("Logits diff:")
    for k, v in logit_diff.items():
        print(f"  {k}: {v}")

    tokens_ok = tok_cmp["match"]
    logits_ok = logit_diff["max_abs"] <= args.atol + args.rtol * logit_diff["max_ref_abs"]
    overall = tokens_ok and logits_ok
    print(f"\n{'PASS' if overall else 'FAIL'}  (tokens_ok={tokens_ok}, logits_ok={logits_ok}, "
          f"tol: atol={args.atol}, rtol={args.rtol})")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
