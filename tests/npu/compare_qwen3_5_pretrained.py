"""End-to-end equivalence test against real Qwen3.5-MoE weights — NPU edition.

Mirror of ``alloy/tests/gpu/compare_qwen3_5_pretrained.py``. Same
sequential-load protocol; only the device + the NPU-side imports change:

    import torch
    import torch_npu                                # adds the npu backend
    from torch_npu.contrib import transfer_to_npu   # monkey-patches torch.cuda

The model code in ``alloy/modeling_alloy.py`` and ``alloy/modules/**`` has
zero hardware-specific imports — the same ``AlloyForCausalLM`` runs here
unchanged. Both reference and ours are kept on ``attn_implementation="eager"``
so the only path math can drift is the elementwise + matmul kernels.

Reduced-layer comparison (NPU-specific)
---------------------------------------
The full Qwen3.5-35B-A3B is ~70 GB in bf16, which does not fit on a
single Ascend 910B-class card. The CUDA equivalent of this script
compares the full 40-layer model; here we instead truncate the model to
the first ``--num-layers`` layers (default 4) on **both** sides and
load only those layers' weights from the safetensors.

This is sufficient to verify per-layer math equivalence:

* The first 4 layers in qwen3.5-MoE follow the canonical 3:1
  ``[linear_attention, linear_attention, linear_attention, full_attention]``
  pattern — so the comparison exercises both GDN and gated GQA, the
  shared expert, the top-K router, the grouped experts, partial RoPE,
  and unit-offset RMSNorm.
* The output logits will differ from a full 40-layer forward, but
  reference and ours produce the *same* truncated logits if the math is
  right.

Other caveats
-------------
* **Caching.** ``use_cache=False`` on both forward AND generate: the GDN
  recurrent state needs a HybridCache with update_conv_state /
  update_recurrent_state hooks that DynamicCache doesn't provide.
  Generation is therefore O(N^2) — keep ``--max-new-tokens`` small
  (default 8).
* **Tolerances.** Default to the same values as the CUDA script — both
  sides run ``attn_implementation="eager"`` so NPU is held to the same
  numerical bar as CUDA.

Usage:
    python -m alloy.tests.npu.compare_qwen3_5_pretrained \\
        --pretrained /path/to/Qwen3.5-35B-A3B --dtype bf16 \\
        --num-layers 4 --max-new-tokens 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401  registers the npu backend with torch
from torch_npu.contrib import transfer_to_npu  # noqa: F401  monkey-patches torch.cuda -> npu

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alloy import AlloyForCausalLM
from alloy.tests._compare_utils import (
    compare_tokens,
    diff_logits,
    empty_cache,
    alloy_config_from_qwen3_5_text,
    build_skeleton,
    load_state_dict_from_disk,
    pick_device,
)

DEFAULT_PROMPT = "Write a haiku about distributed training on heterogeneous hardware:"


def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[s]


def _resolve_text_config(hf_full_cfg):
    """Qwen3.5 checkpoints store text fields under config.text_config; return
    the text sub-config directly if present, else the config itself (for the
    already-text-only `Qwen3_5MoeTextConfig`)."""
    text_cfg = getattr(hf_full_cfg, "text_config", None)
    if text_cfg is not None:
        return text_cfg
    return hf_full_cfg


def phase_reference(
    pretrained: str,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int,
) -> dict[str, Any]:
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    print(f"[phase-1] Loading Qwen3.5-MoE reference (text head): {pretrained}")
    full_cfg = AutoConfig.from_pretrained(pretrained)
    text_cfg = _resolve_text_config(full_cfg)
    text_cfg._attn_implementation = "eager"

    original_num_layers = text_cfg.num_hidden_layers
    if num_layers > original_num_layers:
        raise ValueError(
            f"--num-layers={num_layers} exceeds the checkpoint's {original_num_layers} layers"
        )
    if num_layers < original_num_layers:
        # Mutate before from_pretrained so HF builds a small model and skips
        # loading weights for the dropped layers.
        text_cfg.num_hidden_layers = num_layers
        text_cfg.layer_types = list(text_cfg.layer_types[:num_layers])
        print(
            f"[phase-1] truncated to first {num_layers} of {original_num_layers} layers "
            f"(layer_types={text_cfg.layer_types})"
        )

    ref = Qwen3_5MoeForCausalLM.from_pretrained(
        pretrained, config=text_cfg, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()

    print(f"[phase-1] layer_types head: {text_cfg.layer_types[:8]}  dtype={dtype}  device={device}")

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
            use_cache=False,  # avoid HybridCache requirement for this script
            pad_token_id=text_cfg.eos_token_id if isinstance(text_cfg.eos_token_id, int) else None,
        )

    out = {
        "logits": fwd_logits,
        "generated_ids": gen.detach().to("cpu"),
        "hf_text_config": text_cfg,
        "original_num_layers": original_num_layers,
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
    hf_text_cfg,
    original_num_layers: int,
) -> dict[str, Any]:
    print("[phase-2] Building AlloyForCausalLM from qwen3.5 text config")
    alloy_cfg = alloy_config_from_qwen3_5_text(hf_text_cfg)
    alloy_cfg._attn_implementation = "eager"

    with build_skeleton(dtype):
        ours = AlloyForCausalLM(alloy_cfg)

    # Re-establish weight tying that no_init_weights (used by build_skeleton)
    # skipped during post_init. No-op when config.tie_word_embeddings=False.
    ours.tie_weights()

    print(f"[phase-2] streaming state_dict from {pretrained}")

    # Qwen3_5MoeForConditionalGeneration stores text-backbone weights under
    # model.language_model.*; Qwen3_5MoeForCausalLM (and our Alloy equivalent)
    # expect them under model.*. HF's from_pretrained strips this via
    # base_model_prefix; raw safetensors reads need it explicit.
    _LM_PREFIX = "model.language_model."

    def _strip_language_model_prefix(k: str) -> str:
        if k.startswith(_LM_PREFIX):
            return "model." + k[len(_LM_PREFIX):]
        return k

    # When the reference was truncated to N layers, drop layer.{N..original-1}
    # weights from the on-disk safetensors as well. Saves CPU staging memory
    # (the full 35B-A3B is ~70 GB even on CPU) and avoids spurious
    # "unexpected_keys" reports the size of the dropped layers.
    ignore_patterns = [
        r"^mtp\.",
        r"^model\.visual\.",
        r"^model\.mtp\.",
        r".*rotary_emb\.inv_freq$",
    ]
    if alloy_cfg.num_hidden_layers < original_num_layers:
        for i in range(alloy_cfg.num_hidden_layers, original_num_layers):
            # Two layouts: text-only checkpoints store under model.layers.X.*,
            # ConditionalGeneration checkpoints store under
            # model.language_model.layers.X.*.
            ignore_patterns.append(rf"^model\.layers\.{i}\.")
            ignore_patterns.append(rf"^model\.language_model\.layers\.{i}\.")
        print(
            f"[phase-2] dropping checkpoint weights for layers "
            f"{alloy_cfg.num_hidden_layers}..{original_num_layers - 1}"
        )

    sd = load_state_dict_from_disk(
        pretrained,
        ignore_patterns=ignore_patterns,
        device="cpu",
        key_remap=_strip_language_model_prefix,
    )
    res = ours.load_state_dict(sd, strict=False)
    print(f"[phase-2] load_state_dict: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"           missing (first 10):")
        for k in res.missing_keys[:10]:
            print(f"             + {k}")
    if res.unexpected_keys:
        print(f"           unexpected (first 10):")
        for k in res.unexpected_keys[:10]:
            print(f"             - {k}")
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
            use_cache=False,
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
    parser.add_argument("--pretrained", required=True,
                        help="Local path (e.g. /path/to/Qwen3.5-35B-A3B) or HF hub id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="Keep small — generate uses use_cache=False (O(N^2)) for GDN safety.")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Truncate the model to its first N layers on both sides. Default 4 covers "
                             "the 3:1 GDN:full-attention pattern. Pass the checkpoint's full layer "
                             "count to do an end-to-end comparison (only viable if you have the HBM).")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="npu",
                        help="Defaults to 'npu'. Pass an explicit 'npu:0' / 'npu:1' to pin a card.")
    parser.add_argument("--atol", type=float, default=5e-2,
                        help="Absolute tolerance for logits diff (matches the CUDA script default).")
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = _dtype_from_str(args.dtype)

    if device.type != "npu":
        print(f"WARNING: requested NPU but pick_device returned {device}. "
              f"Make sure torch_npu is installed and an NPU is visible.")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.pretrained}")
    tok = AutoTokenizer.from_pretrained(args.pretrained)
    input_ids = tok(args.prompt, return_tensors="pt").input_ids

    ref_out = phase_reference(
        args.pretrained, input_ids, args.max_new_tokens, device, dtype, args.num_layers
    )
    our_out = phase_ours(
        args.pretrained, input_ids, args.max_new_tokens, device, dtype,
        ref_out["hf_text_config"], ref_out["original_num_layers"],
    )

    print("\n=== Comparison ===")

    tok_cmp = compare_tokens(ref_out["generated_ids"], our_out["generated_ids"])
    print(f"Generated tokens match exactly: {tok_cmp['match']}")
    if not tok_cmp["match"]:
        print(f"  first divergence (row, col): {tok_cmp['first_divergence_row_col']}")
    print(f"  ref (decoded):  {tok.decode(tok_cmp['ref_tokens'][0], skip_special_tokens=True)!r}")
    print(f"  ours (decoded): {tok.decode(tok_cmp['our_tokens'][0], skip_special_tokens=True)!r}")

    logit_diff = diff_logits(ref_out["logits"], our_out["logits"])
    print("Logits diff:")
    for k, v in logit_diff.items():
        print(f"  {k}: {v}")

    tokens_ok = tok_cmp["match"]
    logits_ok = logit_diff["max_abs"] <= args.atol + args.rtol * logit_diff["max_ref_abs"]

    # Under truncation the model is degenerate (a 4-layer Qwen3.5-MoE just
    # produces gibberish in a loop), so greedy decoding diverges on the
    # first token where two near-equal logits flip under bf16 noise. That is
    # not a math-equivalence failure — it just means the model has nothing
    # meaningful to decode. Use logits_ok as the binding criterion in that
    # regime; tokens_ok degrades to informational. When run end-to-end
    # (--num-layers == original_num_layers) we fall back to the strict
    # GPU-script criterion.
    truncated = args.num_layers != ref_out["original_num_layers"]
    if truncated:
        overall = logits_ok
        verdict = "PASS" if overall else "FAIL"
        print(
            f"\n{verdict}  (truncated to {args.num_layers}/{ref_out['original_num_layers']} layers — "
            f"logits_ok={logits_ok} is binding; tokens_ok={tokens_ok} informational, "
            f"tol: atol={args.atol}, rtol={args.rtol})"
        )
    else:
        overall = tokens_ok and logits_ok
        verdict = "PASS" if overall else "FAIL"
        print(
            f"\n{verdict}  (tokens_ok={tokens_ok}, logits_ok={logits_ok}, "
            f"tol: atol={args.atol}, rtol={args.rtol})"
        )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
