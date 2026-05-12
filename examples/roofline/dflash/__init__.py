"""DFlash speculative-decoding draft-model roofline examples + adjacent scripts.

A DFlash draft (per https://huggingface.co/collections/z-lab/dflash) is a
small companion model invoked B times per main-model step to propose B
speculative tokens; the main then verifies them in one batched forward.
Total wall-clock per accepted main-token is roughly::

    B * t_draft_decode + t_main_verify
    -------------------------------------   /  expected_accepts
                  expected_accepts

so a *good* draft is one whose ``B * t_draft_decode`` is much less than
``t_main_verify`` AND whose acceptance rate stays high enough to amortise.

The examples in this directory model both halves of that equation by
running the roofline on draft configs at the same shapes (decode at long
kv_cache_len typically dominates) and reporting the per-forward time
alongside main-model numbers.

z-lab's observed pattern: drafts are vanilla Qwen3 dense (no MoE / MLA /
linear-attention / CSA), 4-10 layers, hidden_size matches main, GQA
4 or 8 KV heads, vocab matches main. The "hybrid" variants here
deliberately violate that pattern by retaining main's MLA + CSA/HCA
attention machinery (dense FFN to avoid MoE overhead) — useful for
asking whether the standard Qwen3-dense convention is actually optimal
at very long context (where MLA's tiny KV cache and CSA's sparse
attention may pay back the implementation complexity).
"""
