# DeepSeek-V4 reference (HF transformers)

Verbatim copy of `transformers/src/transformers/models/deepseek_v4/` from
HF main, kept inside alloy's tree for offline reference while porting
DSV4 modules. Read-only mirror — do not edit. When upstream changes
materially, refresh by re-copying from the local transformers checkout
(`D:/work/transformers/...` on the dev machine).

## Files

- `configuration_deepseek_v4.py` — `DeepseekV4Config`. Defines the layer
  vocabulary (`heavily_compressed_attention`, `compressed_sparse_attention`,
  `sliding_attention`), `compress_rates`, `hc_mult` (MHC streams),
  `mlp_layer_types` (`moe` / `hash_moe`), and the legacy fold-in for
  earlier per-layer integer compress ratios.
- `modeling_deepseek_v4.py` — Full modeling: HCA/CSA caches, compressors,
  indexer, attention (LoRA Q + grouped output proj + sinks + partial RoPE
  + conjugate undo), MoE block (top-k + hash routers, 3D experts), MHC
  (HyperConnection + HyperHead), decoder layer.
- `modular_deepseek_v4.py` — HF's "modular" source-of-truth file from
  which `modeling_deepseek_v4.py` is auto-generated. Useful when you want
  to trace which class is inherited vs locally overridden.

## How alloy port lines up

| HF class | alloy destination |
|---|---|
| `DeepseekV4RMSNorm`, `DeepseekV4UnweightedRMSNorm` | `alloy/modules/shared/norm.py` |
| `DeepseekV4RotaryEmbedding` | `alloy/modules/shared/rotary.py` |
| `DeepseekV4HCACache`, `DeepseekV4CSACache` | `alloy/modules/attention/dsv4_attention.py` (co-located with the mixer that uses them) |
| `DeepseekV4HCACompressor`, `DeepseekV4CSACompressor`, `DeepseekV4Indexer`, `DeepseekV4GroupedLinear`, `DeepseekV4Attention` | `alloy/modules/attention/dsv4_attention.py` |
| `DeepseekV4MLP`, `DeepseekV4Experts`, `DeepseekV4TopKRouter`, `DeepseekV4HashRouter`, `DeepseekV4SparseMoeBlock` | `alloy/modules/ffn/dsv4_moe.py` |
| `DeepseekV4HyperConnection`, `DeepseekV4HyperHead` | `alloy/modeling_alloy.py` (MHC variant — phase 5, behind `config.use_mhc`) |
| `DeepseekV4DecoderLayer` | covered by `AlloyDecoderLayer` (single-stream, MHC off) or `AlloyMhcDecoderLayer` (multi-stream, MHC on) |
| `DeepseekV4Model`, `DeepseekV4ForCausalLM` | covered by `AlloyModel` / `AlloyForCausalLM` |

## Porting conventions (reminder)

- `register_mixer("dsv4_<name>", attr_name="self_attn")` — co-locate cache
  class + mixer in the same `.py` (the user's call). No separate cache
  framework; if HF DynamicCache's dispatch table doesn't recognize DSV4
  layer types, patch from the DSV4 module's import-time side effect.
- `HF_LAYER_TYPE_TO_ALLOY` adds entries for DSV4's canonical layer type
  strings → alloy source-coupled keys.
- Each ported class gets a docstring "ported from
  references/dsv4/modeling_deepseek_v4.py:LINE" so future readers can
  bounce back to the original.
