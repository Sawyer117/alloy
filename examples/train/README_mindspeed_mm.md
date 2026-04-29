# Training alloy under MindSpeed-MM (FSDP2)

MindSpeed-MM has a clean plugin system that loads any HuggingFace-native
model as long as `model_id` resolves through `model_register` and the model
directory is loadable via `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
Alloy meets both. This doc walks through the workflow end-to-end.

## What lives where

| File | Role |
|---|---|
| `alloy/integrations/mindspeed_mm.py` | Plugin shim — registers `AlloyForCausalLM` under `model_id="alloy"` and forwards `_<module>_implementation` yaml fields onto the HF config |
| `alloy/integrations/hf_npu_binder.py` | Binder bridge — registers `triton` / `flash` fast-path kernels into alloy + HF dispatch tables (optional but recommended on NPU) |
| `alloy/tools/export_for_hub.py` | Produces the `model_name_or_path` directory: `config.json` + 1-line `modeling_alloy.py` shim + `auto_map` |
| `examples/train/pretrain_alloy_qwen3_5_moe_mindspeed.yaml` | Example config for qwen3.5-MoE-style alloy (GDN + MoE) |
| `examples/train/pretrain_alloy_qwen3_dense_mindspeed.yaml` | Example config for qwen3-style dense alloy |

## End-to-end workflow

### 1. Freeze the model architecture into a Hub-style directory

```python
# Once: produce ./hf_models/alloy_qwen3_5_moe/
from alloy import AlloyConfig
from alloy.tools.export_for_hub import export_for_hub

cfg = AlloyConfig(
    vocab_size=151936,
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=2,
    head_dim=128,
    layer_types=["qwen3_5_gdn"] * 18 + ["qwen3_attention"] * 6,    # canonical 3:1
    ffn_types=["qwen3_5_moe"] * 24,
    num_experts=64,
    num_experts_per_tok=8,
    moe_intermediate_size=512,
    shared_expert_intermediate_size=512,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
    rms_norm_unit_offset=True,
    attn_output_gate=True,
)
export_for_hub(cfg, "./hf_models/alloy_qwen3_5_moe", tokenizer_src="./qwen3_5_tokenizer/")
```

After this step `./hf_models/alloy_qwen3_5_moe/` contains:
- `config.json` — alloy's PretrainedConfig dump (no `_*` runtime fields)
- `modeling_alloy.py` — 1-line shim importing `AlloyForCausalLM`
- tokenizer files (copied from `tokenizer_src`)

### 2. Install alloy + binder editable into the training env

On the NPU machine:

```bash
pip install -e /path/to/alloy
pip install -e /path/to/hf-npu-binder
```

`mindspeed_mm` and its deps come from the bytedance-ms-mm conda env that
already has `torch_npu` / `triton` / `transformers` configured.

### 3. Pick a yaml, set the data path, run

```bash
# Copy one of the templates into your scripts dir
cp /path/to/alloy/examples/train/pretrain_alloy_qwen3_5_moe_mindspeed.yaml \
   /path/to/qwen3.5_omni_creative/scripts_qwen3_5/configs/my_alloy_run.yaml

# Edit my_alloy_run.yaml: data.basic_parameters.dataset_dir / dataset / save / etc.
# Then launch via the standard mindspeed-mm trainer:
torchrun \
    --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=6000 \
    mindspeed_mm/fsdp/train/trainer.py \
    scripts_qwen3_5/configs/my_alloy_run.yaml
```

(Use the existing `pretrain-exp4_qwen3_5_10b-a2b_optimized.sh` as a launcher
template — same env-setup, just point it at the yaml above.)

## What yaml controls (and how it reaches alloy)

| yaml field | Reaches alloy via | Meaning |
|---|---|---|
| `model.model_id: alloy` | `model_register.get("alloy")` → `AlloyForCausalLMPlugin` | which model class to build |
| `model.model_name_or_path` | `AutoConfig.from_pretrained(...)` | where `config.json` + `modeling_alloy.py` live |
| `model.trust_remote_code: true` | `AutoConfig.from_pretrained(..., trust_remote_code=True)` | required to load `auto_map` |
| `model.attn_implementation` | `AutoConfig.from_pretrained(_attn_implementation=...)` | HF standard attn dispatch (`eager` / `sdpa` / `flash_attention_2`) |
| `model._qwen3_5_gdn_implementation` | `AlloyForCausalLMPlugin.overwrite_transformer_config` → `transformer_config._qwen3_5_gdn_implementation` | alloy GDN backend (`torch` / `triton` / `flash`) |
| `model._experts_implementation` | same | HF / alloy MoE dispatch (`eager` / `grouped_mm` / `batched_mm` / `flash`) |
| `parallel.fsdp_plan.apply_modules` | mindspeed FSDP wrapping | which alloy submodules get sharded |
| `parallel.ep_plan.apply_modules` | mindspeed EP wrapping | which alloy submodules participate in expert parallel |

The plugin shim's pattern for `_<key>_implementation` means you can **drop new
alloy modules with their own dispatch surfaces in alloy core, then expose
them in yaml without editing the shim**.

## Switching backends without retraining

Once the model is frozen in `./hf_models/alloy_qwen3_5_moe/`:

```yaml
# binder OFF (alloy default torch dispatch — byte-exact baseline)
model:
  _qwen3_5_gdn_implementation: torch
  _experts_implementation: eager

# binder ON (NPU fast path)
model:
  _qwen3_5_gdn_implementation: flash      # GDN fused: ascendc + triton
  _experts_implementation: flash          # MoE fused: permute + GMM + swiglu + GMM + unpermute

# pure triton GDN (fallback when ascendc kernels unavailable)
model:
  _qwen3_5_gdn_implementation: triton
  _experts_implementation: eager
```

These are pure yaml knobs. **No alloy code edits, no model dir regeneration.**
That's the whole point of having alloy's `_<module>_implementation` fields
runtime-only and never serialised into `config.json`.

## FSDP plan caveats

Alloy's module paths differ from `Qwen3_5MoeForConditionalGeneration` (which
wraps a vision encoder + a language_model). Alloy is text-only, so:

- `model.layers.{*}` — not `model.language_model.layers.{*}`
- `model.layers.{*}.linear_attn` — alloy's GDN attribute name (matches the
  registry's `attr_name`)
- `model.layers.{*}.self_attn` — full-attention layers
- `model.layers.{*}.mlp` — FFN (dense MLP or `Qwen35SparseMoE` depending on
  layer's `ffn_types[i]`)
- `model.layers.{*}.mlp.experts` — only on MoE layers
- `model.embed_tokens`, `model.norm`, `lm_head` — top-level

The `hook_modules` and `recompute_plan.apply_modules` ranges
(`model.layers.{0-N}`) must match `num_hidden_layers` in your
`./hf_models/alloy_*/config.json`.

## Troubleshooting

**`KeyError: 'alloy' is not registered`** — the plugin import never ran.
Check `training.plugin` includes `alloy.integrations.mindspeed_mm` and
that alloy is `pip install -e`'d into the env.

**`ValueError: model_id 'alloy' is not registered in MODEL_MAPPINGS`** — same
cause, plugin not imported. Verify by running

```bash
python -c "from alloy.integrations.mindspeed_mm import AlloyForCausalLMPlugin; print('OK')"
```

**`KeyError: 'flash' is not a valid experts implementation`** — bytedance fork's
`ExpertsInterface.get_interface` is strict; the binder bridge needs to import
*before* the model is built. Add `alloy.integrations.hf_npu_binder` to the
yaml's `training.plugin` list (mindspeed imports plugins in order during
`Trainer.initialize()`).

**Alloy module paths don't match `apply_modules` patterns** — verify the
patterns against an actual constructed model:

```python
from alloy import AlloyConfig, AlloyForCausalLM
cfg = AlloyConfig.from_pretrained("./hf_models/your_dir", trust_remote_code=True)
m = AlloyForCausalLM(cfg)
for name, _ in m.named_modules():
    print(name)
```
