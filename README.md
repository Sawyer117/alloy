# Alloy

A HuggingFace-native library for composing hybrid transformer architectures layer by layer. Mix grouped-query attention, gated DeltaNet, MoE, and more in arbitrary proportions — defined by a single `layer_types` list.

Modules carry a source-coupled name (`qwen3_attention`, `qwen3_mlp`, `qwen3_5_moe`, `qwen3_5_gdn`, ...) so that variants ported from different upstream models — e.g. a `qwen3_mlp` (SwiGLU) and a future `bert_mlp` (GELU, no gate) — can coexist without colliding.

## Vision

Modern language model research increasingly explores *hybrid* architectures: alternating softmax attention with linear-time token mixers (DeltaNet, Mamba, RWKV), interleaving dense and MoE layers, and tuning attention configuration per depth. Today, each new recipe either requires forking a heavyweight trainer (torchtitan, Megatron-Core) or shipping bespoke `modeling_*.py` files that fragment HuggingFace interoperability.

Alloy takes a different stance on these three axes:

1. **HuggingFace-native model definitions.** Models are plain `PreTrainedModel` subclasses backed by `PretrainedConfig`. They integrate with `from_pretrained`, `save_pretrained`, `generate`, Trainer, Accelerate, PEFT, and any FSDP2-based backend out of the box.
2. **Composition by config, not by code.** A model's architecture is fully determined by two ordered lists — `layer_types` (the token mixer at each depth) and `ffn_types` (the feed-forward block at each depth). Changing the architectural mix is a JSON edit.
3. **Extension without forking.** New token mixers and feed-forward blocks register themselves through a lightweight decorator. The core decoder layer stays agnostic of which mixer you plug in.
4. **Cross-hardware by design.** Model code carries no `torch_npu` / CUDA-specific imports. Hardware-specific fused kernels are applied as opt-in runtime patches, so the same model definition runs unchanged on GPU and NPU.

The target use cases are architecture research (systematic layer-mix ablations, novel attention variants), hybrid-architecture pretraining and fine-tuning, and portable model definitions that survive the move between GPU training and NPU-optimized deployment.

## Design Overview

The core abstraction is a decoder layer whose components are looked up in a registry at construction time:

```python
class AlloyDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        mixer_entry = get_mixer(config.layer_types[layer_idx])
        setattr(self, mixer_entry.attr_name, mixer_entry.cls(config, layer_idx))
        self.mlp = get_ffn(config.ffn_types[layer_idx])(config, layer_idx)
        ...
```

- `config.layer_types` is a `list[str]` of length `num_hidden_layers`, one string per layer naming the token mixer.
- `config.ffn_types` is a `list[str]` of the same length, one string per layer naming the feed-forward block.
- `MIXER_REGISTRY` and `FFN_REGISTRY` (defined in `alloy.modules.registry`) map those strings to `nn.Module` classes. Each mixer entry also declares a `mask_kind` (`"causal"`, `"sliding"`, or `"linear"`) that the model-level mask precompute uses to dispatch the right mask family — so adding a new mixer never requires editing `modeling_alloy.py`.
- Sub-module attribute names (`self_attn`, `linear_attn`, `mlp`) match the conventions used by HuggingFace reference implementations, so checkpoint `state_dict` keys line up without rewriting.
- Each mixer reads only the config fields it cares about; the config is a single flat bag of hyperparameters covering every registered module.

The forward path pre-computes all needed causal masks (full, sliding-window, linear-attention) once at the model level and passes everything downward as `**kwargs`; each mixer picks what it needs and ignores the rest.

## Project Structure

```
alloy/
├── configuration_alloy.py            # AlloyConfig(PretrainedConfig)
├── modeling_alloy.py                 # AlloyDecoderLayer, AlloyModel, AlloyForCausalLM
├── modules/
│   ├── registry.py                   # MIXER_REGISTRY, FFN_REGISTRY, decorators (with mask_kind)
│   ├── attention/
│   │   ├── gqa.py                    # "qwen3_attention", "qwen3_attention_sliding" (Qwen3Attention)
│   │   └── gdn.py                    # "qwen3_5_gdn" (Qwen35GatedDeltaNet)
│   ├── ffn/
│   │   ├── mlp.py                    # "qwen3_mlp" (Qwen3MLP, SwiGLU)
│   │   └── moe.py                    # "qwen3_5_moe" (Qwen35SparseMoE: top-K router + grouped + shared experts)
│   └── shared/
│       ├── norm.py                   # RMSNorm with unit_offset flag
│       ├── rotary.py                 # RoPE with partial + interleaved-mRoPE support
│       └── attention_kernels.py      # eager_attention_forward, repeat_kv
├── scripts/
│   ├── compare_qwen3.py              # Small-scale Qwen3 equivalence demo
│   └── compare_qwen3_5.py            # Small-scale Qwen3.5-MoE equivalence demo
└── tests/
    ├── _compare_utils.py             # Shared helpers (device/cache/streaming/diff)
    ├── test_construct.py             # Hardware-agnostic smoke test
    ├── gpu/                          # Full pretrained-weight comparisons
    │   ├── compare_qwen3_pretrained.py
    │   └── compare_qwen3_5_pretrained.py
    └── npu/                          # Memory-constrained comparisons (planned)
```

## Quick Start

### Installation

```bash
pip install torch transformers safetensors

git clone https://github.com/Sawyer117/alloy
# Put the parent directory on PYTHONPATH so `import alloy` works.
```

### Build a Hybrid Model

Four-to-one linear-to-full attention interleave with dense MLP feed-forwards:

```python
from alloy import AlloyConfig, AlloyForCausalLM

config = AlloyConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=2,
    head_dim=128,
    intermediate_size=8192,
    layer_types=[
        "qwen3_5_gdn", "qwen3_5_gdn", "qwen3_5_gdn", "qwen3_attention",
    ] * 4,
    ffn_types=["qwen3_mlp"] * 16,
)
model = AlloyForCausalLM(config)
```

Swap the FFN for MoE by changing one field:

```python
config.ffn_types = ["qwen3_5_moe"] * 16
config.num_experts = 128
config.num_experts_per_tok = 8
```

### Register a Custom Mixer

Drop a file somewhere importable and decorate a module class:

```python
from alloy.modules.registry import register_mixer

@register_mixer("my_rwkv", attr_name="linear_attn", mask_kind="linear")
class MyRWKV(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        ...

    def forward(self, hidden_states, past_key_values=None, **kwargs):
        ...
```

Then reference it by string in the config:

```python
config = AlloyConfig(..., layer_types=["my_rwkv"] * 8 + ["qwen3_attention"] * 8, ...)
```

`mask_kind` tells the model-level mask precompute which mask to hand this layer (`"causal"`, `"sliding"`, or `"linear"`) — pick whichever matches your forward's expectations. Naming convention: `<source_model>_<kind>` when the implementation is materially copied from a specific upstream model.

Alloy's core never knew this mixer existed, and no PR to the library is required to use it.

## Built-in Modules

| Registry key                | Class                  | Source     | mask_kind | Notes                                                                            |
|-----------------------------|------------------------|------------|-----------|----------------------------------------------------------------------------------|
| `qwen3_attention`           | `Qwen3Attention`       | qwen3 / qwen3.5 | `causal`  | Softmax GQA. `attn_output_gate=True` reproduces qwen3.5 gated output projection. |
| `qwen3_attention_sliding`   | `Qwen3Attention`       | qwen3 / qwen3.5 | `sliding` | Same class, `config.sliding_window` controls window mask at the model level.     |
| `qwen3_5_gdn`               | `Qwen35GatedDeltaNet`  | qwen3.5    | `linear`  | Chunked + recurrent delta-rule kernels, torch fallback when `fla` unavailable.   |
| `qwen3_mlp`                 | `Qwen3MLP`             | qwen3 / qwen3.5 | n/a       | SwiGLU MLP matching qwen3 / llama parameter names.                               |
| `qwen3_5_moe`               | `Qwen35SparseMoE`      | qwen3.5    | n/a       | Top-K router, grouped experts (3D weight tensors), gated shared expert.          |

Shared primitives used across mixers:

- `RMSNorm` — parameterized to cover both qwen3 (`w * x`, `ones_` init) and qwen3.5 (`(1 + w) * x`, `zeros_` init) styles via an `unit_offset` flag.
- `RotaryEmbedding` — supports partial rotary (`partial_rotary_factor`) and interleaved mRoPE (`mrope_section`), covering qwen3 and qwen3.5 rope conventions from a single class.
- `eager_attention_forward` — HF-style eager kernel used as fallback; `ALL_ATTENTION_FUNCTIONS.get_interface` routes SDPA / flash-attention automatically when `config._attn_implementation` is set accordingly.

## Hardware Abstraction

Model code in `modeling_alloy.py` and `modules/**` has zero hardware-specific imports. Hardware optimizations are layered on top as opt-in patches:

```python
# (planned) alloy.npu_patch
from alloy.npu_patch import transform_for_npu

model = AlloyForCausalLM(config)
if device.type == "npu":
    transform_for_npu(model)   # swap RMSNorm / attention / rope / fused kernels
```

The same `AlloyForCausalLM(config)` runs on GPU (no patch, pure PyTorch) and NPU (with patch, calling `torch_npu.npu_fusion_attention`, `torch_npu.npu_rms_norm`, etc.). This keeps model definitions portable and research reproducible across hardware.

## Checkpoint Compatibility

Parameter names match HuggingFace qwen3 / qwen3.5 exactly (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`, `gate_proj`, `up_proj`, `down_proj`, `conv1d`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`, `A_log`, `dt_bias`, `norm`, `experts.gate_up_proj`, `experts.down_proj`, `shared_expert.*`, `shared_expert_gate`). HuggingFace Hub checkpoints load with a plain `load_state_dict(strict=False)` given the right config flags:

| Target checkpoint       | Key config flags                                                                                       |
|-------------------------|--------------------------------------------------------------------------------------------------------|
| qwen3 family            | `attn_output_gate=False`, `rms_norm_unit_offset=False`, `ffn_types=["qwen3_mlp"]*N`                    |
| qwen3.5-MoE family      | `attn_output_gate=True`, `rms_norm_unit_offset=True`, `ffn_types=["qwen3_5_moe"]*N`, partial + mRoPE  |

HF `layer_types` strings (`"full_attention"`, `"sliding_attention"`, `"linear_attention"`) translate via `alloy.configuration_alloy.hf_layer_types_to_alloy` into alloy's vocabulary (`"qwen3_attention"`, `"qwen3_attention_sliding"`, `"qwen3_5_gdn"`).

Helper functions `alloy_config_from_qwen3` and `alloy_config_from_qwen3_5_text` (in `alloy/tests/_compare_utils.py`) perform this translation automatically from a loaded HF config.

## Testing

Tests are split by hardware because NPU and GPU have fundamentally different memory budgets for this kind of validation:

```bash
# Hardware-agnostic construction + forward-shape smoke test
python -m alloy.tests.test_construct

# GPU: load real pretrained weights, diff forward + greedy tokens against HF
python -m alloy.tests.gpu.compare_qwen3_pretrained --pretrained Qwen/Qwen3-4B --dtype bf16
python -m alloy.tests.gpu.compare_qwen3_5_pretrained --pretrained /path/to/Qwen3.5-35B-A3B --dtype bf16
```

The GPU tests execute the following sequential protocol to avoid holding two copies of weights simultaneously:

1. Load the HF reference model, run forward + greedy generation on a fixed prompt, save CPU logits and generated token ids, release the model from HBM.
2. Construct the equivalent `AlloyForCausalLM`, stream the state dict directly from the on-disk safetensors shards, run the identical forward + generation.
3. Compare. Generated token ids must match exactly (`torch.equal`); logits diff statistics (max-abs, mean-abs, relative-max) provide a finer diagnostic.

Under fp32 eager mode the small-scale random-init comparison against `Qwen3ForCausalLM` produces `max_abs = 0.0` — bit-exact equivalence.

## Known Limitations

- **Incremental decoding for linear attention.** `Qwen35GatedDeltaNet` expects a `HybridCache` exposing `update_conv_state` / `update_recurrent_state`. Generation with `DynamicCache` falls back to full re-forward per new token (`use_cache=False`). A proper hybrid-cache implementation is on the roadmap.
- **NPU patch layer.** `alloy.npu_patch` is scoped and not yet implemented in this initial scaffolding.
- **MoE routing micro-optimization.** The grouped-expert forward uses a Python `for` loop over hit experts. Correct and matches the HuggingFace reference, but not tuned for very large expert counts.

## Roadmap

- `HybridCache` for incremental generation with heterogeneous mixer types.
- `alloy.npu_patch`: runtime patch set that swaps `RMSNorm`, `apply_rotary_pos_emb`, softmax attention, and grouped expert paths for `torch_npu` fused kernels.
- Additional registered mixers: Mamba2, MLA (Multi-head Latent Attention), RWKV variants.
- Integration recipes for MindSpeed-LLM's FSDP2 training path (native HF `AutoModelForCausalLM` consumer).
- Sliding-window KV cache path for long-context training / inference.

## Acknowledgements

The `Qwen3Attention`, `Qwen35GatedDeltaNet`, and `Qwen35SparseMoE` implementations are ported from HuggingFace `transformers` (`modeling_qwen3.py`, `modeling_qwen3_5_moe.py`), preserving math and parameter names so upstream checkpoints load without modification. The registry-based decoder-layer pattern is inspired by HuggingFace's canonical hybrid decoder layout.

## License

License is not yet chosen. Treat the code as source-available pending a formal decision.
