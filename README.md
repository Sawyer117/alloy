<div align="center">

# 🧪 Alloy

**A HuggingFace-native library for composing hybrid transformer architectures, layer by layer.**

Mix grouped-query attention, gated DeltaNet, MoE, and dense MLP in arbitrary proportions — the architecture is fully described by two ordered lists, `layer_types` and `ffn_types`, no forking or `modeling_*.py` rewrites required.

[Updates](#-updates) • [Why Alloy](#-why-alloy) • [How it works](#-how-it-works) • [Quick start](#-quick-start) • [Built-in modules](#-built-in-modules) • [Testing](#-testing) • [Roadmap](#-roadmap)

</div>

---

## ✨ Updates

- **[2026-04-27]** 🪄 NPU edition of the pretrained-weight comparison scripts — `torch_npu` + `transfer_to_npu` wired in, qwen3.5-MoE script supports `--num-layers N` truncation for memory-constrained Ascend cards.
- **[2026-04-27]** 🏷️ Source-coupled module naming. Registry keys now carry their upstream model in the name (`qwen3_mlp`, `qwen3_attention`, `qwen3_5_moe`, `qwen3_5_gdn`) so a future `bert_mlp` (GELU, no gate) can coexist with the qwen3-derived SwiGLU without colliding. Mask routing moves to a declared `mask_kind` on `MixerEntry` so adding a new mixer no longer requires editing `modeling_alloy.py`.
- **[2026-04-24]** 🧬 GDN + MoE hybrid config example landed for Qwen3.5-35B-A3B.
- **[2026-04-24]** 📦 Loading helpers (`build_skeleton`, `build_on_device`, `load_state_dict_from_disk`, `strip_language_model_prefix`) promoted to public alloy API; HF MoE expert dispatch routed through `transformers.integrations.moe`.
- **[2026-04-24]** 🚀 First end-to-end Qwen3 / Qwen3.5-MoE equivalence demos. Under fp32 eager, small-scale random-init `AlloyForCausalLM` matches `Qwen3ForCausalLM` with `max_abs = 0.0` — bit-exact.

## 💡 Why Alloy

Modern language-model research increasingly explores *hybrid* architectures — alternating softmax attention with linear-time token mixers (DeltaNet, Mamba, RWKV), interleaving dense and MoE layers, tuning attention configuration per depth. Today, each new recipe either requires forking a heavyweight trainer (torchtitan, Megatron-Core) or shipping bespoke `modeling_*.py` files that fragment HuggingFace interoperability.

Alloy takes a different stance:

- 🤗 **HuggingFace-native end-to-end.** Models are plain `PreTrainedModel` subclasses backed by `PretrainedConfig`. They integrate with `from_pretrained`, `save_pretrained`, `generate`, Trainer, Accelerate, PEFT, and any FSDP2-based backend out of the box.
- 🧩 **Composition by config, not by code.** A model's architecture is fully determined by two ordered lists — `layer_types` (token mixer per depth) and `ffn_types` (feed-forward per depth). Changing the architectural mix is a JSON edit.
- 🔌 **Extension without forking.** New token mixers and feed-forward blocks register themselves through a lightweight decorator. The core decoder layer stays agnostic of which mixer you plug in.
- 🖥️ **Cross-hardware by design.** Model code carries no `torch_npu` / CUDA-specific imports. Hardware-specific fused kernels are layered on as opt-in runtime patches, so the same model definition runs unchanged on GPU and NPU.

Target use cases: architecture research (systematic layer-mix ablations, novel attention variants), hybrid-architecture pretraining and fine-tuning, and portable model definitions that survive the move between GPU training and NPU-optimized deployment.

## 🧱 How it works

The core abstraction is a decoder layer whose components are looked up in a registry at construction time:

```python
class AlloyDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        mixer_entry = get_mixer(config.layer_types[layer_idx])
        setattr(self, mixer_entry.attr_name, mixer_entry.cls(config, layer_idx))
        self.mlp = get_ffn(config.ffn_types[layer_idx])(config, layer_idx)
        ...
```

- `config.layer_types[i]` names the token mixer at layer `i`; `config.ffn_types[i]` names the feed-forward block.
- `MIXER_REGISTRY` and `FFN_REGISTRY` (in `alloy.modules.registry`) map those strings to `nn.Module` classes. Each mixer entry also declares a `mask_kind` (`"causal"`, `"sliding"`, or `"linear"`) that the model-level mask precompute uses to dispatch the right mask family — adding a new mixer never requires editing `modeling_alloy.py`.
- Sub-module attribute names (`self_attn`, `linear_attn`, `mlp`) match the conventions HuggingFace reference implementations use, so checkpoint `state_dict` keys line up without rewriting.
- Each mixer reads only the config fields it cares about; the config is a single flat bag of hyperparameters covering every registered module.
- The forward path pre-computes all needed masks once at the model level (causal, sliding-window, linear) and passes everything downward as `**kwargs`; each mixer picks what it needs and ignores the rest.

## 📁 Project layout

```
alloy/
├── configuration_alloy.py            # AlloyConfig(PretrainedConfig)
├── modeling_alloy.py                 # AlloyDecoderLayer, AlloyModel, AlloyForCausalLM
├── loading.py                        # build_skeleton / build_on_device / load_state_dict_from_disk
├── modules/
│   ├── registry.py                   # MIXER_REGISTRY, FFN_REGISTRY, decorators (with mask_kind)
│   ├── attention/
│   │   ├── gqa.py                    # "qwen3_attention", "qwen3_attention_sliding"  → Qwen3Attention
│   │   └── gdn.py                    # "qwen3_5_gdn"                                 → Qwen35GatedDeltaNet
│   ├── ffn/
│   │   ├── mlp.py                    # "qwen3_mlp"                                   → Qwen3MLP (SwiGLU)
│   │   └── moe.py                    # "qwen3_5_moe"                                 → Qwen35SparseMoE
│   └── shared/
│       ├── norm.py                   # RMSNorm with unit_offset flag
│       ├── rotary.py                 # RoPE with partial + interleaved-mRoPE support
│       └── attention_kernels.py      # eager_attention_forward, repeat_kv
├── examples/
│   ├── build_from_config.py          # JSON-driven model build + load + generate
│   ├── build_from_python.py          # Programmatic model build + load + generate
│   └── configs/
│       ├── qwen3_4b.json
│       └── qwen3_5_35b_a3b.json
├── scripts/
│   ├── compare_qwen3.py              # Small-scale Qwen3 equivalence demo
│   └── compare_qwen3_5.py            # Small-scale Qwen3.5-MoE equivalence demo
└── tests/
    ├── _compare_utils.py             # Shared helpers (device/cache/streaming/diff)
    ├── test_construct.py             # Hardware-agnostic smoke test
    ├── gpu/                          # CUDA pretrained-weight comparisons
    │   ├── compare_qwen3_pretrained.py
    │   └── compare_qwen3_5_pretrained.py
    └── npu/                          # Ascend NPU comparisons (torch_npu + transfer_to_npu)
        ├── compare_qwen3_pretrained.py
        └── compare_qwen3_5_pretrained.py
```

## ⚡ Quick start

### Installation

```bash
pip install torch transformers safetensors

git clone https://github.com/Sawyer117/alloy
# Put the parent directory on PYTHONPATH so `import alloy` works.
```

### Build a hybrid model

A 16-layer model alternating linear attention and full attention 3-to-1, with dense SwiGLU FFNs throughout:

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

Two end-to-end examples live under [`alloy/examples/`](alloy/examples/) — one driven by a JSON config, one constructed programmatically. Both load real Qwen3 / Qwen3.5-MoE weights and run greedy generation.

### Register a custom mixer

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

`mask_kind` declares which mask the model-level precompute hands this layer (`"causal"`, `"sliding"`, or `"linear"`) — pick whichever matches your forward's expectations. Naming convention: `<source_model>_<kind>` when the implementation is materially copied from a specific upstream model. Alloy's core never knew this mixer existed, and no PR to the library is required to use it.

## 🔌 Built-in modules

| Registry key                | Class                  | Source            | mask_kind  | Notes                                                                            |
|-----------------------------|------------------------|-------------------|------------|----------------------------------------------------------------------------------|
| `qwen3_attention`           | `Qwen3Attention`       | qwen3 / qwen3.5   | `causal`   | Softmax GQA. `attn_output_gate=True` reproduces qwen3.5 gated output projection. |
| `qwen3_attention_sliding`   | `Qwen3Attention`       | qwen3 / qwen3.5   | `sliding`  | Same class, `config.sliding_window` controls window mask at the model level.     |
| `qwen3_5_gdn`               | `Qwen35GatedDeltaNet`  | qwen3.5           | `linear`   | Chunked + recurrent delta-rule kernels, torch fallback when `fla` unavailable.   |
| `qwen3_mlp`                 | `Qwen3MLP`             | qwen3 / qwen3.5   | n/a        | SwiGLU MLP matching qwen3 / llama parameter names.                               |
| `qwen3_5_moe`               | `Qwen35SparseMoE`      | qwen3.5           | n/a        | Top-K router, grouped experts (3D weight tensors), gated shared expert.          |

Shared primitives used across mixers:

- **`RMSNorm`** — parameterized to cover both qwen3 (`w * x`, `ones_` init) and qwen3.5 (`(1 + w) * x`, `zeros_` init) styles via an `unit_offset` flag.
- **`RotaryEmbedding`** — supports partial rotary (`partial_rotary_factor`) and interleaved mRoPE (`mrope_section`), covering qwen3 and qwen3.5 RoPE conventions from a single class.
- **`eager_attention_forward`** — HF-style eager kernel used as fallback; `ALL_ATTENTION_FUNCTIONS.get_interface` routes SDPA / flash-attention automatically when `config._attn_implementation` is set accordingly.

## 🔁 Loading HuggingFace checkpoints

Parameter names match HuggingFace qwen3 / qwen3.5 exactly (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`, `gate_proj`, `up_proj`, `down_proj`, `conv1d`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`, `A_log`, `dt_bias`, `norm`, `experts.gate_up_proj`, `experts.down_proj`, `shared_expert.*`, `shared_expert_gate`). HuggingFace Hub checkpoints load with a plain `load_state_dict(strict=False)` given the right config flags:

| Target checkpoint   | Key config flags                                                                                       |
|---------------------|--------------------------------------------------------------------------------------------------------|
| qwen3 family        | `attn_output_gate=False`, `rms_norm_unit_offset=False`, `ffn_types=["qwen3_mlp"]*N`                    |
| qwen3.5-MoE family  | `attn_output_gate=True`, `rms_norm_unit_offset=True`, `ffn_types=["qwen3_5_moe"]*N`, partial + mRoPE   |

HF's canonical `layer_types` strings (`"full_attention"`, `"sliding_attention"`, `"linear_attention"`) translate via `alloy.configuration_alloy.hf_layer_types_to_alloy` into alloy's vocabulary. The helpers `alloy_config_from_qwen3` and `alloy_config_from_qwen3_5_text` (in [`alloy/tests/_compare_utils.py`](alloy/tests/_compare_utils.py)) wrap the full HF→alloy config translation.

## 🖥️ Hardware abstraction

Model code in `modeling_alloy.py` and `modules/**` has zero hardware-specific imports. Hardware optimizations are layered on top as opt-in patches:

```python
# (planned) alloy.npu_patch
from alloy.npu_patch import transform_for_npu

model = AlloyForCausalLM(config)
if device.type == "npu":
    transform_for_npu(model)   # swap RMSNorm / attention / rope / fused kernels
```

The same `AlloyForCausalLM(config)` runs on GPU (no patch, pure PyTorch) and on NPU (with patch, calling `torch_npu.npu_fusion_attention`, `torch_npu.npu_rms_norm`, etc.). Model definitions stay portable and research stays reproducible across hardware.

## 🧪 Testing

```bash
# Hardware-agnostic construction + forward-shape smoke test
python -m alloy.tests.test_construct

# CUDA: load real pretrained weights, diff forward + greedy tokens against HF
python -m alloy.tests.gpu.compare_qwen3_pretrained --pretrained Qwen/Qwen3-4B --dtype bf16
python -m alloy.tests.gpu.compare_qwen3_5_pretrained --pretrained /path/to/Qwen3.5-35B-A3B --dtype bf16

# Ascend NPU: same protocol, requires torch_npu installed
python -m alloy.tests.npu.compare_qwen3_pretrained --pretrained Qwen/Qwen3-4B --dtype bf16
python -m alloy.tests.npu.compare_qwen3_5_pretrained --pretrained /path/to/Qwen3.5-35B-A3B --dtype bf16 --num-layers 4
```

The pretrained-weight tests (CUDA and NPU) follow the same sequential protocol to avoid holding two copies of weights simultaneously:

1. Load the HF reference model, run forward + greedy generation on a fixed prompt, save CPU logits and generated token ids, release the model from HBM.
2. Construct the equivalent `AlloyForCausalLM`, stream the state dict directly from the on-disk safetensors shards, run the identical forward + generation.
3. Compare. Generated token ids must match exactly (`torch.equal`); logits diff statistics (`max_abs`, `mean_abs`, `relative_max`) provide a finer diagnostic.

The NPU scripts add two top-level imports — `import torch_npu` and `from torch_npu.contrib import transfer_to_npu` — so any HF-internal `torch.cuda.*` / `.cuda()` call is monkey-patched to the NPU backend. Both reference and ours run on `attn_implementation="eager"`, so the only place math can drift is the elementwise + matmul kernels themselves; default tolerances on NPU therefore match the CUDA script.

The qwen3.5-MoE NPU script additionally takes `--num-layers N` (default 4) and truncates *both* the HF reference and the alloy model to their first N layers, loading only those layers' weights from the on-disk safetensors. Qwen3.5-35B-A3B in bf16 is ~70 GB and does not fit on a single Ascend 910B-class card; the truncated comparison still exercises GDN, gated GQA, the shared expert, the top-K router, the grouped experts, partial RoPE, and unit-offset RMSNorm, because the first 4 layers cover the canonical 3:1 GDN:full-attention pattern.

Under fp32 eager mode the small-scale random-init comparison against `Qwen3ForCausalLM` produces `max_abs = 0.0` — bit-exact equivalence.

## 🚧 Known limitations

- **Incremental decoding for linear attention.** `Qwen35GatedDeltaNet` expects a `HybridCache` exposing `update_conv_state` / `update_recurrent_state`. Generation with `DynamicCache` falls back to full re-forward per new token (`use_cache=False`). A proper hybrid-cache implementation is on the roadmap.
- **NPU patch layer.** `alloy.npu_patch` is scoped and not yet implemented in this initial scaffolding.
- **MoE expert dispatch.** The fallback eager path is a batched-bmm forward; on `transformers` v5 the class is wrapped by `@use_experts_implementation` and routes through `grouped_mm` / `batched_mm` automatically. The fallback is correct but not tuned for very large expert counts.

## 🗺️ Roadmap

- `HybridCache` for incremental generation with heterogeneous mixer types.
- `alloy.npu_patch`: runtime patch set that swaps `RMSNorm`, `apply_rotary_pos_emb`, softmax attention, and grouped expert paths for `torch_npu` fused kernels.
- Additional registered mixers: Mamba2, MLA (Multi-head Latent Attention), RWKV variants.
- Integration recipes for MindSpeed-LLM's FSDP2 training path (native HF `AutoModelForCausalLM` consumer).
- Sliding-window KV cache path for long-context training / inference.

## 🤝 Acknowledgements

The `Qwen3Attention`, `Qwen35GatedDeltaNet`, and `Qwen35SparseMoE` implementations are ported from HuggingFace `transformers` (`modeling_qwen3.py`, `modeling_qwen3_5_moe.py`), preserving math and parameter names so upstream checkpoints load without modification. The registry-based decoder-layer pattern is inspired by HuggingFace's canonical hybrid decoder layout, and the README structure follows the lead of [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention).

## 📄 License

License is not yet chosen. Treat the code as source-available pending a formal decision.
