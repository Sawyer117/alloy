<div align="center">

# Alloy

**Hugging Face-native hybrid transformer composition with built-in roofline analysis.**

Build Qwen3, Qwen3.5-MoE, DeepSeek-V4, and new hybrid decoder stacks from config.
Load upstream checkpoints without state-dict rewrites. Keep hardware fast paths outside
the model core. Analyze FLOPs and HBM traffic before you train or serve.

[Why Alloy](#why-alloy) - [Support Matrix](#support-matrix) - [Install](#install) - [Quick Start](#quick-start) - [Fast Paths](#fast-paths) - [Roofline](#roofline) - [Training](#training) - [Testing](#testing)

</div>

---

## Why Alloy

Alloy is a small model-composition layer for transformer research. The design
principle is simple: the model definition should stay Hugging Face-native, while
architecture choices and hardware kernels remain swappable.

- **Architecture by config.** `layer_types[i]` chooses the token mixer at layer
  `i`; `ffn_types[i]` chooses the FFN. Switching from dense attention to Gated
  DeltaNet, sliding attention, CSA, HCA, dense MLP, or MoE is a JSON edit.
- **HF checkpoint compatibility.** Modules preserve upstream parameter names,
  cache conventions, `PretrainedConfig`, `PreTrainedModel`, `generate`,
  `save_pretrained`, Trainer, Accelerate, and FSDP2 integration points.
- **Backend-agnostic core.** `modeling_alloy.py` and `modules/**` do not import
  `torch_npu`, CUDA extensions, Triton, or AscendC. Accelerated implementations
  are opt-in through `hf-npu-binder` or other bridges.
- **Analytical performance model.** `alloy.roofline` computes theoretical FLOPs,
  HBM bytes, arithmetic intensity, and roofline time directly from config in
  O(num_layers). No model construction required.

## Support Matrix

| Family | Alloy keys | Current status | Checkpoint / API compatibility | Fast path | Roofline |
| --- | --- | --- | --- | --- | --- |
| Qwen3 dense | `qwen3_attention`, `qwen3_attention_sliding`, `qwen3_mlp` | Ready | HF key-compatible; fp32 eager equivalence tests and pretrained comparison scripts | Torch core; NPU runs through normal PyTorch dispatch | Yes |
| Qwen3.5-MoE / Qwen3-Next style | `qwen3_5_gdn`, `qwen3_attention`, `qwen3_5_moe` | Ready for torch/eager; fast-path selection is opt-in | HF key-compatible; fp32 eager `max_abs = 0.0` target in equivalence tests | `hf-npu-binder` registers GDN and MoE expert backends | Yes |
| DeepSeek-V4 attention | `dsv4_sliding_attention`, `dsv4_hca_attention`, `dsv4_csa_attention` | Active port with torch reference and NPU adapters | Mirrors HF DSV4 cache and rotary conventions; validation coverage is growing with the upstream HF surface | Triton and AscendC adapters via `hf-npu-binder`; default remains conservative | Yes |
| DeepSeek-V4 FFN / MoE | `dsv4_moe`, `dsv4_hash_moe` | Active port | HF-style router and expert parameter layout | Shared binder MoE expert path can serve compatible expert modules | Yes |
| DeepSeek-V4 MHC | `use_mhc=True` with any registered mixer/FFN | Experimental but wired end-to-end | Multi-stream residual path modeled after DSV4 HyperConnection / HyperHead | Triton HyperConnection path via binder for `hc_mult=4` | Yes |
| New modules | `register_mixer`, `register_ffn`, `register_spec` | Extension point | Caller owns checkpoint naming and tests | Register new implementations under `IMPL_REGISTRY` | Add a `RooflineSpec` |

Status notes:

- `Ready` means the in-tree torch path is intended to be usable and covered by
  construction / equivalence / serialization tests for the relevant surface.
- NPU and Triton kernels are deliberately opt-in. Alloy prefers a correct torch
  fallback over a silent fast path that changes numerics or fails on partial
  environments.
- Linear-attention incremental cache support is still a known limitation; see
  [Known Limitations](#known-limitations).

## Install

From a clone:

```bash
git clone https://github.com/Sawyer117/alloy
cd alloy
pip install -e .
```

Core dependencies are intentionally ordinary Python packages:

```bash
pip install torch transformers safetensors
```

For Ascend NPU fast paths, install `hf-npu-binder` separately and use the bridge
shown in [Fast Paths](#fast-paths). The core package imports cleanly on CPU, CUDA,
and NPU machines because hardware-specific dependencies are not imported by
Alloy itself.

## Quick Start

Build a hybrid decoder from config:

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
    layer_types=["qwen3_5_gdn", "qwen3_5_gdn", "qwen3_5_gdn", "qwen3_attention"] * 4,
    ffn_types=["qwen3_mlp"] * 16,
)
model = AlloyForCausalLM(config)
```

Load an upstream Hugging Face checkpoint when the Alloy config matches the source
model shape:

```python
from transformers import AutoModelForCausalLM

hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
alloy = AlloyForCausalLM(config)  # config must match the checkpoint architecture
missing, unexpected = alloy.load_state_dict(hf.state_dict(), strict=False)
```

For a concrete Qwen3 -> AlloyConfig mapper, see
[examples/build_from_python.py](examples/build_from_python.py).

Run one of the end-to-end demos:

```bash
python -m alloy.examples.build_from_config \
    --config examples/configs/qwen3_4b.json \
    --pretrained /path/to/Qwen3-4B \
    --dtype bf16 --max-new-tokens 32

python -m alloy.examples.build_from_python --toy
```

See [examples/README.md](examples/README.md) for JSON-driven builds, Python-driven
builds, Hub packaging, and checkpoint-loading notes.

## Composing Architectures

A decoder layer never hardcodes a particular attention or FFN class. It asks the
registry what to build for that layer index.

```python
from alloy.modules.registry import register_mixer

@register_mixer("my_linear_mixer", attr_name="linear_attn", mask_kind="linear")
class MyLinearMixer(nn.Module):
    ...

config = AlloyConfig(
    ...,
    layer_types=["my_linear_mixer"] * 8 + ["qwen3_attention"] * 8,
    ffn_types=["qwen3_mlp"] * 16,
)
```

`mask_kind` declares the model-level mask family (`causal`, `sliding`, or
`linear`), so mixed models still precompute masks once and route them by layer.
To make the new module visible to roofline analysis, register a matching
`RooflineSpec` under the same key.

## Fast Paths

Alloy owns the model contract; backend packages own kernel policy. The bridge in
`alloy.integrations.hf_npu_binder` maps binder callables into Alloy's
`IMPL_REGISTRY` and Hugging Face's MoE expert dispatch table.

```python
import alloy.integrations.hf_npu_binder as binder

chosen = binder.activate(model, prefer="auto")
print(chosen)
```

`prefer` can be a single intent (`auto`, `flash`, `triton`, `ascendc`, `torch`) or
a per-module mapping:

```python
binder.activate(model, {
    "qwen3_5_gdn": "triton",
    "experts": "flash",
    "dsv4_csa": "ascendc",
})
```

Runtime fields such as `_qwen3_5_gdn_implementation` are intentionally filtered
out of `config.json`. They select local execution policy; they are not part of
the model architecture and should not travel with Hub artifacts.

## Roofline

`alloy.roofline` walks `layer_types` and `ffn_types`, dispatches to registered
specs, and aggregates theoretical work and traffic for one forward pass.

```python
from alloy.roofline import (
    CustomHardware,
    roofline_decode,
    roofline_mini_prefill,
    roofline_prefill,
)

print(roofline_prefill(config, batch=1, seq_len=4096, hardware="H100"))

my_device = CustomHardware(
    name="my-device",
    hbm_bandwidth=8e12,
    bf16=2250e12,
    fp32=80e12,
    fp8=4500e12,
)

for hw in ["H100", "Ascend910C", my_device]:
    p = roofline_prefill(config, batch=1, seq_len=4096, hardware=hw)
    m = roofline_mini_prefill(config, batch=1, chunk_len=512, kv_cache_len=2048, hardware=hw)
    d = roofline_decode(config, batch=1, kv_cache_len=4096, hardware=hw)
    name = hw if isinstance(hw, str) else hw.name
    print(
        f"{name:11} prefill={p.roofline_time_s*1e3:6.2f} ms "
        f"mini={m.roofline_time_s*1e3:6.2f} ms "
        f"decode={d.roofline_time_s*1e6:6.0f} us"
    )
```

Built-in hardware presets include `A100`, `H100`, `Ascend910B1`, and
`Ascend910C`. `CustomHardware(...)` accepts absolute FLOP/s for dtype-specific
cube/tensor throughput (`bf16`, `fp16`, `fp8`, `fp32`, etc.) and optional
`vector_*` throughput for Ascend-style vector units.

## Built-in Modules

| Registry key | Kind | Source family | Notes |
| --- | --- | --- | --- |
| `qwen3_attention` | mixer | Qwen3 / Qwen3.5 | Causal MHA + GQA, optional output gate |
| `qwen3_attention_sliding` | mixer | Qwen3 / Qwen3.5 | Sliding-window variant |
| `qwen3_5_gdn` | mixer | Qwen3.5 / Qwen3-Next style | Gated DeltaNet with chunk and recurrent dispatch surfaces |
| `dsv4_sliding_attention` | mixer | DeepSeek-V4 | Sliding-window shared-KV attention with sinks |
| `dsv4_hca_attention` | mixer | DeepSeek-V4 | Heavily compressed attention |
| `dsv4_csa_attention` | mixer | DeepSeek-V4 | Compressed sparse attention with Lightning Indexer |
| `qwen3_mlp` | FFN | Qwen3 / Qwen3.5 | SwiGLU MLP |
| `qwen3_5_moe` | FFN | Qwen3.5 | TopK router plus shared expert |
| `dsv4_moe` | FFN | DeepSeek-V4 | TopK router plus always-on shared expert |
| `dsv4_hash_moe` | FFN | DeepSeek-V4 | Hash routing through `tid2eid` lookup |

Shared primitives cover RMSNorm variants, rotary embeddings, eager attention,
and family-specific normalization / mRoPE differences while keeping checkpoint
keys aligned with upstream references.

## Training

For MindSpeed-MM FSDP2 training, start with
[examples/train/README_mindspeed_mm.md](examples/train/README_mindspeed_mm.md).
It contains YAML templates, data preparation notes, launch commands, and backend
switching for `torch`, `triton`, and `flash` paths.

The model class remains a standard `PreTrainedModel`, so ordinary HF Trainer and
Accelerate flows can also construct it when the surrounding training stack does
not require NPU-specific patches.

## Project Layout

```text
alloy/
├── configuration_alloy.py            # AlloyConfig and HF layer-type translation
├── modeling_alloy.py                 # AlloyModel / AlloyForCausalLM
├── loading.py                        # skeleton construction and state-dict streaming
├── modules/
│   ├── registry.py                   # mixer, FFN, and implementation registries
│   ├── attention/                    # Qwen3, Qwen3.5 GDN, DSV4 attention
│   ├── ffn/                          # dense MLP and MoE blocks
│   └── shared/                       # norm, rotary, attention helpers
├── roofline/                         # analytical FLOPs / bytes / roofline reports
├── integrations/                     # hf-npu-binder and MindSpeed-MM bridges
├── examples/                         # configs, build scripts, training templates
├── tools/                            # export and conversion helpers
└── tests/                            # construction, equivalence, binder, roofline suites
```

## Testing

Hardware-agnostic checks:

```bash
python -m alloy.tests.infra.test_construct
python -m alloy.tests.infra.test_impl_registry
python -m alloy.roofline.tests.test_smoke
python -m alloy.roofline.tests.test_modes
python -m alloy.roofline.tests.test_attention_specs
python -m alloy.roofline.tests.test_qwen3_attn_specs
python -m alloy.roofline.tests.test_gdn_specs
python -m alloy.roofline.tests.test_ffn_specs
```

Pretrained comparisons are heavier and load real checkpoints sequentially so a
single accelerator holds at most one model copy at a time:

```bash
python -m alloy.tests.models.qwen3.compare_alloy_eq_hf_pretrained_gpu \
    --pretrained /path/to/Qwen3-4B --dtype bf16

python -m alloy.tests.models.qwen3_5_moe.compare_alloy_eq_hf_pretrained_npu \
    --pretrained /path/to/Qwen3.5-35B-A3B --dtype bf16 --num-layers 4
```

Generated token ids must match exactly; fp32 eager equivalence targets
`max_abs = 0.0` where the upstream HF reference is deterministic.

## Known Limitations

- **Linear-attention incremental decode.** `Qwen35GatedDeltaNet` needs the right
  hybrid recurrent cache. Some generation paths still fall back to full
  re-forward per new token rather than using a fully fused incremental path.
- **Backend defaults are evidence-based.** Importing the binder registers fast
  paths, but `auto` may still choose `torch` for a module when measured Triton /
  AscendC behavior is not yet better than the eager NPU path.
- **Roofline is an upper-bound model.** Specs count mathematical work and ideal
  HBM traffic under fusion assumptions. They intentionally do not model launch
  overhead, kernel occupancy, communication, or framework scheduling gaps.
- **License pending.** Treat the repository as source-available until a formal
  license is chosen.

## Acknowledgements

Qwen3, Qwen3.5-MoE, and DeepSeek-V4 modules are ports of Hugging Face
`transformers` reference implementations, with math and parameter names preserved
so upstream checkpoints load without state-dict rewrites. The registry-oriented
hybrid-model presentation is inspired by the clarity of
[`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention).

## License

License is not yet chosen. Treat the code as source-available pending a formal
decision.
