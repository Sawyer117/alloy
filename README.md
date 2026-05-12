<div align="center">

# 🧪 Alloy

**HuggingFace-native hybrid transformer composition with built-in roofline analysis.**

Define hybrid architectures by config. Load real HF checkpoints. Analyze theoretical
performance. One library, three pillars.

[Quick start](#-quick-start) • [Three pillars](#-three-pillars) • [Built-in modules](#-built-in-modules) • [Project layout](#-project-layout) • [Testing](#-testing)

</div>

---

## ✨ Three pillars

### 1. 🧩 Composable by config, not code

A model's architecture is fully described by two ordered lists: `layer_types` (token
mixer per depth) and `ffn_types` (feed-forward per depth). Mix softmax attention
with linear attention, dense with MoE, sliding with full attention, in any pattern.
**Switching architectures is a JSON edit** — no forking, no per-architecture
`modeling_*.py` rewrites.

```python
from alloy import AlloyConfig, AlloyForCausalLM

config = AlloyConfig(
    vocab_size=32000, hidden_size=2048, num_hidden_layers=16,
    num_attention_heads=16, num_key_value_heads=2, head_dim=128, intermediate_size=8192,
    layer_types=["qwen3_5_gdn", "qwen3_5_gdn", "qwen3_5_gdn", "qwen3_attention"] * 4,
    ffn_types=["qwen3_mlp"] * 16,
)
model = AlloyForCausalLM(config)
```

Adding a new mixer is a decorator and one line in your config:

```python
from alloy.modules.registry import register_mixer

@register_mixer("my_rwkv", attr_name="linear_attn", mask_kind="linear")
class MyRWKV(nn.Module):
    ...

config = AlloyConfig(..., layer_types=["my_rwkv"] * 8 + ["qwen3_attention"] * 8, ...)
```

The core decoder layer never knew this mixer existed; no PR to alloy required.
Mask routing follows the declared `mask_kind` (`"causal"` / `"sliding"` / `"linear"`),
so model-level mask precompute stays generic.

### 2. 🤗 HuggingFace-native end-to-end

Models are plain `PreTrainedModel` subclasses backed by `PretrainedConfig`. They
integrate with `from_pretrained`, `save_pretrained`, `generate`, Trainer,
Accelerate, PEFT, and any FSDP2-based backend out of the box. **Parameter names
match HF qwen3 / qwen3.5 checkpoints exactly** — Hub weights load with
`load_state_dict(strict=False)` and run bit-exact under fp32 eager
(`max_abs = 0.0` verified end-to-end).

```python
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

alloy = AlloyForCausalLM(alloy_config_from_qwen3(hf.config))
alloy.load_state_dict(hf.state_dict(), strict=False)
# generate(), save_pretrained(), Trainer, etc. all work as expected
```

Hardware portability is part of the same story: `modeling_alloy.py` and
`modules/**` carry zero `torch_npu` / CUDA-specific imports. NPU fast-path
kernels (triton / flash) layer on as opt-in runtime patches via the
`hf-npu-binder` bridge — same model definition, same checkpoint, runs on
both GPU and Ascend NPU.

### 3. 📐 Roofline modeling per architecture

`alloy.roofline` computes theoretical FLOPs and HBM bytes for one forward pass —
**purely analytical, config-driven, no model construction**. Handles
10T-parameter configs in O(num_layers) time. Three modes
(prefill / mini-prefill / decode), four hardware presets
(A100 / H100 / Ascend910B1 / Ascend910C) plus `CustomHardware(...)` for
chips not in the preset list, auto-scaled report.

```python
from alloy.roofline import (
    CustomHardware, roofline_prefill, roofline_mini_prefill, roofline_decode,
)

# Custom chip not in the preset list — build from kwargs (numbers are illustrative)
my_device = CustomHardware(
    name="my-device", hbm_bandwidth=8e12,
    bf16=2250e12, fp32=80e12, fp8=4500e12,
)

# Print one full report to see the format
print(roofline_prefill(config, batch=1, seq_len=4096, hardware="H100"))

# Three modes (cold prefill / mini-prefill / decode) cover the serving pipeline.
# Mix preset strings and CustomHardware instances freely.
# FLOPs and bytes are config-determined; only timing changes per device.
for hw in ["H100", "Ascend910C", my_device]:
    p = roofline_prefill     (config, batch=1, seq_len=4096, hardware=hw)
    m = roofline_mini_prefill(config, batch=1, chunk_len=512, kv_cache_len=2048, hardware=hw)
    d = roofline_decode      (config, batch=1, kv_cache_len=4096, hardware=hw)
    name = hw if isinstance(hw, str) else hw.name
    print(f"{name:11}  prefill: {p.roofline_time_s*1e3:>5.2f} ms ({p.bottleneck:7})"
          f"  mini: {m.roofline_time_s*1e3:>5.2f} ms ({m.bottleneck:7})"
          f"  decode: {d.roofline_time_s*1e6:>4.0f} us ({d.bottleneck})")
```

```
Roofline | H100-SXM5 | bf16 | prefill (B=1, Q=4096)
------------------------------------------------------------------------------
TOTAL: 10.18 TFLOPs / 3.77 GB / AI = 2703.7
H100-SXM5: compute=10.30 ms, memory=1.12 ms -> bottleneck=compute (10.30 ms / forward)

H100         prefill: 10.30 ms (compute)  mini:  1.26 ms (compute)  decode:  702 us (memory)
Ascend910C   prefill: 13.43 ms (compute)  mini:  1.65 ms (compute)  decode:  735 us (memory)
my-device    prefill:  4.53 ms (compute)  mini:  0.55 ms (compute)  decode:  294 us (memory)
```

`CustomHardware(...)` accepts named-dtype kwargs (`int8` / `fp8` / `fp16` /
`bf16` / `fp32` / `fp64` for the cube/tensor unit, `vector_*` for Ascend-style
separate vector throughput), all in absolute FLOP/s. It returns a regular
`Hardware` instance — pass it to `hardware=` anywhere a preset string works.

---

## ⚡ Quick start

```bash
pip install torch transformers safetensors

git clone https://github.com/Sawyer117/alloy
# Put the parent directory on PYTHONPATH so `import alloy` works.
```

Build, load, analyze:

```python
from alloy import AlloyConfig, AlloyForCausalLM
from alloy.roofline import roofline_prefill

config = AlloyConfig(
    vocab_size=32000, hidden_size=2048, num_hidden_layers=16,
    num_attention_heads=16, num_key_value_heads=2, head_dim=128, intermediate_size=8192,
    layer_types=["qwen3_5_gdn", "qwen3_5_gdn", "qwen3_5_gdn", "qwen3_attention"] * 4,
    ffn_types=["qwen3_mlp"] * 16,
)
model = AlloyForCausalLM(config)

print(roofline_prefill(config, batch=1, seq_len=2048, hardware="A100"))
```

Two end-to-end demos in [`alloy/examples/`](alloy/examples/) load real
Qwen3 / Qwen3.5-MoE weights and run greedy generation. For training, see
**[`examples/train/README_mindspeed_mm.md`](examples/train/README_mindspeed_mm.md)** —
yaml templates, data prep, launchers, and backend switching
(`torch` / `triton` / `flash`) for the MindSpeed-MM FSDP2 workflow.

---

## 🔌 Built-in modules

| Registry key                | Family            | Source            | Notes                                              |
|-----------------------------|-------------------|-------------------|----------------------------------------------------|
| `qwen3_attention`           | mixer             | qwen3 / qwen3.5   | Causal MHA + GQA, optional `attn_output_gate`      |
| `qwen3_attention_sliding`   | mixer             | qwen3 / qwen3.5   | Sliding-window variant of the above                |
| `qwen3_5_gdn`               | mixer (linear)    | qwen3.5           | Gated DeltaNet, chunk + fused recurrent kernels    |
| `dsv4_sliding_attention`    | mixer             | DeepSeek V4       | Sliding window, MQA, per-head sinks                |
| `dsv4_hca_attention`        | mixer             | DeepSeek V4       | Heavy compressed attention (128:1)                 |
| `dsv4_csa_attention`        | mixer             | DeepSeek V4       | Compressed sparse attention + Lightning Indexer    |
| `qwen3_mlp`                 | ffn               | qwen3 / qwen3.5   | SwiGLU MLP                                         |
| `qwen3_5_moe`               | ffn (sparse)      | qwen3.5           | TopK router + gated shared expert                  |
| `dsv4_moe`                  | ffn (sparse)      | DeepSeek V4       | TopK router + always-on shared expert              |
| `dsv4_hash_moe`             | ffn (sparse)      | DeepSeek V4       | Hash routing via `tid2eid` lookup                  |

All entries have paired `RooflineSpec`s registered. Shared primitives
(`RMSNorm`, `RotaryEmbedding`, `eager_attention_forward`) cover both qwen3
(`w * x`, full rotary) and qwen3.5 (`(1 + w) * x`, partial + interleaved mRoPE)
conventions from a single class via flag-driven dispatch.

---

## 📁 Project layout

```
alloy/
├── configuration_alloy.py            # AlloyConfig(PretrainedConfig)
├── modeling_alloy.py                 # AlloyDecoderLayer, AlloyModel, AlloyForCausalLM
├── loading.py                        # build_skeleton / build_on_device / state_dict streaming
├── modules/
│   ├── registry.py                   # MIXER_REGISTRY, FFN_REGISTRY (mask_kind, decorators)
│   ├── attention/                    # qwen3_attention, qwen3_5_gdn, dsv4_attention
│   ├── ffn/                          # qwen3_mlp, qwen3_5_moe, dsv4_moe
│   └── shared/                       # norm, rotary, attention_kernels
├── roofline/                         # ★ analytical FLOPs / bytes / arithmetic intensity
│   ├── analyze.py                    # roofline / roofline_prefill / _mini_prefill / _decode
│   ├── specs.py                      # RooflineSpec ABC + LinearSpec + RMSNormSpec
│   ├── specs_attention.py            # Qwen3AttentionSpec, DSV4AttentionSpec
│   ├── specs_ffn.py                  # SwiGLUMLPSpec, DSV4MoESpec, Qwen35MoESpec
│   ├── specs_gdn.py                  # Qwen35GDNSpec (chunk + fused_recurrent dispatch)
│   └── hardware.py                   # A100 / H100 / Ascend910B1 / Ascend910C presets
├── integrations/                     # hf_npu_binder, mindspeed_mm bridges
├── tools/                            # export_for_hub
├── examples/                         # configs/, train/, build_*.py demos
├── scripts/                          # compare_*.py equivalence demos
└── tests/                            # construct + gpu/ + npu/ + 6 roofline suites
```

---

## 🧪 Testing

```bash
# Roofline analytics — 63 hand-computed tests, 6 suites
python -m alloy.tests.test_roofline_smoke
python -m alloy.tests.test_roofline_modes
python -m alloy.tests.test_roofline_attention_specs
python -m alloy.tests.test_roofline_qwen3_attn_specs
python -m alloy.tests.test_roofline_gdn_specs
python -m alloy.tests.test_roofline_ffn_specs

# Construction smoke (hardware-agnostic)
python -m alloy.tests.test_construct

# CUDA: real Qwen3 weights vs HF reference (max_abs, max relative, token-id equality)
python -m alloy.tests.gpu.compare_qwen3_pretrained --pretrained Qwen/Qwen3-4B --dtype bf16

# Ascend NPU: same protocol, torch_npu auto-redirected; --num-layers N for memory-bound cards
python -m alloy.tests.npu.compare_qwen3_5_pretrained \
    --pretrained /path/to/Qwen3.5-35B-A3B --dtype bf16 --num-layers 4
```

Pretrained-weight comparisons run sequentially (HF reference forward → save logits +
token ids → release HBM → build alloy → stream weights from on-disk safetensors →
compare) so a single 80 GB card holds at most one model copy. Generated token ids
must match exactly (`torch.equal`); fp32 eager achieves `max_abs = 0.0` end-to-end.

---

## 🚧 Known limitations

- **Incremental decoding for linear attention.** `Qwen35GatedDeltaNet` expects a
  `HybridCache`. Generation with `DynamicCache` falls back to full re-forward per
  new token (`use_cache=False`).
- **NPU fused-kernel patch** is provided via `hf-npu-binder`; the in-tree
  `alloy.npu_patch` placeholder is not yet implemented.
- **Roofline read/write split.** Bytes are aggregated into one HBM-traffic number
  (correct under the shared-bus model). Per-direction breakdown is on the v2
  list.

---

## 🤝 Acknowledgements

`Qwen3Attention`, `Qwen35GatedDeltaNet`, `Qwen35SparseMoE`, and the DSV4 attention
and MoE blocks are ports of HuggingFace `transformers` reference implementations,
preserving math and parameter names so upstream checkpoints load without
modification. The registry-based decoder-layer pattern follows HuggingFace's
canonical hybrid decoder layout. README structure follows
[`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention).

## 📄 License

License is not yet chosen. Treat the code as source-available pending a formal decision.
