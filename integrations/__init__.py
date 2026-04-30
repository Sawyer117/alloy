"""Optional integrations with external packages.

Modules in this subpackage are **opt-in**: alloy core (``alloy.modules.*``,
``alloy.modeling_alloy``, ``alloy.loading``) never imports anything from
here. A consumer who wants a particular integration imports it explicitly,
e.g.

    import alloy.integrations.mindspeed_mm           # plugin shim for trainer
    import alloy.integrations.hf_npu_binder          # wires NPU/triton fast paths

This keeps alloy's model code free of optional / hardware-specific deps
(architecture decision: model code imports zero hardware-specific symbols).

Auto-load exception — hf_npu_binder
-----------------------------------
``alloy/__init__.py`` does an opportunistic ``try: from alloy.integrations
import hf_npu_binder``. If ``hf_npu_binder`` is installed, the bridge
auto-registers and consumers (vLLM / TGI / SGLang / HF Inference / your
own Python) get binder kernels just by setting
``config._qwen3_5_gdn_implementation: "flash"`` (or similar). If
``hf_npu_binder`` isn't installed, the ImportError is swallowed and
alloy falls back to the torch reference path.

Set ``ALLOY_DISABLE_AUTO_BRIDGE=1`` to opt out — useful for testing the
torch reference on a machine that happens to have binder installed, or
for diagnosing a flaky binder install. The explicit
``import alloy.integrations.hf_npu_binder`` form still works either way.
"""
