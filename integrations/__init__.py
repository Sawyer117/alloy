"""Optional integrations with external packages.

Modules in this subpackage are **opt-in**: alloy core (``alloy.modules.*``,
``alloy.modeling_alloy``, ``alloy.loading``) never imports anything from
here. A consumer who wants a particular integration imports it explicitly,
e.g.

    import alloy.integrations.hf_npu_binder        # wires NPU/triton fast paths

This keeps alloy's model code free of optional / hardware-specific deps
(architecture decision: model code imports zero hardware-specific symbols).
"""
