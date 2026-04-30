import os as _os

from .configuration_alloy import AlloyConfig
from .loading import (
    build_on_device,
    build_skeleton,
    empty_cache,
    load_state_dict_from_disk,
    strip_language_model_prefix,
)
from .modeling_alloy import AlloyForCausalLM, AlloyModel, AlloyPreTrainedModel

from . import modules  # triggers all @register_mixer / @register_ffn side effects


# Auto-load the hf-npu-binder bridge if the binder package is importable.
# Rationale: third-party loaders (vLLM, TGI, SGLang, HF Inference Endpoints)
# call AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True) and
# go through the modeling_alloy shim, which does ``from alloy import
# AlloyForCausalLM`` — that triggers this file. Without the auto-load they
# never see binder kernels even when hf_npu_binder is installed and the
# config carries _qwen3_5_gdn_implementation: "flash". With the auto-load,
# install + config field is sufficient and consumers don't need to inject
# ``import alloy.integrations.hf_npu_binder`` into someone else's framework.
#
# Opt out by setting ``ALLOY_DISABLE_AUTO_BRIDGE=1`` — useful for testing
# the torch reference path without uninstalling binder, or for diagnosing
# a flaky binder install.
#
# Only ImportError is swallowed. If binder is installed but loads with a
# different exception (missing .so, version mismatch, ...) we let it
# propagate — silent failure here is a debugging nightmare and the user
# wants to know binder is broken.
if _os.environ.get("ALLOY_DISABLE_AUTO_BRIDGE") != "1":
    try:
        from .integrations import hf_npu_binder as _hf_npu_binder  # noqa: F401
    except ImportError:
        pass

__all__ = [
    "AlloyConfig",
    "AlloyForCausalLM",
    "AlloyModel",
    "AlloyPreTrainedModel",
    "build_on_device",
    "build_skeleton",
    "empty_cache",
    "load_state_dict_from_disk",
    "strip_language_model_prefix",
]
