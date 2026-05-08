"""Roofline analysis for alloy models.

Computes theoretical FLOPs, HBM bytes-movement, and arithmetic intensity for a
given config + input shape, under optimal-fusion assumptions. Pure analytical
(no model construction, no meta forward) — handles arbitrarily large configs
in O(num_layers) time and memory.

Public API::

    from alloy.roofline import roofline

    report = roofline(config, batch=1, seq_len=4096,
                      dtype=torch.bfloat16, hardware="A100")
    print(report)

Specs are registered against the same names used by ``register_mixer`` /
``register_ffn`` in ``alloy.modules.registry``. Forgetting to register a
spec is fail-open by default (``strict=False``): the analyzer warns, tags
the module ``"unknown"``, contributes zero, and continues. Use
``strict=True`` in CI gates to enforce coverage.
"""
from __future__ import annotations

from .analyze import (
    ModuleStat,
    RooflineReport,
    roofline,
    roofline_decode,
    roofline_mini_prefill,
    roofline_prefill,
)
from .hardware import (
    A100,
    ASCEND_910B,
    ASCEND_910B1,
    ASCEND_910C,
    H100,
    PRESETS,
    CustomHardware,
    Hardware,
    get_hardware,
)
from .specs import (
    SPEC_REGISTRY,
    LinearSpec,
    RMSNormSpec,
    RooflineSpec,
    dtype_size,
    get_spec,
    register_spec,
)

# Importing the family modules triggers their register_spec() side effects,
# binding specs to the same names used by register_ffn / register_mixer in
# alloy.modules. Mirrors how alloy/__init__.py imports alloy.modules to
# trigger mixer/ffn registration.
from . import specs_attention, specs_ffn, specs_gdn  # noqa: F401


__all__ = [
    # Entry point + report
    "roofline",
    "roofline_prefill",
    "roofline_decode",
    "roofline_mini_prefill",
    "RooflineReport",
    "ModuleStat",
    # Specs
    "RooflineSpec",
    "LinearSpec",
    "RMSNormSpec",
    "register_spec",
    "get_spec",
    "SPEC_REGISTRY",
    "dtype_size",
    # Hardware
    "Hardware",
    "CustomHardware",
    "A100",
    "H100",
    "ASCEND_910B",
    "ASCEND_910B1",
    "ASCEND_910C",
    "PRESETS",
    "get_hardware",
]
