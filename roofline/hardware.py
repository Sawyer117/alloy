"""Hardware peak performance presets for roofline analysis.

A :class:`Hardware` instance carries the numbers that define the roofline for
a given accelerator: peak FLOPS per dtype, optional separate vector-unit FLOPS
(Ascend-style hardware where the vector unit is physically separate from the
tensor-core / cube unit), and HBM bandwidth.

WARNING: Numbers are *peak* (theoretical) figures. Real workloads typically
achieve 50-80% of peak FLOPS (compute-bound) or 70-90% of peak HBM bandwidth
(memory-bound). The roofline is an upper bound on what's physically possible
on the hardware; the gap to measured performance = optimization opportunity.

Sources:
  * NVIDIA A100 SXM datasheet (Ampere whitepaper) — 80GB HBM2e variant
  * NVIDIA H100 SXM5 datasheet (Hopper whitepaper) — HBM3 variant
  * Huawei Ascend 910B1 / 910C product specs

For SKUs with multiple variants (A100 PCIe vs SXM, 910B1 vs 910B3) we use the
higher-bandwidth training variant (SXM, 910B1). Tensor-core / Cube-AI numbers
are dense (no 2:4 sparsity). Pure FP32 is CUDA-core / vector throughput on
NVIDIA; TF32 (which matches BF16 throughput) is reported separately by the
vendor and not included here.

Cube vs Vector throughput on Ascend: the 910 family has physically separate
tensor (Cube) and elementwise (Vector) compute units. Cube unit handles
matmul / conv at high TFLOPS; Vector unit handles elementwise / norm /
softmax at much lower TFLOPS (16x ratio at FP16 on 910B1). The roofline
analyzer doesn't currently split flops by unit, but the data is exposed via
:attr:`peak_vector_flops` for future analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# Unit constants — keep peak_flops in FLOP/s and hbm_bandwidth in B/s so the
# arithmetic in analyze.py is dimensionally clean (no implicit scale juggling).
T = 1e12  # tera
G = 1e9   # giga


@dataclass(frozen=True)
class Hardware:
    """Peak FLOPS-per-dtype + HBM bandwidth for one accelerator.

    Roofline math::

        compute_time = total_flops / peak_flops[dtype]
        memory_time  = total_bytes / hbm_bandwidth
        bottleneck   = max(compute_time, memory_time)

    ``peak_flops`` covers tensor-core / cube-unit compute (matmul-friendly).
    ``peak_vector_flops`` is optional and carries the Ascend-style separate
    vector-unit throughput for elementwise ops (None on hardware where the
    vector unit isn't distinct, e.g. NVIDIA SMs).
    """

    name: str
    peak_flops: dict[torch.dtype, float]                          # tensor-core / cube unit FLOP/s
    hbm_bandwidth: float                                           # bytes/s
    peak_vector_flops: Optional[dict[torch.dtype, float]] = None  # vector unit FLOP/s (Ascend-style)

    def get_peak_flops(self, dtype: torch.dtype) -> float:
        """Tensor-core / cube-unit peak for the given dtype."""
        if dtype not in self.peak_flops:
            raise KeyError(
                f"{self.name} has no peak-FLOPs entry for {dtype}; "
                f"available: {list(self.peak_flops)}."
            )
        return self.peak_flops[dtype]

    def get_peak_vector_flops(self, dtype: torch.dtype) -> Optional[float]:
        """Vector-unit peak for the given dtype, or None if not separately
        modelled (e.g. NVIDIA hardware where everything goes through the SMs).
        """
        if self.peak_vector_flops is None:
            return None
        return self.peak_vector_flops.get(dtype)


# --------------------------------------------------------------------------- #
# NVIDIA presets
# --------------------------------------------------------------------------- #


# A100 80GB SXM (HBM2e). Tensor-core TFLOPS in dense mode (no 2:4 sparsity).
# FP32 here is pure CUDA-core 19.5 TFLOPS, NOT TF32 (which is 156 TFLOPS and
# matches BF16 throughput). Choose explicitly when constructing analysis.
A100 = Hardware(
    name="A100-80GB-SXM",
    peak_flops={
        torch.bfloat16: 312 * T,
        torch.float16: 312 * T,
        torch.float32: 19.5 * T,
    },
    hbm_bandwidth=2039 * G,  # 2.039 TB/s
)


# H100 SXM5 (HBM3). Tensor-core TFLOPS in dense mode (no 2:4 sparsity).
# FP8 not included here (separate dtype family); add in v2 if needed.
H100 = Hardware(
    name="H100-SXM5",
    peak_flops={
        torch.bfloat16: 989 * T,
        torch.float16: 989 * T,
        torch.float32: 67 * T,
    },
    hbm_bandwidth=3350 * G,  # 3.35 TB/s
)


# --------------------------------------------------------------------------- #
# Huawei Ascend presets
# --------------------------------------------------------------------------- #


# Ascend 910B1 (HBM2e, training variant). Cube + Vector unit FLOPS.
# Source: vendor product spec. Note the ~16x cube/vector throughput gap at
# FP16 — vector ops on this hardware can become a bottleneck when their
# share of total FLOPs grows (e.g. heavy norm / softmax / RoPE workloads).
ASCEND_910B1 = Hardware(
    name="Ascend-910B1",
    peak_flops={
        torch.int8:     758 * T,    # cube INT8 TOPS
        torch.bfloat16: 379 * T,    # cube BF16
        torch.float16:  379 * T,    # cube FP16
        torch.float32:  95 * T,     # cube FP32
    },
    peak_vector_flops={
        torch.float16:  24 * T,     # vector FP16
        torch.float32:  11.84 * T,  # vector FP32
    },
    hbm_bandwidth=1800 * G,  # 1.8 TB/s
)


# Ascend 910C — next gen, all compute throughput doubled relative to 910B1
# and HBM bandwidth bumped to 3.2 TB/s. Same cube/vector split structure.
ASCEND_910C = Hardware(
    name="Ascend-910C",
    peak_flops={
        torch.int8:     1516 * T,
        torch.bfloat16: 758 * T,
        torch.float16:  758 * T,
        torch.float32:  190 * T,
    },
    peak_vector_flops={
        torch.float16:  48 * T,
        torch.float32:  23.68 * T,
    },
    hbm_bandwidth=3200 * G,  # 3.2 TB/s
)


# Backward-compat alias — earlier code referenced ASCEND_910B; keep the
# symbol pointing at the 910B1 Hardware instance so existing imports and
# preset names ("Ascend910B") continue to resolve.
ASCEND_910B = ASCEND_910B1


PRESETS: dict[str, Hardware] = {
    "A100": A100,
    "H100": H100,
    "Ascend910B": ASCEND_910B1,    # backward-compat alias
    "Ascend910B1": ASCEND_910B1,
    "Ascend910C": ASCEND_910C,
}


def get_hardware(name_or_obj) -> Hardware:
    """Resolve a preset name or pass-through a :class:`Hardware` instance."""
    if isinstance(name_or_obj, Hardware):
        return name_or_obj
    if name_or_obj not in PRESETS:
        raise KeyError(
            f"Unknown hardware preset '{name_or_obj}'. "
            f"Available: {sorted(PRESETS)}. Or pass a Hardware instance directly."
        )
    return PRESETS[name_or_obj]


__all__ = [
    "Hardware",
    "A100",
    "H100",
    "ASCEND_910B",
    "ASCEND_910B1",
    "ASCEND_910C",
    "PRESETS",
    "get_hardware",
]
