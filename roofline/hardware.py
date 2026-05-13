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


# Ascend 950PR — published / leaked specs for the next-gen Ascend with
# native FP8 / FP4 cube support. BF16 throughput is not directly published
# at the time of this preset; we infer it from the FP8:BF16 = 2:1 industry
# pattern (H100, B100, MI300X all keep BF16 at ~half of FP8). The vector
# unit numbers from 910C scale proportionally — kept conservative since
# the actual 950 vector spec hasn't been independently verified.
# FP4 cube throughput is 2 PFLOPS (twice FP8); not stored in peak_flops
# because alloy roofline runs in bf16 by default and torch lacks a
# float4 dtype key.
ASCEND_950PR = Hardware(
    name="Ascend-950PR",
    peak_flops={
        torch.bfloat16: 500 * T,    # estimated, FP8 / 2 per the industry convention
        torch.float16:  500 * T,    # same as BF16
        torch.float32:  125 * T,    # estimated, FP16 / 4
        torch.float8_e4m3fn: 1000 * T,  # 1 PFLOPS FP8 (published)
        # FP4: 2 PFLOPS (published) — not stored, no torch dtype for it.
    },
    peak_vector_flops={
        torch.float16:  60 * T,    # scaled from 910C
        torch.float32:  30 * T,
    },
    hbm_bandwidth=1600 * G,  # 1.6 TB/s
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
    "Ascend950PR": ASCEND_950PR,
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


# --------------------------------------------------------------------------- #
# Custom hardware factory — kwarg-style construction
# --------------------------------------------------------------------------- #


def CustomHardware(
    name: str = "custom",
    *,
    hbm_bandwidth: float,
    int8: Optional[float] = None,
    fp8: Optional[float] = None,
    fp16: Optional[float] = None,
    bf16: Optional[float] = None,
    fp32: Optional[float] = None,
    fp64: Optional[float] = None,
    vector_fp16: Optional[float] = None,
    vector_bf16: Optional[float] = None,
    vector_fp32: Optional[float] = None,
) -> Hardware:
    """Build a :class:`Hardware` from named-dtype keyword args.

    Each compute kwarg is the peak FLOP/s (or TOPS for int8) on the cube /
    tensor-core unit; ``vector_*`` go to the separate vector unit on
    Ascend-style hardware (leave None on NVIDIA where the distinction
    doesn't apply). ``hbm_bandwidth`` is bytes/s.

    Pass numbers in absolute units — ``989e12`` for 989 TFLOPS, ``3.35e12``
    for 3.35 TB/s — or use the module-level ``T`` / ``G`` constants
    (``989 * T``, ``3.35 * T``).

    Unspecified dtypes are absent from the resulting ``peak_flops`` dict;
    calling ``roofline(..., dtype=that_missing_dtype, ...)`` will raise a
    helpful KeyError.

    Example::

        from alloy.roofline import CustomHardware, roofline_prefill

        my_chip = CustomHardware(
            name="my-chip-v1",
            hbm_bandwidth=4e12,     # 4 TB/s
            bf16=1500e12,           # 1500 TFLOPS BF16
            fp32=200e12,
            int8=3000e12,
        )
        report = roofline_prefill(config, batch=1, seq_len=4096, hardware=my_chip)
    """
    peak: dict[torch.dtype, float] = {}
    if int8 is not None:
        peak[torch.int8] = int8
    if fp16 is not None:
        peak[torch.float16] = fp16
    if bf16 is not None:
        peak[torch.bfloat16] = bf16
    if fp32 is not None:
        peak[torch.float32] = fp32
    if fp64 is not None:
        peak[torch.float64] = fp64
    if fp8 is not None:
        if hasattr(torch, "float8_e4m3fn"):
            peak[torch.float8_e4m3fn] = fp8
        else:
            raise ValueError(
                "fp8 specified but this torch version lacks torch.float8_e4m3fn. "
                "Upgrade to torch>=2.1, or construct Hardware(...) directly with "
                "a custom dtype key if you need a non-standard quantization format."
            )

    vec: Optional[dict[torch.dtype, float]] = None
    if any(v is not None for v in (vector_fp16, vector_bf16, vector_fp32)):
        vec = {}
        if vector_fp16 is not None:
            vec[torch.float16] = vector_fp16
        if vector_bf16 is not None:
            vec[torch.bfloat16] = vector_bf16
        if vector_fp32 is not None:
            vec[torch.float32] = vector_fp32

    return Hardware(
        name=name,
        peak_flops=peak,
        hbm_bandwidth=hbm_bandwidth,
        peak_vector_flops=vec,
    )


__all__ = [
    "Hardware",
    "A100",
    "H100",
    "ASCEND_910B",
    "ASCEND_910B1",
    "ASCEND_910C",
    "ASCEND_950PR",
    "PRESETS",
    "get_hardware",
    "CustomHardware",
    "T",
    "G",
]
