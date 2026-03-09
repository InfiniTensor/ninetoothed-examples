"""
TileLang 操作接口包
提供与 ninetoothed、triton 一致的算子接口
支持自动调优功能
"""

from .torch import (
    mm,
    rms_norm,
    softmax,
    benchmark_tilelang_op,
    get_kernel_source,
)

__all__ = [
    "mm",
    "rms_norm",
    "softmax",
    "benchmark_tilelang_op",
    "get_kernel_source",
]
