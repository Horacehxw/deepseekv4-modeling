"""OpProfile dataclass, roofline engine, and communication helpers."""

from dataclasses import dataclass
from typing import List

from .config import Config


@dataclass
class OpProfile:
    name: str
    flops: float = 0.0        # Cube/matmul FLOPs
    vec_ops: float = 0.0      # Vector FLOPs
    mem_bytes: float = 0.0    # Physical HBM bytes after quant policy
    comm_bytes: float = 0.0   # Communication bytes
    cube_time_s: float = 0.0
    vec_time_s: float = 0.0
    mem_time_s: float = 0.0
    comm_time_s: float = 0.0
    time_s: float = 0.0       # max(cube + vec, mem) + comm
    bottleneck: str = ""       # "CUBE" / "VEC" / "MEM" / "COMM"


# Valid op_kind values accepted by roofline_time().
OP_KINDS = frozenset({"gemm", "attention", "vector", "other", "comm"})

# Quantization byte ratios for GEMM weights and KV cache.
WEIGHT_BYTE_RATIOS = {"bf16": 1.0, "w8a8": 0.5}
KV_BYTE_RATIOS     = {"bf16": 1.0, "kv8": 0.5, "kv4": 0.25}


def roofline_time(name: str, flops: float, vec_ops: float, mem_bytes: float,
                  cfg: Config, op_kind: str,
                  comm_time_s: float = 0.0,
                  comm_bytes: float = 0.0) -> OpProfile:
    """Compute roofline time with inline quantization policy.

    op_kind controls which quantization scaling applies:
      - "gemm":      W8A8 uses effective_w8a8_tflops; mem_bytes scaled by WEIGHT_BYTE_RATIOS.
      - "attention": mem_bytes scaled by KV_BYTE_RATIOS; compute throughput stays BF16.
      - "vector", "other", "comm": no quantization scaling.

    Cube and vector execution are serial on the compute datapath:
      compute_time = max(cube_time + vec_time, mem_time)
      total_time   = compute_time + comm_time

    Raises ValueError for unknown op_kind values.
    """
    if op_kind not in OP_KINDS:
        raise ValueError(
            f"Unknown op_kind {op_kind!r}. Must be one of: {sorted(OP_KINDS)}"
        )

    hw = cfg.hw
    rt = cfg.rt

    if rt.quant_mode not in WEIGHT_BYTE_RATIOS:
        raise ValueError(
            f"Unknown quant_mode {rt.quant_mode!r}. Valid values: {sorted(WEIGHT_BYTE_RATIOS)}"
        )
    if rt.kv_cache_quant_mode not in KV_BYTE_RATIOS:
        raise ValueError(
            f"Unknown kv_cache_quant_mode {rt.kv_cache_quant_mode!r}. Valid values: {sorted(KV_BYTE_RATIOS)}"
        )

    # Apply quantization policy to cube throughput and mem_bytes.
    cube_tflops = hw.cube_tflops
    effective_mem = mem_bytes

    if op_kind == "gemm" and rt.quant_mode == "w8a8":
        cube_tflops   = hw.effective_w8a8_tflops
        effective_mem = mem_bytes * WEIGHT_BYTE_RATIOS[rt.quant_mode]
    elif op_kind == "attention":
        effective_mem = mem_bytes * KV_BYTE_RATIOS[rt.kv_cache_quant_mode]

    cube_time = flops / (cube_tflops * 1e12 * hw.effective_cube_utilization) if flops > 0 else 0.0
    if vec_ops > 0:
        vec_time = (vec_ops / (hw.vec_tflops * 1e12 * hw.effective_vec_utilization)
                    + hw.vec_static_latency_us * 1e-6)
    else:
        vec_time = 0.0
    mem_time  = effective_mem / (hw.hbm_bandwidth_gbps * 1e9 * hw.hbm_bw_utilization) if effective_mem > 0 else 0.0

    compute_side = cube_time + vec_time
    compute_time = max(compute_side, mem_time)
    total_time = compute_time + comm_time_s

    if total_time == 0.0:
        bottleneck = ""
    elif comm_time_s > compute_time:
        bottleneck = "COMM"
    elif mem_time > compute_side:
        bottleneck = "MEM"
    elif cube_time >= vec_time:
        bottleneck = "CUBE"
    else:
        bottleneck = "VEC"

    return OpProfile(
        name=name,
        flops=flops,
        vec_ops=vec_ops,
        mem_bytes=effective_mem,
        comm_bytes=comm_bytes,
        cube_time_s=cube_time,
        vec_time_s=vec_time,
        mem_time_s=mem_time,
        comm_time_s=comm_time_s,
        time_s=total_time,
        bottleneck=bottleneck,
    )


def allreduce_time(vol_bytes: float, n: int, bw_gbps: float,
                   latency_us: float, bw_util: float) -> float:
    """AllReduce: 2*(n-1)/n * vol / effective_bw + latency."""
    if n <= 1:
        return 0.0
    factor = 2.0 * (n - 1) / n
    steps = 2 * (n - 1)
    return factor * vol_bytes / (bw_gbps * 1e9 * bw_util) + steps * latency_us * 1e-6


def alltoall_time(vol_bytes: float, n: int, bw_gbps: float,
                  latency_us: float, bw_util: float) -> float:
    """AllToAll: (n-1)/n * vol / effective_bw + latency."""
    if n <= 1:
        return 0.0
    factor = (n - 1) / n
    return factor * vol_bytes / (bw_gbps * 1e9 * bw_util) + latency_us * 1e-6


def allgather_time(vol_bytes: float, n: int, bw_gbps: float,
                   latency_us: float, bw_util: float) -> float:
    """AllGather: (n-1)/n * vol / effective_bw + latency."""
    if n <= 1:
        return 0.0
    factor = (n - 1) / n
    steps = n - 1
    return factor * vol_bytes / (bw_gbps * 1e9 * bw_util) + steps * latency_us * 1e-6


def sum_ops(ops: List[OpProfile], name: str) -> OpProfile:
    """Sum a list of OpProfiles into a single aggregate."""
    total = OpProfile(name=name)
    for op in ops:
        total.flops += op.flops
        total.vec_ops += op.vec_ops
        total.mem_bytes += op.mem_bytes
        total.comm_bytes += op.comm_bytes
        total.cube_time_s += op.cube_time_s
        total.vec_time_s += op.vec_time_s
        total.mem_time_s += op.mem_time_s
        total.comm_time_s += op.comm_time_s
        total.time_s += op.time_s
    compute_side = total.cube_time_s + total.vec_time_s
    if total.time_s == 0:
        total.bottleneck = ""
    elif total.comm_time_s > max(compute_side, total.mem_time_s):
        total.bottleneck = "COMM"
    elif total.mem_time_s > compute_side:
        total.bottleneck = "MEM"
    elif total.cube_time_s >= total.vec_time_s:
        total.bottleneck = "CUBE"
    else:
        total.bottleneck = "VEC"
    return total


def bytes2(count: int) -> float:
    """BF16 bytes for `count` elements."""
    return count * 2.0


def bytes4(count: int) -> float:
    """FP32 bytes for `count` elements."""
    return count * 4.0
