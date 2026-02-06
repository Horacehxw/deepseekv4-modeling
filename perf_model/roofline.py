"""OpProfile dataclass, roofline engine, and communication helpers."""

from dataclasses import dataclass
from typing import List

from .config import HardwareConfig


@dataclass
class OpProfile:
    name: str
    flops: float = 0.0        # Cube/matmul FLOPs
    vec_ops: float = 0.0      # Vector FLOPs
    mem_bytes: float = 0.0    # HBM read + write bytes
    comm_bytes: float = 0.0   # Communication bytes
    cube_time_s: float = 0.0
    vec_time_s: float = 0.0
    mem_time_s: float = 0.0
    comm_time_s: float = 0.0
    time_s: float = 0.0       # max(cube, vec, mem) + comm
    bottleneck: str = ""       # "CUBE" / "VEC" / "MEM" / "COMM"


def roofline_time(name: str, flops: float, vec_ops: float, mem_bytes: float,
                  hw: HardwareConfig, comm_time_s: float = 0.0,
                  comm_bytes: float = 0.0) -> OpProfile:
    """Compute roofline time for an operation.

    bottleneck = argmax(cube_time, vec_time, mem_time).
    total time = max(cube, vec, mem) + comm.
    """
    cube_time = flops / (hw.cube_tflops * 1e12 * hw.flops_utilization) if flops > 0 else 0.0
    vec_time = vec_ops / (hw.vec_tflops * 1e12 * hw.flops_utilization) if vec_ops > 0 else 0.0
    mem_time = mem_bytes / (hw.hbm_bandwidth_gbps * 1e9 * hw.hbm_bw_utilization) if mem_bytes > 0 else 0.0

    compute_time = max(cube_time, vec_time, mem_time)
    if compute_time == 0.0 and comm_time_s == 0.0:
        bottleneck = ""
    elif comm_time_s > compute_time:
        bottleneck = "COMM"
    elif cube_time >= vec_time and cube_time >= mem_time:
        bottleneck = "CUBE"
    elif vec_time >= mem_time:
        bottleneck = "VEC"
    else:
        bottleneck = "MEM"

    return OpProfile(
        name=name,
        flops=flops,
        vec_ops=vec_ops,
        mem_bytes=mem_bytes,
        comm_bytes=comm_bytes,
        cube_time_s=cube_time,
        vec_time_s=vec_time,
        mem_time_s=mem_time,
        comm_time_s=comm_time_s,
        time_s=compute_time + comm_time_s,
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
     # FIXME: the alltoall should consider load imbalance factor => more volume.
    if n <= 1:
        return 0.0
    factor = (n - 1) / n
    return factor * vol_bytes / (bw_gbps * 1e9 * bw_util) + latency_us * 1e-6


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
    # Bottleneck for the aggregate = largest component
    times = {"CUBE": total.cube_time_s, "VEC": total.vec_time_s,
             "MEM": total.mem_time_s, "COMM": total.comm_time_s}
    total.bottleneck = max(times, key=times.get) if total.time_s > 0 else ""
    return total


def bytes2(count: int) -> float:
    """BF16 bytes for `count` elements."""
    return count * 2.0
