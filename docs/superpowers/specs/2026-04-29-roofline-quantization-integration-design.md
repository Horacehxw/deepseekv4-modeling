# Roofline Quantization Integration Design

## Goal

Move quantization timing into the core roofline path using the low-risk scheme A:
operation kind drives coarse quantization adjustments inside `roofline_time()`.
Delete `perf_model/quantization.py` and remove phase-level quantization post-processing.

This is a one-shot refactor. There is no backward compatibility layer and no
temporary transition path.

## Constraints

- Keep the current coarse quantization semantics.
- Do not introduce role-based memory traffic in this refactor.
- Write tests first, then refactor.
- Default BF16 behavior must match the current unquantized behavior.
- W8A8/KV8/KV4 behavior must match the existing `quantization.py` post-process
  behavior as closely as possible.
- `quantization.py` is removed from the public API.

## Current Problem

The current implementation has two timing paths:

```text
ops.py
  -> roofline.py
       produces BF16 OpProfile
  -> quantization.py
       recomputes timings from OpProfile
```

That duplicates the roofline formula in `quantization.py` and makes serving
metrics depend on a special post-processing path. Other callers such as
`main.py`, `param_search`, and older reports can bypass quantized timing
entirely.

## Target Architecture

```text
Config
  hw: HardwareConfig
  rt: RuntimeConfig
      quant_mode
      kv_cache_quant_mode
        |
        v
ops.py
  computes flops, vec_ops, BF16-style mem_bytes, op_kind
        |
        v
roofline_time(..., cfg, op_kind)
  applies coarse quantization policy
  computes cube/vec/mem/comm time once
        |
        v
OpProfile
  final timing and final physical mem_bytes
```

## Config Boundary

`HardwareConfig` describes card capability only:

```text
cube_tflops
vec_tflops
hbm_bandwidth_gbps
hbm_capacity_gb
cube_utilization
vec_utilization
hbm_bw_utilization
w8a8_tflops
```

`Config` describes a full run:

```text
Config
  hw: HardwareConfig
  net: NetworkConfig
  model: ModelConfig
  rt: RuntimeConfig
      quant_mode
      kv_cache_quant_mode
      batch_size / tp / ep / dp
      seq_len / output_len / prefix cache / mtp
```

Quantization requires both hardware and runtime fields, so the new
`roofline_time()` accepts `Config`, not `HardwareConfig | Config`.

## Roofline API

`roofline_time()` becomes the only public timing entry point:

```python
def roofline_time(
    name: str,
    flops: float,
    vec_ops: float,
    mem_bytes: float,
    cfg: Config,
    op_kind: str,
    comm_time_s: float = 0.0,
    comm_bytes: float = 0.0,
) -> OpProfile:
    ...
```

Valid `op_kind` values:

```text
gemm
attention
vector
comm
other
```

`op_kind` is explicit. The refactor does not keep `infer_op_kind(name)` as a
runtime behavior. To preserve current behavior, `ops.py` assigns kinds using
the same categories that `quantization.py` currently infers from names.

## Roofline Policy

Constants move to `roofline.py`:

```python
WEIGHT_BYTE_RATIOS = {"bf16": 1.0, "w8a8": 0.5}
KV_BYTE_RATIOS = {"bf16": 1.0, "kv8": 0.5, "kv4": 0.25}
OP_KINDS = {"gemm", "attention", "vector", "comm", "other"}
```

Timing policy:

```text
gemm + quant_mode=w8a8
  cube_tflops = cfg.hw.effective_w8a8_tflops
  mem_bytes *= 0.5

attention + kv_cache_quant_mode=kv8
  mem_bytes *= 0.5

attention + kv_cache_quant_mode=kv4
  mem_bytes *= 0.25

comm
  unchanged

vector / other
  unchanged
```

The attention rule deliberately preserves the existing coarse behavior: the
whole attention `mem_bytes` field is scaled by the KV ratio. It does not split
Q/O activations from KV cache traffic. That precision belongs to a future
role-based memory model, not this refactor.

## OpProfile Semantics

`OpProfile` fields remain structurally unchanged.

After the refactor:

```text
flops        logical cube/matmul FLOPs
vec_ops      logical vector FLOPs
mem_bytes    final physical HBM bytes after coarse quant policy
time_s       max(cube_time_s, vec_time_s, mem_time_s) + comm_time_s
```

`bytes2()` and `bytes4()` remain raw byte helpers. They do not apply
quantization policy.

Communication helpers remain unchanged:

```text
allreduce_time()
alltoall_time()
allgather_time()
```

`sum_ops()` remains unchanged because it aggregates already-final
`OpProfile` values.

## Ops Changes

All `ops.py` calls switch from `cfg.hw` to `cfg` and pass explicit `op_kind`.

Examples:

```python
return roofline_time("q_proj_dq", flops, 0, mem_bytes, cfg, op_kind="gemm")

return roofline_time("attention_swa", flops, vec_ops, mem_bytes, cfg,
                     op_kind="attention")

return roofline_time("rmsnorm_attn", 0, vec_ops, mem_bytes, cfg,
                     op_kind="vector")

return roofline_time("kv_compression", cube_flops, vec_ops, mem_bytes, cfg,
                     op_kind="other")
```

Direct `OpProfile(...)` communication constructors can remain direct because
they already contain final communication time and bytes.

To preserve current quantized behavior, classify ops as follows:

```text
gemm
  q_proj_dq
  q_proj_uq
  kv_proj
  wo_a
  wo_b
  index_iq_proj
  moe_gate
  routed_gate_proj
  routed_up_proj
  routed_down_proj
  shared_gate_proj
  shared_up_proj
  shared_down_proj
  embedding
  lm_head

attention
  attention_swa
  attention_comp

vector
  rmsnorm*
  mhc_*
  sinkhorn*
  routed_silu_mul
  shared_silu_mul

other
  index_kv_compress
  index_kv_compress_decode
  index_score
  kv_compression
  kv_compression_decode
  shared_expert_excess manual timing remains direct OpProfile
```

`final_rmsnorm` should be passed as `vector` when produced by `op_rmsnorm()`.

## Memory API

`memory.py` keeps the existing public names and makes them runtime-aware:

```python
def weight_memory_per_rank(cfg: Config) -> dict:
    ...

def kv_cache_memory(cfg: Config) -> dict:
    ...
```

Default BF16 behavior must be identical to the current output:

```text
quant_mode=bf16
kv_cache_quant_mode=bf16
weight_scale_overhead_bytes=0
kv_scale_overhead_bytes=0
  -> same bytes and same keys as before
```

Quantized behavior:

```text
weight_memory_per_rank(cfg)
  base BF16 bytes * WEIGHT_BYTE_RATIOS[cfg.rt.quant_mode]
  total += cfg.rt.weight_scale_overhead_bytes
  includes "quant_mode"

kv_cache_memory(cfg)
  byte fields * KV_BYTE_RATIOS[cfg.rt.kv_cache_quant_mode]
  total_bytes += cfg.rt.kv_scale_overhead_bytes
  includes "kv_cache_quant_mode"
```

The mode metadata keys are required for quantized modes. The default BF16,
zero-overhead path preserves the previous public shape exactly so existing BF16
memory tests and callers remain semantically unchanged.

Internal raw helpers may exist:

```python
def _weight_memory_per_rank_bf16(cfg: Config) -> dict:
    ...

def _kv_cache_memory_bf16(cfg: Config) -> dict:
    ...
```

The public `quantized_weight_memory_per_rank()` and
`quantized_kv_cache_memory()` helpers are removed.

## Serving Changes

`serving.py` stops post-processing phases:

```python
phase = prefill_model(compute_cfg)
first_phase = decode_step(first_context, compute_cfg)
last_phase = decode_step(last_context, compute_cfg)
```

`_hbm_metrics()` uses the runtime-aware public memory API:

```python
wm = weight_memory_per_rank(cfg)
kv = kv_cache_memory(cfg)
```

There must be no `quantize_phase_profile()` call after the refactor.

## Decode Aggregate Semantics

`decode_model()` remains valid. It builds totals from `decode_step()` results.
Once `decode_step()` produces quant-aware `OpProfile` values, aggregate totals
are quant-aware naturally.

The implementation must not reconstruct a decode aggregate by summing the
representative first-step detailed ops. The existing sampling/interpolation
logic remains the source of truth.

## Deleted API

Remove `perf_model/quantization.py`.

Remove exports from `perf_model/__init__.py`:

```text
infer_op_kind
quantized_weight_memory_per_rank
quantized_kv_cache_memory
quantize_op_profile
quantize_phase_profile
```

Tests should import quantization-related constants or behavior from
`roofline.py` and memory behavior from `memory.py`.

## TDD Plan

Write failing tests first:

1. BF16 roofline compatibility
   - `roofline_time(..., cfg=default BF16, op_kind="gemm")` matches the old
     raw cube/vec/mem formula.

2. GEMM W8A8
   - W8A8 GEMM uses `effective_w8a8_tflops`.
   - W8A8 GEMM scales `mem_bytes` by `0.5`.

3. Attention KV quant
   - `kv8` attention scales the whole attention `mem_bytes` by `0.5`.
   - `kv4` attention scales the whole attention `mem_bytes` by `0.25`.
   - Attention compute throughput remains BF16.

4. Vector and other unchanged
   - `op_kind="vector"` does not scale memory or cube throughput.
   - `op_kind="other"` does not scale memory or cube throughput.

5. Communication unchanged
   - Direct communication `OpProfile` behavior remains unchanged.
   - `op_kind="comm"` passed to `roofline_time()` does not scale memory.

6. Memory BF16 compatibility
   - `weight_memory_per_rank()` and `kv_cache_memory()` return the current BF16
     expected values under default runtime config.

7. Memory quantization
   - W8A8 weight memory equals BF16 total * 0.5 plus overhead.
   - KV8/KV4 cache memory equals BF16 total * ratio plus overhead.

8. Serving no double quantization
   - `evaluate_prefill_serving()` time equals direct `prefill_model()` time for
     its compute config.
   - `evaluate_decode_serving()` uses quant-aware first/last `decode_step()`
     without `quantize_phase_profile()`.

9. Deleted module
   - No production code imports `perf_model.quantization`.

## Verification

Run focused tests:

```bash
python -m unittest test.test_roofline test.test_ops test.test_memory test.test_serving test.test_report_0428 -v
```

Run full suite:

```bash
python -m unittest discover -v
```

Run report smoke checks:

```bash
python main.py configs/device_910C.json configs/network_910C.json configs/model_deepseekv4.json configs/runtime_deepseekv4.json
python report/0428/script/generate_report.py
```

Known baseline note: current `origin/master` has existing `test.test_ops`
formula failures in this worktree. The implementation plan must account for
that baseline when interpreting full-suite results.

## Documentation Updates

Update documentation that currently says all sizes are BF16:

```text
README.md
README_zh.md
CLAUDE.md
```

The new wording should state:

```text
Default runtime is BF16. When runtime quantization is configured, roofline
timing and memory sizing apply the configured coarse quantization policy.
```

0428 report text may keep its W8A8/KV8 assumptions, but should no longer imply
that quantization is a serving-only post-processing path.
