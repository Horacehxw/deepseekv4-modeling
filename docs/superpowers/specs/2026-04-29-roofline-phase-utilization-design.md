# Roofline Phase Utilization Design

## Context

The hardware config needs phase-specific efficiency knobs for prefill and decode, plus a static vector latency term. The current production `configs/device_910C.json` already demonstrates the desired fields:

- `prefill_utilization`
- `decode_utilization`
- `vec_static_latency_us`

Current code does not yet accept these keys, so loading the updated `device_910C.json` fails with an unexpected keyword error. The existing roofline formula is also still `max(cube_time, vec_time, mem_time) + comm_time`.

## Requirements

- Add hardware config fields:
  - `prefill_utilization`, default `1.0`
  - `decode_utilization`, default `0.8`
  - `vec_static_latency_us`, default `10.0`
- Interpret prefill/decode utilization as an additional effective utilization multiplier for all non-communication operator compute and HBM components in that phase.
- Add `vec_static_latency_us` to `vec_time_s` only when `vec_ops > 0`.
- Change per-op compute time from `max(cube_time, vec_time, mem_time)` to `max(cube_time + vec_time, mem_time)`.
- Keep communication time additive: `time_s = compute_time + comm_time_s`.
- Keep existing bottleneck labels (`CUBE`, `VEC`, `MEM`, `COMM`, empty) for report compatibility.
- Keep non-phase direct calls to `roofline_time()` backward compatible.

## Design

`HardwareConfig` will own the new fields and defaults. Existing hardware configs that omit the new keys remain valid.

`roofline_time()` will accept an optional `phase` argument:

- `phase="prefill"` uses `hw.prefill_utilization`
- `phase="decode"` uses `hw.decode_utilization`
- `phase=None` uses `1.0` for backward-compatible direct calls and tests unless a caller opts in

The effective formulas become:

```text
cube_time = flops / (cube_tflops * 1e12 * cube_util * phase_util)
vec_time = vec_ops / (vec_tflops * 1e12 * vec_util * phase_util) + vec_static_latency
mem_time = mem_bytes / (hbm_bandwidth_gbps * 1e9 * hbm_bw_util * phase_util)
compute_time = max(cube_time + vec_time, mem_time)
total_time = compute_time + comm_time
```

The vector static latency term is only included when `vec_ops > 0`.

`prefill_layer()`, `prefill_model()`, `decode_layer()`, and `decode_step()` will call operator builders in a phase context. The lowest-risk implementation is to thread phase through `roofline_time()` calls by adding a phase parameter to op builders only where needed, or by using a small phase-aware wrapper around `Config`/hardware access. The implementation should prefer the smallest call-surface change that still makes phase explicit.

`perf_model.quantization._with_roofline_timings()` must use the same phase-aware timing formula because quantized serving evaluation recomputes op timings instead of calling `roofline_time()` directly. `quantize_phase_profile()` can infer the phase from `PhaseProfile.phase` (`prefill`, `decode_step@...`, or `decode_total`) and pass it through.

## Bottleneck Semantics

Because cube and vector time are now serial on the compute side, bottleneck selection should compare `compute_side = cube_time + vec_time` against `mem_time`.

- If communication exceeds compute time, bottleneck is `COMM`.
- If memory exceeds `cube_time + vec_time`, bottleneck is `MEM`.
- Otherwise the op is compute-bound; report `CUBE` when `cube_time >= vec_time`, else `VEC`.

This preserves current report labels while reflecting the new compute-vs-memory decision.

`sum_ops()` should aggregate component times as before, but determine aggregate bottleneck using the same `cube + vec` versus `mem` rule.

## Testing

Update or add focused tests for:

- `HardwareConfig` defaults and JSON loading with the updated `device_910C.json`.
- `vec_static_latency_us` applies only when `vec_ops > 0`.
- Per-op total time uses `max(cube_time + vec_time, mem_time) + comm_time`.
- Prefill and decode phases apply different utilization multipliers.
- `sum_ops()` aggregate bottleneck follows the new compute-side rule.
- Quantization recomputation follows the same timing formula and preserves aggregate decode handling.

Run at least:

```bash
python -m unittest test.test_config test.test_roofline test.test_quantization
python -m unittest
```

## Compatibility

The public report fields remain unchanged. Existing CSV columns (`cube_time_ms`, `vec_time_ms`, `mem_time_ms`, `total_time_ms`, `bottleneck`) remain valid, but `total_time_ms` will no longer equal `max(cube, vec, mem) + comm`; it will equal `max(cube + vec, mem) + comm`.
