# Roofline Quantization Integration

## Goal Description

Move quantization timing into the core roofline path. `roofline_time()` gains `cfg: Config` and `op_kind: str` parameters and applies coarse quantization policy directly — eliminating the duplicate roofline formula in `quantization.py` and the serving-only `quantize_phase_profile()` post-processing path. `perf_model/quantization.py` is deleted in full. `memory.py` applies quantization ratios in its base public functions. This is a one-shot refactor with no backward compatibility layer and no temporary transition path.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: BF16 roofline numerical identity — the new `roofline_time()` produces identical timings to the current implementation when called with a default BF16 config.
  - Positive Tests (expected to PASS):
    - `roofline_time(name, flops, 0, mem_bytes, cfg_bf16, op_kind="gemm")` returns `cube_time_s`, `vec_time_s`, `mem_time_s`, `time_s` equal to the old `roofline_time(name, flops, 0, mem_bytes, cfg.hw)` result.
    - BF16 identity holds for all five op_kind values: `"gemm"`, `"attention"`, `"vector"`, `"comm"`, `"other"`.
    - `OpProfile.bottleneck` is unchanged between old and new call for BF16 configs.
  - Negative Tests (expected to FAIL):
    - `roofline_time(name, flops, 0, mem_bytes, cfg_w8a8, op_kind="gemm")` returns a different `cube_time_s` and `mem_time_s` than the BF16 result (quantization is applied).
    - `roofline_time(name, flops, 0, mem_bytes, cfg_kv8, op_kind="attention")` returns different `mem_time_s` than BF16 (kv quant applied).

- AC-2: W8A8 GEMM quantization policy — GEMM ops under `quant_mode=w8a8` use `effective_w8a8_tflops` and scale `mem_bytes` by exactly `0.5`.
  - Positive Tests (expected to PASS):
    - `roofline_time(..., cfg_w8a8, op_kind="gemm").cube_time_s == flops / (cfg.hw.effective_w8a8_tflops * 1e12)`.
    - `roofline_time(..., cfg_w8a8, op_kind="gemm").mem_time_s == (mem_bytes * 0.5) / (cfg.hw.hbm_bandwidth_gbps * 1e9 * cfg.hw.hbm_bw_utilization)`.
    - W8A8 GEMM `OpProfile.mem_bytes` equals `mem_bytes * 0.5`.
  - Negative Tests (expected to FAIL):
    - `roofline_time(..., cfg_bf16, op_kind="gemm")` does not use `w8a8_tflops` (uses `cube_tflops`).
    - `roofline_time(..., cfg_w8a8, op_kind="vector")` is not scaled — vector is unaffected by `quant_mode`.

- AC-3: Attention KV quantization policy — attention ops under KV quant modes scale `mem_bytes` by the exact KV ratio; compute throughput remains BF16.
  - Positive Tests (expected to PASS):
    - `roofline_time(..., cfg_kv8, op_kind="attention").mem_time_s == (mem_bytes * 0.5) / bw_effective`.
    - `roofline_time(..., cfg_kv4, op_kind="attention").mem_time_s == (mem_bytes * 0.25) / bw_effective`.
    - Attention `cube_time_s` is identical across BF16, KV8, and KV4 configs (compute throughput unchanged).
    - Attention `OpProfile.mem_bytes` equals `mem_bytes * 0.5` for KV8 and `mem_bytes * 0.25` for KV4.
  - Negative Tests (expected to FAIL):
    - `kv_cache_quant_mode` setting has no effect on `op_kind="gemm"` (gemm only responds to `quant_mode`).
    - `kv_cache_quant_mode` setting has no effect on `op_kind="vector"` or `op_kind="other"`.

- AC-4: Vector and other ops unchanged — `op_kind="vector"` and `op_kind="other"` produce identical results regardless of `quant_mode` or `kv_cache_quant_mode`.
  - Positive Tests (expected to PASS):
    - `roofline_time(..., cfg_bf16, op_kind="vector")` equals `roofline_time(..., cfg_w8a8, op_kind="vector")`.
    - `roofline_time(..., cfg_bf16, op_kind="other")` equals `roofline_time(..., cfg_kv4, op_kind="other")`.
  - Negative Tests (expected to FAIL):
    - Changing `quant_mode` from `"bf16"` to `"w8a8"` while `op_kind="vector"` changes the result.

- AC-5: Comm unchanged — `op_kind="comm"` is unaffected by quantization modes; direct `OpProfile(...)` comm constructors remain unmodified.
  - Positive Tests (expected to PASS):
    - `roofline_time(..., cfg_w8a8, op_kind="comm", comm_time_s=t, comm_bytes=b)` returns the same `time_s` as with BF16 config.
    - `allreduce_time()`, `alltoall_time()`, `allgather_time()` return identical values pre- and post-refactor.
    - Direct `OpProfile(...)` constructors used in comm and `shared_expert_excess` produce identical results.
  - Negative Tests (expected to FAIL):
    - `roofline_time(..., cfg_w8a8, op_kind="comm")` produces different `time_s` than BF16.

- AC-6: Memory BF16 identity — `weight_memory_per_rank()` and `kv_cache_memory()` return byte-for-byte identical dicts to the current implementation under default BF16 config.
  - Positive Tests (expected to PASS):
    - `weight_memory_per_rank(cfg_bf16)` matches the current frozen fixture dict exactly (same keys, same values, same types).
    - `kv_cache_memory(cfg_bf16)` matches the current frozen fixture dict exactly.
    - BF16 outputs do NOT contain `"quant_mode"` or `"kv_cache_quant_mode"` metadata keys (preserving previous public shape).
  - Negative Tests (expected to FAIL):
    - `weight_memory_per_rank(cfg_w8a8)["total"]` differs from `weight_memory_per_rank(cfg_bf16)["total"]` (quantization applied).

- AC-7: Memory quantization with exact ratios and overhead — quantized memory functions apply exact data ratios and expose scale overhead bytes separately.
  - Positive Tests (expected to PASS):
    - `weight_memory_per_rank(cfg_w8a8)["total"] == bf16_total * 0.5` (exact float equality; data-only total).
    - `kv_cache_memory(cfg_kv8)["total_bytes"] == bf16_kv_total * 0.5` (exact float equality; data-only total).
    - `kv_cache_memory(cfg_kv4)["total_bytes"] == bf16_kv_total * 0.25` (exact float equality; data-only total).
    - `weight_memory_total_bytes(weight_memory_per_rank(cfg_w8a8)) == bf16_total * 0.5 + cfg.rt.weight_scale_overhead_bytes`.
    - `kv_cache_total_bytes(kv_cache_memory(cfg_kv8)) == bf16_kv_total * 0.5 + cfg.rt.kv_scale_overhead_bytes`.
    - Non-BF16 `weight_memory_per_rank` output contains `"quant_mode"` key with the active mode string.
    - Non-BF16 `kv_cache_memory` output contains `"kv_cache_quant_mode"` key with the active mode string.
  - Negative Tests (expected to FAIL):
    - `kv_cache_total_bytes(kv_cache_memory(cfg_kv8)) == bf16_kv_total * 0.5` when `kv_scale_overhead_bytes` is non-zero.

- AC-8: Serving no double quantization — serving evaluation outputs match golden values from the current `quantize_phase_profile()` path; no `quantize_phase_profile()` call exists after the refactor.
  - Positive Tests (expected to PASS):
    - `evaluate_prefill_serving()` output for a fixed W8A8 config matches the golden value captured from the current path (within `1e-9` relative tolerance).
    - `evaluate_decode_serving()` output for a fixed KV8 config matches the golden value captured from the current path (within `1e-9` relative tolerance).
    - `evaluate_prefill_serving()` BF16 output is numerically unchanged from the current BF16 serving output.
  - Negative Tests (expected to FAIL):
    - `perf_model.serving` or `perf_model` exposes `quantize_phase_profile`.

- AC-9: `perf_model.quantization` module is fully deleted and no longer importable.
  - Positive Tests (expected to PASS):
    - `import perf_model.quantization` raises `ImportError`.
    - `from perf_model import quantize_phase_profile` raises `ImportError`.
    - `from perf_model import infer_op_kind` raises `ImportError`.
  - Negative Tests (expected to FAIL):
    - Any file in `perf_model/` imports from `perf_model.quantization` (grep-verified, zero matches required).
    - Any file in `test/` imports from `perf_model.quantization` (grep-verified, zero matches required).

- AC-10: Complete op_kind audit — every `roofline_time()` call in `ops.py` passes an explicit, valid `op_kind`; invalid `op_kind` raises `ValueError`.
  - Positive Tests (expected to PASS):
    - All `roofline_time()` calls in `ops.py` are verified to pass an `op_kind` argument matching the classification table in the design spec.
    - `roofline_time(..., op_kind="invalid_kind")` raises `ValueError`.
    - `roofline_time(..., op_kind="gemm")` for an op that is supposed to be `"vector"` would change the timing (test by parameterizing a borderline op).
  - Negative Tests (expected to FAIL):
    - Any `roofline_time()` call exists that omits `op_kind` (grep-verified, zero matches allowed).


## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation covers the full scope described in the design: `roofline_time()` with `cfg: Config` and `op_kind: str`, policy constants (`WEIGHT_BYTE_RATIOS`, `KV_BYTE_RATIOS`, `OP_KINDS`) in `roofline.py`, all `ops.py` call sites updated with explicit `op_kind`, `memory.py` base functions quantization-aware with internal `_bf16` helpers, `serving.py` free of `quantize_phase_profile()` calls, `quantization.py` fully deleted, `test_quantization.py` deleted, all five exports removed from `__init__.py`, documentation updated in `README.md`, `README_zh.md`, and `CLAUDE.md`.

### Lower Bound (Minimum Acceptable Scope)

The minimum acceptable implementation satisfies all 10 acceptance criteria: `roofline_time()` applies correct quantization policy inline, all `ops.py` callers pass explicit `op_kind`, `memory.py` functions apply quant ratios, `serving.py` has no `quantize_phase_profile()` calls, and `quantization.py` is deleted with all exports removed. Documentation updates are required.

> **Deterministic Design**: Upper and lower bounds converge. This design has no optional elements — every item in the spec is required.

### Allowed Choices

- Can use: `dataclasses.replace()` for constructing modified `OpProfile` instances; internal `_bf16` helpers in `memory.py`; string literals for `op_kind`; pytest fixtures and the existing test helper patterns from `test/helpers.py`
- Cannot use: `infer_op_kind()` or any runtime name-based inference for `op_kind`; backward-compatibility shims or re-exports that preserve deleted function names; role-based memory traffic modeling (explicitly deferred); temporary transition paths; any module-level import of `quantization.py` symbols after Milestone 5


## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The refactor threads `cfg: Config` and `op_kind: str` through `roofline_time()` and adds a policy dispatch at the top of the function before the existing roofline arithmetic:

```python
# Add to roofline.py

WEIGHT_BYTE_RATIOS = {"bf16": 1.0, "w8a8": 0.5}
KV_BYTE_RATIOS = {"bf16": 1.0, "kv8": 0.5, "kv4": 0.25}
OP_KINDS = {"gemm", "attention", "vector", "comm", "other"}

def roofline_time(
    name: str,
    flops: float,
    vec_ops: float,
    mem_bytes: float,
    cfg: Config,          # was: hw: HardwareConfig
    op_kind: str,         # new required parameter
    comm_time_s: float = 0.0,
    comm_bytes: float = 0.0,
) -> OpProfile:
    if op_kind not in OP_KINDS:
        raise ValueError(f"op_kind must be one of {OP_KINDS!r}, got {op_kind!r}")

    hw = cfg.hw
    effective_cube_tflops = hw.cube_tflops * hw.cube_utilization * hw.flops_utilization

    if op_kind == "gemm" and cfg.rt.quant_mode == "w8a8":
        effective_cube_tflops = hw.effective_w8a8_tflops
        mem_bytes = mem_bytes * WEIGHT_BYTE_RATIOS["w8a8"]
    elif op_kind == "attention" and cfg.rt.kv_cache_quant_mode != "bf16":
        mem_bytes = mem_bytes * KV_BYTE_RATIOS[cfg.rt.kv_cache_quant_mode]

    # ... existing roofline arithmetic using effective_cube_tflops and mem_bytes ...
```

For `memory.py`, the pattern is: compute the BF16 base dict via an internal `_bf16` helper, then apply the ratio and overhead before returning:

```python
def weight_memory_per_rank(cfg: Config) -> dict:
    result = _weight_memory_per_rank_bf16(cfg)
    if cfg.rt.quant_mode == "bf16":
        return result                             # identical to current output
    ratio = WEIGHT_BYTE_RATIOS[cfg.rt.quant_mode]
    # scale all byte fields, add overhead to total, add metadata key
    ...
```

For `serving.py`, remove the three `quantize_phase_profile()` call wrappers — the results of `prefill_model()` and `decode_step()` are already quant-aware after the roofline refactor.

### Relevant References

- `perf_model/roofline.py` — Current `roofline_time()` signature (`hw: HardwareConfig`, no `op_kind`); `OpProfile` dataclass
- `perf_model/quantization.py` — `GEMM_NAMES`, `ATTENTION_NAMES`, `VECTOR_PREFIXES`, `COMM_NAMES` sets define current op classification; `WEIGHT_BYTE_RATIOS`/`KV_BYTE_RATIOS` constants to migrate to `roofline.py`
- `perf_model/ops.py` — ~30 call sites, all currently pass `cfg.hw`; classification table in design spec maps every call to its `op_kind`
- `perf_model/memory.py` — `weight_memory_per_rank(cfg)` and `kv_cache_memory(cfg)` already accept `Config`; add quantization application
- `perf_model/serving.py` — Three `quantize_phase_profile()` call sites to remove
- `perf_model/config.py` — `HardwareConfig.effective_w8a8_tflops` property; `RuntimeConfig.quant_mode`, `RuntimeConfig.kv_cache_quant_mode`, `RuntimeConfig.weight_scale_overhead_bytes`, `RuntimeConfig.kv_scale_overhead_bytes`
- `test/test_roofline.py` — Also calls `roofline_time()` directly; needs signature update in Milestone 3
- `test/test_quantization.py` — Tests the module being deleted; deleted in Milestone 6


## Dependencies and Sequence

### Milestones

1. **Milestone 0 — Pre-flight Audit** (no code changes): Verify full scope before any implementation begins.
   - Grep `perf_model/` and `test/` for all `roofline_time()` call sites (expected: `ops.py` ~30, `test_roofline.py` direct calls)
   - Grep `perf_model/` and `test/` for all files importing `quantization.py` symbols (expected: `serving.py`, `__init__.py`, `test_quantization.py`)
   - Compare the op_kind classification table from the design spec against `GEMM_NAMES`, `ATTENTION_NAMES`, `VECTOR_PREFIXES`, `COMM_NAMES` in `quantization.py`
   - Document deletion checklist: files to delete, exports to remove, call sites to update

2. **Milestone 1 — Write Failing Tests** (TDD red phase): All new tests fail on the current unmodified code.
   - `test_roofline.py`: tests for new `roofline_time()` signature (BF16 compat, W8A8, KV8/KV4, vector/other, comm, `ValueError` for invalid `op_kind`)
   - `test_memory.py`: tests for quantization-aware memory functions (BF16 identity fixtures, W8A8/KV8/KV4 exact ratios, overhead accounting, metadata keys)
   - `test_serving.py`: golden-value behavioral tests for BF16, W8A8, and KV8 serving configs; capture golden values from current `quantize_phase_profile()` path first
   - Depends on: Milestone 0 (scope confirmed)

3. **Milestone 2 — Refactor `roofline_time()`** (green for roofline tests): Change signature, add constants, apply policy, add validation.
   - Update `roofline_time()` signature from `(hw: HardwareConfig)` to `(cfg: Config, op_kind: str)`
   - Add `WEIGHT_BYTE_RATIOS`, `KV_BYTE_RATIOS`, `OP_KINDS` to `roofline.py`
   - Implement quantization policy dispatch; add `op_kind` validation
   - Update existing `test_roofline.py` fixtures that call the old signature
   - Depends on: Milestone 1

4. **Milestone 3 — Update All `roofline_time()` Callers**: All call sites switch to `cfg` + explicit `op_kind`.
   - Update all `ops.py` call sites: `cfg.hw` → `cfg`, add explicit `op_kind` per the classification table
   - Update any other callers found in Milestone 0 audit (e.g., direct calls in `test_roofline.py`)
   - Verify: grep `perf_model/` and `test/` confirms zero `cfg.hw`-style `roofline_time()` calls remain
   - Depends on: Milestone 2

5. **Milestone 4 — Update `memory.py`**: Base public functions become quantization-aware; BF16 output is identical.
   - Apply `WEIGHT_BYTE_RATIOS` ratio in `weight_memory_per_rank()` and report `weight_scale_overhead_bytes` separately as `scale_overhead_bytes`
   - Apply `KV_BYTE_RATIOS` ratio in `kv_cache_memory()` and report `kv_scale_overhead_bytes` separately as `scale_overhead_bytes`
   - Add shared accessors for HBM callers to include separately reported scale overhead
   - Add internal `_weight_memory_per_rank_bf16()` and `_kv_cache_memory_bf16()` helpers
   - BF16 default code path must return byte-for-byte identical output to current
   - Note: `quantized_weight_memory_per_rank()` and `quantized_kv_cache_memory()` live in `quantization.py` and are deleted in Milestone 6, not here
   - Depends on: Milestone 1 (memory tests must be written first)

6. **Milestone 5 — Update `serving.py`**: Remove `quantize_phase_profile()` usage.
   - Remove three `quantize_phase_profile()` call sites (prefill, first decode, last decode)
   - Remove `quantize_phase_profile` import
   - Behavioral serving tests from Milestone 1 verify correctness
   - Depends on: Milestone 3 (roofline quant-aware) + Milestone 4 (memory quant-aware)

7. **Milestone 6 — Delete `quantization.py` and Clean Exports**: Final deletion after all callers are migrated.
   - Pre-deletion: grep `perf_model/` and `test/` for any remaining imports of `quantization` module symbols (zero expected)
   - Delete `perf_model/quantization.py`
   - Delete `test/test_quantization.py`
   - Remove from `perf_model/__init__.py`: `infer_op_kind`, `quantized_weight_memory_per_rank`, `quantized_kv_cache_memory`, `quantize_op_profile`, `quantize_phase_profile`
   - Depends on: Milestone 5

8. **Milestone 7 — Verification and Documentation**: Confirm correctness end-to-end; update docs.
   - Run focused suite: `python -m unittest test.test_roofline test.test_ops test.test_memory test.test_serving test.test_report_0428 -v`
   - Confirm pre-existing `test.test_ops` formula failures are unchanged (document as baseline, not regression)
   - Run full suite: `python -m unittest discover -v`
   - Verify non-serving callers (`report.py`, `report/analyze_scenarios.py`, `param_search/search.py`) produce identical BF16 outputs before and after
   - Run smoke checks: `main.py` and report generator
   - Update `README.md`, `README_zh.md`, `CLAUDE.md`: replace "all sizes use BF16" with quant-aware wording
   - Depends on: Milestone 6


## Task Breakdown

Each task includes exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task-audit | Grep `perf_model/` and `test/` for all `roofline_time()` callers and `quantization.py` importers; compare op_kind table vs `GEMM_NAMES`/`ATTENTION_NAMES`/`VECTOR_PREFIXES`; produce deletion checklist | AC-9, AC-10 | analyze | — |
| task-golden | Capture golden output values from current code: BF16 and W8A8 `evaluate_prefill_serving()`, KV8 `evaluate_decode_serving()`, BF16/W8A8 `weight_memory_per_rank()`, BF16/KV8/KV4 `kv_cache_memory()` | AC-7, AC-8 | coding | task-audit |
| task-test-roofline | Write failing tests in `test_roofline.py` for new `roofline_time()` signature: BF16 compat (all 5 op_kinds), W8A8 exact ratio, KV8/KV4 exact ratio, vector/other unaffected, comm unaffected, `ValueError` for invalid op_kind | AC-1, AC-2, AC-3, AC-4, AC-5, AC-10 | coding | task-audit |
| task-test-memory | Write failing tests in `test_memory.py`: BF16 identity (fixture comparison), W8A8 weight exact ratio, KV8/KV4 cache exact ratios, overhead added to total, metadata keys present/absent | AC-6, AC-7 | coding | task-golden |
| task-test-serving | Write failing behavioral tests in `test_serving.py` using golden values: BF16, W8A8, KV8 serving outputs; static grep assertion for no `quantize_phase_profile` in `serving.py` | AC-8, AC-9 | coding | task-golden |
| task-roofline-impl | Change `roofline_time()` signature to `(cfg: Config, op_kind: str, ...)`; add `WEIGHT_BYTE_RATIOS`, `KV_BYTE_RATIOS`, `OP_KINDS` to `roofline.py`; apply policy dispatch; add `ValueError` validation | AC-1, AC-2, AC-3, AC-4, AC-5, AC-10 | coding | task-test-roofline |
| task-roofline-fixtures | Update existing `test_roofline.py` fixtures that use the old `roofline_time(hw=...)` signature | AC-1 | coding | task-roofline-impl |
| task-ops-update | Update all ~30 `ops.py` call sites: `cfg.hw` → `cfg`, add explicit `op_kind` per classification table; verify grep shows zero old-style calls in `perf_model/` | AC-10 | coding | task-roofline-impl |
| task-other-callers | Update any other `roofline_time()` callers found in task-audit (e.g., direct calls in `test_roofline.py`) | AC-10 | coding | task-ops-update |
| task-memory-impl | Apply `WEIGHT_BYTE_RATIOS` / `KV_BYTE_RATIOS` in public memory functions, report scale overhead separately, add shared total-with-overhead accessors and internal `_bf16` helpers; BF16 path returns exact current output | AC-6, AC-7 | coding | task-test-memory |
| task-serving-clean | Remove three `quantize_phase_profile()` call sites from `serving.py`; remove the import | AC-8 | coding | task-ops-update, task-memory-impl |
| task-ref-audit | Grep `perf_model/` and `test/` for any remaining `quantization` module references before deletion; confirm zero remaining importers | AC-9 | analyze | task-serving-clean |
| task-delete-quant | Delete `perf_model/quantization.py`; delete `test/test_quantization.py`; remove 5 exports from `perf_model/__init__.py` | AC-9 | coding | task-ref-audit |
| task-verify-focused | Run focused test suite; confirm new tests pass; confirm pre-existing `test_ops` baseline failures are unchanged | AC-1 through AC-10 | coding | task-delete-quant |
| task-verify-full | Run full test suite; verify non-serving callers produce identical BF16 outputs (report, analyze_scenarios, param_search) | AC-6, AC-7, AC-8 | coding | task-verify-focused |
| task-smoke | Run `main.py` and `report/0428/script/generate_report.py` smoke checks; verify end-to-end BF16 output unchanged | AC-1, AC-6 | coding | task-verify-full |
| task-docs | Update `README.md`, `README_zh.md`, `CLAUDE.md`: replace "all sizes use BF16" with quant-aware wording | — | coding | task-verify-full |


## Claude-Codex Deliberation

### Agreements

- Quantization belongs in the core roofline/memory path; the serving-only `quantize_phase_profile()` post-processing creates a bypass risk for non-serving callers and duplicates roofline arithmetic.
- BF16 compatibility is the essential safety invariant for this refactor; all regression risks are caught by comparing new output to current BF16 behavior.
- The memory plan structure (compute BF16 base → apply ratio × overhead) is sound and independently testable.
- Explicit `op_kind` with runtime `ValueError` validation is safer than `infer_op_kind()` name-matching.
- Comprehensive deletion (module, tests, exports, imports) is the right approach; partial deletion leaves a confusing half-migrated API surface.
- Repo-wide caller audit must precede any code changes to avoid missing callers.

### Resolved Disagreements

- **Milestone 3 scope** (Round 1): Claude initially scoped to `ops.py` only. Codex correctly identified `test_roofline.py` as a second direct caller. Resolution: Milestone 3 expanded to all callers in `perf_model/` and `test/`; scope confirmed by Milestone 0 audit.
- **`OpProfile.mem_bytes` semantic** (Round 1): Codex raised concern about losing the raw/adjusted distinction. Resolution: Draft is explicit — `mem_bytes` becomes "final physical HBM bytes after coarse quant policy"; `bytes2()`/`bytes4()` remain raw helpers. Design decision accepted; no extra field needed.
- **`test_quantization_deleted.py` placement** (Round 2): Placing it in Milestone 1 creates a long-lived red test that breaks TDD flow. Resolution: Deletion test moved to Milestone 6 alongside the actual deletion.
- **Serving test style** (Round 2): "No `quantize_phase_profile()` reference in `serving.py`" is an implementation-structure test. Resolution: User confirmed golden-value approach; structural check becomes a verification grep, not a unit test.
- **`quantized_*` function file ownership** (Round 2): Plan v1 incorrectly placed these in `memory.py`. Resolution: `quantized_weight_memory_per_rank()` and `quantized_kv_cache_memory()` live in `quantization.py` and are deleted in Milestone 6, not Milestone 4.
- **W8A8 decode throughput ratio** (Round 2): `~2x` as a test value is too brittle. Resolution: User confirmed golden-value approach; test uses frozen expected values from the current path.

### Convergence Status

- Round 1: Partially converged — 4 REQUIRED_CHANGES, 3 UNRESOLVED items carried into plan v2.
- Round 2: Converged — 0 DISAGREEMENTS, 0 REQUIRED_CHANGES after v2 updates, 0 UNRESOLVED items.
- Final Status: `converged`


## Pending User Decisions

All items have been resolved.

- DEC-1: Quantization ratio hard requirements vs. directional targets
  - Claude Position: The spec's exact ratio values should be enforced in tests.
  - Codex Position: N/A — open question surfaced from Codex v1 analysis.
  - Tradeoff Summary: Hard requirements enable regression detection; directional allows implementation flexibility.
  - Decision Status: Hard requirements — tests must enforce exact values (0.5 for W8A8/KV8, 0.25 for KV4).

- DEC-2: Serving behavioral test approach
  - Claude Position: Capture golden values from the current `quantize_phase_profile()` path; verify after refactor.
  - Codex Position: N/A — open question from Round 2.
  - Tradeoff Summary: Golden-value approach is safest for a refactor; formula-based is more fragile.
  - Decision Status: Golden-value approach confirmed.

- DEC-3: API break acceptability
  - Claude Position: One-shot break is cleaner; no shims.
  - Codex Position: N/A — open question from Codex v1.
  - Decision Status: Resolved by draft — "one-shot refactor, no backward compatibility layer."

- DEC-4: Raw BF16 memory availability
  - Claude Position: BF16 mode returns identical output; raw helpers `bytes2()`/`bytes4()` remain.
  - Decision Status: Resolved by spec — BF16 default output is preserved exactly.

- DEC-5: `decode_model()` quantization
  - Claude Position: Once `decode_step()` is quant-aware, aggregates are naturally quant-aware.
  - Decision Status: Resolved by spec — no aggregate reconstruction required.

- DEC-6: `op_kind` enum vs. string literals
  - Claude Position: String literals with `OP_KINDS` set for validation.
  - Decision Status: Resolved by spec — explicit string literals, 5 valid values.


## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.

--- Original Design Draft Start ---

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
  total remains data-only
  scale_overhead_bytes = cfg.rt.weight_scale_overhead_bytes
  includes "quant_mode"

kv_cache_memory(cfg)
  byte fields * KV_BYTE_RATIOS[cfg.rt.kv_cache_quant_mode]
  total_bytes remains data-only
  scale_overhead_bytes = cfg.rt.kv_scale_overhead_bytes
  includes "kv_cache_quant_mode"
```

The mode metadata keys are required for quantized modes. The default BF16,
zero-overhead path preserves the previous public shape exactly so existing BF16
memory tests and callers remain semantically unchanged.

Callers that need physical HBM usage include the separately reported scale
overhead via the shared memory accessors instead of manually reassembling the
fields.

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

--- Original Design Draft End ---
