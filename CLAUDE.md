# DeepSeek V4 Inference Performance Model

Roofline-based performance model for DeepSeek V4 inference on Ascend 910C hardware.
No external dependencies — Python stdlib only.

## How to Run

```bash
python main.py configs/device_910C.json configs/network_910C.json configs/model_deepseekv4.json configs/runtime_deepseekv4.json
```

Output is saved to `output/<timestamp>/` with CSV exports and console log.

## Directory Structure

```
configs/                  # Hardware, network, model, runtime JSON configs
perf_model/               # Core package
  __init__.py             # Re-exports public API
  config.py               # Config dataclasses + JSON loader
  roofline.py             # OpProfile, roofline engine, comm helpers; OP_KINDS, WEIGHT_BYTE_RATIOS, KV_BYTE_RATIOS constants; inline quant policy via op_kind
  ops.py                  # ~30 per-op cost functions (attention, MoE, index, etc.)
  layers.py               # Layer/phase aggregation (prefill_layer, decode_layer, prefill_model, decode_model)
  memory.py               # KV cache + weight memory analysis; applies W8A8/KV8/KV4 quant ratios and overhead
  report.py               # Formatting, printing, CSV export, comm vs compute analysis
main.py                   # CLI entry point (thin wrapper)
output/                   # Auto-generated: timestamped runs with CSV + console output
param_search/             # Parameter search tool
  search.py               # Grid search across TP/EP/DP/BS/seq for 4 scenarios
  analyze.py              # Analyze results and generate search_report.md
  report.md               # Detailed analysis of search results
  results/                # Auto-generated: timestamped search results with CSVs
report/                   # Analysis reports
  analyze_scenarios.py    # Comprehensive analysis: search, P/D ratio, op analysis, V3 comparison
  report_en.md            # Main analysis report (English, 8-section structure)
  report_zh.md            # Main analysis report (Chinese translation)
  ppt_outline_en.md       # PPT outline (English)
  ppt_outline_zh.md       # PPT outline (Chinese)
  data/                   # Auto-generated: 10 JSON data files from analyze_scenarios.py
```

## Architecture

Pipeline: **config** -> **roofline** -> **ops** -> **layers** -> **memory/report**

- Each op computes `cube_time`, `vec_time`, `mem_time` separately
- Bottleneck = `argmax(cube, vec, mem)`
- Total time = `max(cube, vec, mem) + comm`
- All sizes use BF16 = 2 bytes per element (unless quantization enabled via `quant_mode`/`kv_cache_quant_mode`)
- FLOPs convention: matmul `[M, K] x [K, N]` = `M * N * K * 2`

## Key Conventions

- DP (Data Parallel) splits global batch across ranks; per-rank batch = batch_size / dp
- TP (Tensor Parallel) splits Q heads and output projections
- EP (Expert Parallel) splits routed experts across ranks
- SP (Sequence Parallel) splits sequence dimension for non-matmul ops; AllGather at T_sp -> T_full transitions
- MoE `load_balance_factor` = 1.0 for first `n_hash_layers`, user-specified otherwise
- Shared expert can overlap with routed experts (configurable)
- `mhc_kernel_fused` (default: True): Fused mHC ops eliminate separate sinkhorn/pre/post, ~10x less HBM traffic
- `mhc_sp` (default: False): Sequence Parallelism for mHC operations (parallelize across TP)
- `mhc_fused_bf16` (default: False): Use BF16 precision in fused mHC ops (requires mhc_kernel_fused=True)
- `shared_expert_overlapped` (default: True): Shared expert overlaps with MoE dispatch/combine communication

## Attention Memory Model

- **`head_dim` (512) already contains `rope_head_dim` (64)**: RoPE is embedded within the head dimension, not a separate additive projection. Q byte calculations use `Dqc` only (not `Dqc + Dr`).
- **Prefill KV reads**: In prefill, attention ops read the full KV cache (`B * S * kv_d`) rather than just the local window. Since every Q position needs its own local window, a single sequential read of the entire KV is more efficient than per-Q random window reads. This applies to both SWA and compressed attention prefill.
- **Decode KV reads**: In decode (single query), SWA reads only the window (`B * W * kv_d`); compressed attention reads the compressed cache or top-K entries.

## Output

Each run produces `output/<timestamp>/` containing:
- `prefill_ops.csv`, `decode_ops.csv` — per-op breakdown
- `layer_summary.csv` — per-layer summary with comp/comm split
- `memory.csv` — KV cache + weight memory
- `summary.csv` — end-to-end metrics including comm vs compute breakdown
- `config.json` — merged config snapshot
- `console_output.txt` — full console log

## KV Compression Ops

All compression ops are now implemented with exact per-step costs:
- `op_kv_compression_prefill()` — K/V compression with group projections
- `op_kv_compression_decode()` — K/V compression per decode step (cost varies by `S_total % ratio`)
- `op_index_kv_compression_prefill()` — index key compression for Lightning Index (prefill)
- `op_index_kv_compression_decode()` — index key compression per decode step (cost varies by `S_total % ratio`)

## Quantization

`RuntimeConfig` fields control quantization behavior:
- `quant_mode`: `"bf16"` (default) or `"w8a8"` — W8A8 GEMM ops use `effective_w8a8_tflops`; weight memory halved
- `kv_cache_quant_mode`: `"bf16"` (default), `"kv8"`, or `"kv4"` — attention memory and KV cache halved/quartered
- `weight_scale_overhead_bytes` / `kv_scale_overhead_bytes`: extra bytes for quantization scale storage, reported separately from data-only `total` / `total_bytes` values and added explicitly by callers when needed (for example, HBM checks)

The quantization policy is inline in `roofline_time()` via the required `op_kind` parameter. All 5 valid op_kinds: `"gemm"`, `"attention"`, `"vector"`, `"other"`, `"comm"`. Invalid op_kind raises `ValueError`.

## Future Work (TODO)
- Overlap optimization: SWA and compressed attention currently modeled as separate ops.
  Future analysis could model shared Q reads and overlapped execution.

## Parameter Search

Grid search for optimal DeepSeek V4 deployment configurations across 4 independent scenarios.

```bash
python param_search/search.py     # Run search (~30s)
python param_search/analyze.py    # Analyze results and generate report
```

**4 Scenarios:** prefill latency, decode latency, prefill throughput, decode throughput
**GPU formula:** `physical_gpus = TP * DP`, constraint `(TP*DP) % EP == 0`
**Search grid:** TP ∈ {1..64}, EP ∈ {1..256}, DP ∈ {1..8}, BS ∈ {1..512}, seq ∈ {1K..32K}

Key results (Ascend 910C, 8K/4K, no prefix cache):
- Best prefill latency: TP=8, EP=64, DP=8, BS=8 → 715.2ms (64 GPUs)
- Best decode latency: TP=8, EP=64, DP=8, BS=8 → 36.1ms/step (64 GPUs)
- Best prefill throughput: TP=2, EP=16, DP=8, BS=512 → 3,644 tps/gpu (16 GPUs)
- Best decode throughput: TP=2, EP=16, DP=8, BS=512 → 322 tps/gpu (16 GPUs)

4 serving combos analyzed: 8K/4K, 32K/4K, 128K/4K, 256K/4K.
See `report/report_en.md` for detailed analysis and `param_search/report.md` for search details.

## Decode Fast Mode

`decode_model()` uses periodic sampling + trapezoidal interpolation instead of iterating every decode step:

- Per-step cost decomposes as `t(S) = constant + linear(S) + periodic(S)`
- Period `P = LCM(compress_ratios)` = 128 for DeepSeek V4
- Samples only the first P and last P steps, then interpolates: `T_total = N × (T_first + T_last) / (2P)`
- Falls back to exact iteration when `output_len ≤ 2P`
- Mathematically exact when `output_len` is a multiple of P; error < 0.001% otherwise
- 16× speedup for `output_len=4096`

See `_compression_period()` and `decode_model()` in `perf_model/layers.py`.

## Model Parameters

- 43 layers: 2 SWA (ratio=1), 21 C4A, 20 C128A
- MQA: 64 Q heads, 1 KV head
- Lightning Index: 64 heads, dim=128, topK=512
- MoE: 256 routed experts, top-6, 1 shared expert

## Bilingual Reports

Every `*.md` report, doc, and README must have a `*_zh.md` Chinese translation. Translations must have identical data/tables/numbers with translated prose and headings.

Current file pairs:
- `report/report_en.md` ↔ `report/report_zh.md`
- `report/ppt_outline_en.md` ↔ `report/ppt_outline_zh.md`
- `README.md` ↔ `README_zh.md`
- `param_search/report.md` ↔ `param_search/report_zh.md`

Exceptions (no Chinese mirror needed):
- `CLAUDE.md` — internal instructions for Claude Code
- `.claude/commands/*.md` — internal command definitions


## Keyword Explanation
mHC -> manifold Hyper Connection
