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
  roofline.py             # OpProfile, roofline engine, comm helpers (allreduce/alltoall/allgather)
  ops.py                  # ~30 per-op cost functions (attention, MoE, index, etc.)
  layers.py               # Layer/phase aggregation (prefill_layer, decode_layer, prefill_model, decode_model)
  memory.py               # KV cache + weight memory analysis
  report.py               # Formatting, printing, CSV export, comm vs compute analysis
main.py                   # CLI entry point (thin wrapper)
output/                   # Auto-generated: timestamped runs with CSV + console output
param_search/             # Parameter search tool
  search.py               # Grid search across TP/EP/DP/BS/seq for 4 scenarios
  analyze.py              # Analyze results and generate search_report.md
  report.md               # Detailed analysis of search results
  results/                # Auto-generated: timestamped search results with CSVs
```

## Architecture

Pipeline: **config** -> **roofline** -> **ops** -> **layers** -> **memory/report**

- Each op computes `cube_time`, `vec_time`, `mem_time` separately
- Bottleneck = `argmax(cube, vec, mem)`
- Total time = `max(cube, vec, mem) + comm`
- All sizes use BF16 = 2 bytes per element
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

## Parameter Search

Grid search for optimal DeepSeek V4 deployment configurations across 4 independent scenarios.

```bash
python param_search/search.py     # Run search (~30s)
python param_search/analyze.py    # Analyze results and generate report
```

**4 Scenarios:** prefill latency, decode latency, prefill throughput, decode throughput
**GPU formula:** `physical_gpus = TP * DP`, constraint `(TP*DP) % EP == 0`
**Search grid:** TP ∈ {1..64}, EP ∈ {1..256}, DP ∈ {1..8}, BS ∈ {1..512}, seq ∈ {1K..32K}

Key results (Ascend 910C):
- Best prefill latency: TP=8, EP=64, DP=8, BS=8 → 330ms (64 GPUs)
- Best decode latency: TP=4, EP=32, DP=8, BS=8 → 19.3ms/step (32 GPUs)
- Best prefill throughput: TP=8, EP=16, DP=2, BS=256 → 1,656 tps/gpu (16 GPUs)
- Best decode throughput: TP=4, EP=16, DP=4, BS=512 → 181 tps/gpu (16 GPUs)

See `param_search/report.md` for detailed analysis.

## Model Parameters

- 43 layers: 2 full-attn (ratio=1), 21 C4A, 20 C128A
- MQA: 64 Q heads, 1 KV head
- Lightning Index: 64 heads, dim=128, topK=512
- MoE: 256 routed experts, top-6, 1 shared expert
