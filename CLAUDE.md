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
  roofline.py             # OpProfile, roofline engine, comm helpers (allreduce/alltoall)
  ops.py                  # ~30 per-op cost functions (attention, MoE, index, etc.)
  layers.py               # Layer/phase aggregation (prefill_layer, decode_layer, prefill_model, decode_model)
  memory.py               # KV cache + weight memory analysis
  report.py               # Formatting, printing, CSV export, comm vs compute analysis
main.py                   # CLI entry point (thin wrapper)
output/                   # Auto-generated: timestamped runs with CSV + console output
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
- SP (Sequence Parallel) splits sequence dimension for non-matmul ops
- MoE `load_balance_factor` = 1.0 for first `n_hash_layers`, user-specified otherwise
- Shared expert can overlap with routed experts (configurable)

## Output

Each run produces `output/<timestamp>/` containing:
- `prefill_ops.csv`, `decode_ops.csv` — per-op breakdown
- `layer_summary.csv` — per-layer summary with comp/comm split
- `memory.csv` — KV cache + weight memory
- `summary.csv` — end-to-end metrics including comm vs compute breakdown
- `config.json` — merged config snapshot
- `console_output.txt` — full console log

## Placeholder Locations

KV compression ops in `perf_model/ops.py` return zero profiles:
- `op_kv_compression_prefill()` — fill in compression algorithm costs
- `op_kv_compression_decode()` — fill in amortized per-step cost
- `op_index_kv_compression()` — fill in index key compression costs

## Model Parameters

- 43 layers: 2 full-attn (ratio=1), 21 C4A, 20 C128A
- MQA: 64 Q heads, 1 KV head
- Lightning Index: 64 heads, dim=128, topK=512
- MoE: 256 routed experts, top-6, 1 shared expert
