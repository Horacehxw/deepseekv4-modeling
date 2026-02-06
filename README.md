# DeepSeek V4 Inference Performance Model

A roofline-based performance model for estimating DeepSeek V4 inference latency on Ascend 910C hardware. Computes per-operation, per-layer, and end-to-end latency for both prefill and decode phases.

## Features

- **Roofline model**: Each operation tracks cube (matmul), vector, and memory time separately; bottleneck = argmax
- **Parallelism**: Models TP (Tensor Parallel), DP (Data Parallel), EP (Expert Parallel), and SP (Sequence Parallel)
- **Communication analysis**: AllReduce and AllToAll cost estimation; comm vs compute breakdown per layer and per phase
- **Per-op breakdown**: ~30 individual operation cost functions covering attention projections, Lightning Index, MoE, mHC, and more
- **Memory analysis**: KV cache sizing and weight memory per rank
- **CSV export**: Timestamped output directory with per-op, per-layer, memory, and summary CSVs
- **No dependencies**: Python standard library only

## Quick Start

```bash
python main.py configs/device_910C.json configs/network_910C.json configs/model_deepseekv4.json configs/runtime_deepseekv4.json
```

Output is saved to `output/<timestamp>/` containing CSV exports and a console log.

Sample output includes:
- Configuration summary (hardware, network, model, runtime)
- Prefill phase: per-op breakdown for representative layers, layer summary with comm%, communication vs computation analysis, total latency
- Decode phase: per-op breakdown, comm vs compute analysis, per-step and total latency, tokens/s
- Memory analysis: KV cache per layer, weight memory per rank, total HBM usage
- End-to-end summary with throughput

## Config Files

| File | Description |
|------|-------------|
| `configs/device_910C.json` | Hardware specs: BF16 TFLOPS, vector TFLOPS, HBM capacity/bandwidth, utilization factors |
| `configs/network_910C.json` | Network specs: TP/EP bandwidth, latency, bandwidth utilization |
| `configs/model_deepseekv4.json` | Model architecture: hidden size, layers, heads, dimensions, MoE config, compress ratios |
| `configs/runtime_deepseekv4.json` | Runtime config: seq_len, batch_size, dp, TP/EP/SP, load balance factor, output_len |

## Project Structure

```
configs/                  # JSON configuration files
perf_model/               # Core package
  __init__.py             # Public API re-exports
  config.py               # Config dataclasses + JSON loader
  roofline.py             # OpProfile, roofline engine, communication helpers
  ops.py                  # Per-operation cost functions (~30 ops)
  layers.py               # Layer and phase aggregation
  memory.py               # KV cache + weight memory analysis
  report.py               # Formatting, printing, CSV export, comm vs compute analysis
main.py                   # CLI entry point
output/                   # Auto-generated: timestamped runs with CSV + console output
```

## Architecture

The data flow follows a simple pipeline:

1. **config.py** — Load JSON configs into typed dataclasses
2. **roofline.py** — Core roofline engine: given FLOPs/vec_ops/mem_bytes, compute time breakdown
3. **ops.py** — Each op function computes its FLOPs/memory and calls roofline
4. **layers.py** — Aggregates ops into layers, layers into phases (prefill/decode)
5. **memory.py** — Computes KV cache and weight memory requirements
6. **report.py** — Formats and prints all results; exports CSV files; comm vs compute analysis

## Customization

### Adding new hardware configs
Create a new `configs/device_xxx.json` with fields matching `HardwareConfig`:
- `bf16_tflops`, `vec_tflops`, `hbm_capacity_gb`, `hbm_bandwidth_gbps`
- `flops_utilization`, `hbm_bw_utilization`

### Adding new model configs
Create a new `configs/model_xxx.json`. Key field: `compress_ratios` must be a list of length `num_layers` specifying the compression ratio per layer (1 = full attention).

### Filling in KV compression placeholders
In `perf_model/ops.py`, three functions return zero profiles and are meant to be filled in:
- `op_kv_compression_prefill()` — compression algorithm cost during prefill
- `op_kv_compression_decode()` — amortized per-step cost during decode
- `op_index_kv_compression()` — index key compression cost

## Key Assumptions

- BF16 (2 bytes) for all weights and activations
- Flash attention memory model (no intermediate materialization to HBM)
- MoE load balance factor = 1.0 for hash-routing layers, configurable for others
- Shared expert can fully overlap with routed expert computation (configurable)
- Communication modeled as additive (not overlapped with compute)
- Single-batch decode: SP provides no benefit (T=1)
- DP splits global batch evenly; per-rank batch = batch_size / dp
