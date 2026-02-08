# DeepSeek V4 Inference Performance Model

A roofline-based performance model for estimating DeepSeek V4 inference latency on Ascend 910C hardware. Computes per-operation, per-layer, and end-to-end latency for both prefill and decode phases.

## Features

- **Roofline model**: Each operation tracks cube (matmul), vector, and memory time separately; bottleneck = argmax
- **Parallelism**: Models TP (Tensor Parallel), DP (Data Parallel), EP (Expert Parallel), and SP (Sequence Parallel)
- **Communication analysis**: AllReduce, AllToAll, and AllGather cost estimation; comm vs compute breakdown per layer and per phase
- **Per-op breakdown**: ~30 individual operation cost functions covering attention projections, Lightning Index, MoE, mHC, and more
- **mHC kernel fusion** (enabled by default): Fused mHC ops reduce HBM traffic ~10x by keeping intermediates in registers/SRAM
- **Shared expert overlap** (enabled by default): Shared expert computation overlaps with MoE dispatch/combine communication
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
param_search/             # Parameter search tool
  search.py               # Grid search across TP/EP/DP/BS/seq for 4 scenarios
  analyze.py              # Analyze results and generate search_report.md
  report.md               # Detailed analysis of search results
  results/                # Auto-generated: timestamped search results with CSVs
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
- `cube_tflops`, `vec_tflops`, `hbm_capacity_gb`, `hbm_bandwidth_gbps`
- `flops_utilization`, `hbm_bw_utilization`

### Adding new model configs
Create a new `configs/model_xxx.json`. Key field: `compress_ratios` must be a list of length `num_layers` specifying the compression ratio per layer (1 = full attention).

## Parameter Search

Find optimal deployment configurations by grid-searching across parallelism strategies, batch sizes, and sequence lengths.

```bash
python param_search/search.py     # Run search (~30s)
python param_search/analyze.py    # Analyze results and generate report
```

The search evaluates 4 independent scenarios:

| Scenario | Optimizes | Metric |
|:---|:---|:---|
| Prefill Latency | Time to first token | `prefill_time_ms` (minimize) |
| Decode Latency | Per-step generation speed | `decode_first_step_ms` (minimize) |
| Prefill Throughput | Prefill tokens per GPU per second | `B*S / prefill_s / GPUs` (maximize) |
| Decode Throughput | Output tokens per GPU per second | `B*output_len / decode_s / GPUs` (maximize) |

**Search grid:** TP ∈ {1,2,4,8,16,32,64}, EP ∈ {1,2,4,...,256}, DP ∈ {1,2,4,8}, BS ∈ {1,...,512}, seq ∈ {1K,...,32K}
**GPU formula:** `physical_gpus = TP * DP`, constraint `(TP*DP) % EP == 0`
**Constraints:** GPU count ∈ [8, 64], HBM ≤ 64 GB

**Key results (Ascend 910C):**

| Scenario | Best Config | Key Metric | GPUs |
|:---|:---|---:|---:|
| Prefill Latency | TP=8, EP=64, DP=8, BS=8 | 330 ms | 64 |
| Decode Latency | TP=4, EP=32, DP=8, BS=8 | 19.3 ms/step | 32 |
| Prefill Throughput | TP=8, EP=16, DP=2, BS=256 | 1,656 tok/s/GPU | 16 |
| Decode Throughput | TP=4, EP=16, DP=4, BS=512 | 181 tok/s/GPU | 16 |

See [`param_search/report.md`](param_search/report.md) for detailed analysis including per-sequence-length breakdowns, SP impact, batch scaling, and deployment recommendations.

## Key Assumptions

- BF16 (2 bytes) for all weights and activations
- Flash attention memory model (no intermediate materialization to HBM)
- MoE load balance factor = 1.0 for hash-routing layers, configurable for others
- Shared expert can fully overlap with routed expert computation (configurable)
- Communication modeled as additive (not overlapped with compute)
- SP (Sequence Parallel) inserts AllGather at every T_sp -> T_full transition in prefill (before attention, before/after MoE, before LM head)
- Single-batch decode: SP provides no benefit (T=1)
- DP splits global batch evenly; per-rank batch = batch_size / dp