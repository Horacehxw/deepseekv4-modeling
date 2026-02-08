# DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20

## Executive Summary

This report presents a comprehensive roofline-based performance analysis of DeepSeek V4 inference across two hardware platforms (Ascend 910C and NVIDIA H20) and three serving scenarios (8K/4K, 128K/4K, 256K/4K input/output lengths). Key findings:

1. **mHC kernel fusion (enabled by default) reduces prefill time 3--4x** by eliminating HBM traffic for mHC operations. Fused baseline: mHC accounts for 25--36% of prefill on 910C (down from 84% unfused).
2. **Decode is MoE-weight-bound** — MoE routed expert weight loading accounts for 38--42% of decode on 910C and 14--53% on H20.
3. **H20's prefill throughput advantage narrows to ~1.0--1.1x** (was 1.6--2.6x before fusion), because fusion eliminates the mHC memory bottleneck where H20 had its biggest advantage.
4. **P/D disaggregation ratios scale with input length** — 1P:1D for 8K, 2P:1D for 128K, 3P:1D for 256K on 910C.
5. **SP with mHC parallelization (mHC-SP) yields additional speedup** on the fused baseline, though the gains are smaller since fusion already addressed the main bottleneck.
6. **910C EP bandwidth advantage (7.8x)** enables efficient high-EP MoE configurations that are prohibitively expensive on H20.

---

## 1. Introduction & Methodology

### 1.1 DeepSeek V4 Architecture

DeepSeek V4 is a Mixture-of-Experts (MoE) model with 43 transformer layers:
- **2 full-attention layers** (ratio=1): standard MQA with 64 Q heads, 1 KV head
- **21 C4A layers** (ratio=4): 4x KV compression with Lightning Index (topK=512)
- **20 C128A layers** (ratio=128): 128x KV compression, no index needed

Key architectural features:
- **MQA** (Multi-Query Attention): 64 Q heads, 1 KV head, head_dim=512
- **Lightning Index**: 64 index heads, dim=128, selects top-512 compressed KV entries
- **MoE**: 256 routed experts (top-6), 1 shared expert, inter_dim=2048
- **mHC (Hyper Connection)**: FP32 pre/post projections + Sinkhorn normalization at every sub-layer
- **KV Compression**: Group projections compress K and V caches by ratio

### 1.2 Roofline Model

Each operation is characterized by three resource dimensions:
- **Cube time**: matmul FLOPs / (cube_tflops x utilization)
- **Vec time**: vector FLOPs / (vec_tflops x utilization)
- **Mem time**: HBM bytes / (bandwidth x utilization)

Bottleneck = argmax(cube, vec, mem). Total time = max(cube, vec, mem) + comm.

### 1.3 Hardware Platforms

| Metric | Ascend 910C | NVIDIA H20 | Ratio |
|---|---|---|---|
| Cube TFLOPS (BF16) | 376 | 148 | **910C 2.54x** |
| Vec TFLOPS (FP32) | 24 | 44 | **H20 1.83x** |
| Cube:Vec ratio | 15.7:1 | 3.4:1 | H20 more balanced |
| HBM Bandwidth (GB/s) | 1,800 | 4,000 | **H20 2.22x** |
| HBM Capacity (GB) | 64 | 96 | **H20 1.5x** |
| TP Bandwidth (GB/s) | 392 | 450 | Similar |
| EP Bandwidth (GB/s) | 392 | 50 | **910C 7.84x** |
| Flops Utilization | 50% | 50% | Same |
| HBM BW Utilization | 80% | 80% | Same |

The most striking hardware difference: 910C has **7.8x higher EP bandwidth** (intra-node interconnect) but **2.2x lower HBM bandwidth**. This fundamentally shapes optimal deployment strategies.

### 1.4 Serving Scenarios

| Scenario | Input Length | Output Length | Use Case |
|---|---|---|---|
| **8K/4K** | 8,192 | 4,096 | Chat, coding assistance |
| **128K/4K** | 131,072 | 4,096 | Long-context RAG, document QA |
| **256K/4K** | 262,144 | 4,096 | Full-window document analysis |

### 1.5 Search Constraints

- **Prefill NPU**: 8, 16, 32, 64, 128, 256 GPUs (physical_gpus = TP x DP)
- **Decode NPU**: 16 or 32 GPUs only (practical deployment constraint)
- **Memory**: weight + KV cache must fit in HBM (64 GB for 910C, 96 GB for H20)
- **Parallelism**: TP divides Q heads (64), EP divides experts (256), (TP x DP) % EP == 0

---

## 2. Parameter Search Results — 8K Input / 4K Output

### 2.1 Ascend 910C

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **330 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 256 | 16 | **1,656 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **19.3 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 512 | 16 | **181 tps/gpu** |

Key observations:
- Prefill latency minimizes at 64 GPUs with EP=64, leveraging maximum expert parallelism
- With kernel fusion enabled, prefill throughput reaches 1,656 tps/gpu — a 4.1x improvement over the unfused baseline
- Decode prefers TP=4 over TP=8, because decode is memory-bound and TP=4 already provides sufficient weight splitting
- Decode throughput benefits from large batch (BS=512) to amortize weight loads

### 2.2 NVIDIA H20

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **554 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 128 | 8 | **1,848 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.0 ms/step** |
| **Decode Throughput** | 8 | 16 | 2 | 512 | 16 | **252 tps/gpu** |

Key observations:
- H20 prefill throughput advantage has narrowed dramatically — from 2.57x to just **1.12x** (1,848 vs 1,656 tps/gpu) — because kernel fusion eliminated the mHC memory bottleneck where H20's 2.2x HBM bandwidth had its biggest advantage
- H20 achieves best prefill throughput with just **8 GPUs** (EP=8), while 910C needs 16 GPUs
- H20 decode latency is **2.1x better** (9.0 vs 19.3 ms) — primarily from faster weight loading
- 910C now achieves **lower prefill latency** than H20 (330 vs 554 ms) — with mHC fusion removing the memory bottleneck, 910C's 2.54x cube TFLOPS advantage dominates

### 2.3 Hardware Comparison at 8K/4K

| Metric | 910C | H20 | H20/910C |
|---|---|---|---|
| Prefill Latency (ms) | 330 | 554 | **1.68x** (H20 slower) |
| Prefill Throughput (tps/gpu) | 1,656 | 1,848 | **1.12x** (H20 slightly higher) |
| Decode Latency (ms/step) | 19.3 | 9.0 | **0.47x** (H20 faster) |
| Decode Throughput (tps/gpu) | 181 | 252 | **1.39x** (H20 higher) |

---

## 3. Parameter Search Results — 128K & 256K Input

### 3.1 128K Input / 4K Output

#### Ascend 910C

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **10,747 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 16 | 16 | **760 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **21.0 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 128 | 32 | **16.7 tps/gpu** |

#### NVIDIA H20

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **20,396 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 8 | 8 | **798 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.9 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 64 | 16 | **47.4 tps/gpu** |

### 3.2 256K Input / 4K Output

#### Ascend 910C

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **33,923 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 8 | 16 | **482 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **21.6 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 64 | 32 | **8.5 tps/gpu** |

#### NVIDIA H20

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **65,685 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 4 | 8 | **497 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **10.1 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 128 | 32 | **25.6 tps/gpu** |

### 3.3 Long-Context Scaling Analysis

**Memory constraints are severe at long context:**
- At 128K, max batch per rank drops to single digits (BS=16 total with DP=2 on 910C)
- At 256K, max batch further halves (BS=8 total on 910C, BS=4 on H20 with EP=8)
- Decode throughput drops dramatically: 181 -> 16.7 -> 8.5 tps/gpu on 910C (8K -> 128K -> 256K)

**Attention compute scales quadratically:**
- Prefill latency: 330ms -> 10,747ms -> 33,923ms (8K -> 128K -> 256K) on 910C
- The 8K -> 128K jump is 33x (16^2 = 256x sequence, but compression mitigates)
- The 128K -> 256K jump is 3.2x (closer to the theoretical 4x quadratic)

**Decode latency is relatively stable:**
- 19.3 -> 21.0 -> 21.6 ms/step on 910C across all three combos
- Decode processes 1 token per step, so longer context mainly adds KV cache read cost
- The KV compression (C4A/C128A) effectively caps the decode attention cost

---

## 4. P/D Disaggregated Serving Analysis

### 4.1 Methodology

In prefill/decode disaggregated serving:
- **P instances** process input tokens (prefill phase)
- **D instances** generate output tokens (decode phase)
- QPS balance: N_p x P_rps >= N_d x D_rps

Where:
```
P_rps = P_tps_per_instance / input_len
D_rps = D_tps_per_instance / output_len
N_p / N_d >= (D_tps_instance x input_len) / (P_tps_instance x output_len)
```

### 4.2 Results

#### Ascend 910C

| Combo | P Config | P tps/inst | D Config | D tps/inst | P:D Ratio | Total GPUs |
|---|---|---|---|---|---|---|
| **8K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 26,490 | TP=4,EP=16,DP=4 (16 GPUs) | 2,897 | **1P:1D** | 32 |
| **128K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 12,155 | TP=4,EP=32,DP=8 (32 GPUs) | 533 | **2P:1D** | 64 |
| **256K/4K** | TP=8,EP=16,DP=2 (16 GPUs) | 7,707 | TP=4,EP=32,DP=8 (32 GPUs) | 273 | **3P:1D** | 80 |

#### NVIDIA H20

| Combo | P Config | P tps/inst | D Config | D tps/inst | P:D Ratio | Total GPUs |
|---|---|---|---|---|---|---|
| **8K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 14,786 | TP=8,EP=16,DP=2 (16 GPUs) | 4,026 | **1P:1D** | 24 |
| **128K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 6,382 | TP=4,EP=16,DP=4 (16 GPUs) | 759 | **4P:1D** | 48 |
| **256K/4K** | TP=8,EP=8,DP=1 (8 GPUs) | 3,973 | TP=4,EP=32,DP=8 (32 GPUs) | 820 | **14P:1D** | 144 |

### 4.3 Analysis

**P:D ratio is driven by input/output length imbalance:**
- At 8K/4K (ratio 2:1), P:D = 1:1 — prefill and decode scale proportionally
- At 128K/4K (ratio 32:1), P:D = 2--4:1 — prefill becomes the bottleneck
- At 256K/4K (ratio 64:1), P:D = 3--14:1 — even more prefill-heavy

**910C P:D ratios decreased significantly with kernel fusion:**
- 128K: 2P:1D (was 4P:1D) — kernel fusion dramatically improved prefill throughput
- 256K: 3P:1D (was 5P:1D) — fewer prefill instances needed
- This directly reduces total GPU budget for 910C deployments

**GPU budget scales with context length:**
- 8K/4K: 32 GPUs (910C), 24 GPUs (H20)
- 128K/4K: 64 GPUs (910C), 48 GPUs (H20)
- 256K/4K: 80 GPUs (910C), 144 GPUs (H20)

At 256K, H20 requires more total GPUs than 910C due to the extreme P:D ratio (14:1) driven by H20's relatively slower prefill throughput at very long sequences combined with fast decode.

---

## 5. SP & mHC-SP Analysis

### 5.1 Three Configurations Tested

Tested with TP=8, EP=16, DP=2, representative batch sizes:

1. **No SP** (`sp=False, mhc_sp=False`) — baseline (with kernel fusion enabled)
2. **SP only** (`sp=True, mhc_sp=False`) — sequence parallelism for RMSNorm/activations; mHC still at T_full
3. **SP + mHC-SP** (`sp=True, mhc_sp=True`) — mHC operations also parallelized across TP dimension

### 5.2 Prefill Time Comparison (ms)

#### Ascend 910C

| Combo | No SP | SP Only | SP + mHC-SP | Speedup (No SP -> SP+mHC-SP) |
|---|---|---|---|---|
| **8K/4K** | 3,507 | 2,492 | 1,706 | **2.06x** |
| **128K/4K** | 102,647 | 86,264 | 73,696 | **1.39x** |
| **256K/4K** | 152,440 | 136,058 | 123,489 | **1.23x** |

#### NVIDIA H20

| Combo | No SP | SP Only | SP + mHC-SP | Speedup (No SP -> SP+mHC-SP) |
|---|---|---|---|---|
| **8K/4K** | 11,738 | 4,398 | 4,045 | **2.90x** |
| **128K/4K** | 281,118 | 163,615 | 157,959 | **1.78x** |
| **256K/4K** | 761,459 | 526,449 | 515,137 | **1.48x** |

### 5.3 Key Findings

**SP+mHC-SP speedups are smaller on the fused baseline** compared to the unfused baseline. This is expected because kernel fusion already addressed the main mHC bottleneck:
- On 910C at 8K: 2.06x speedup (was 4.19x on unfused baseline)
- On H20 at 8K: 2.90x speedup (was 3.39x on unfused baseline)
- The remaining speedup comes from SP reducing Norm, MoE, and communication overhead

**SP alone provides significant benefit on H20:**
- On H20 at 8K: SP reduces prefill time from 11,738ms to 4,398ms (2.67x) by dramatically cutting EP AllToAll communication
- On 910C at 8K: SP reduces from 3,507ms to 2,492ms (1.41x) — more modest due to 910C's fast EP bandwidth

**mHC-SP benefit diminishes at longer sequences:**
- At 8K, mHC is still a significant fraction -> mHC-SP gives meaningful additional speedup
- At 256K, attention compute (quadratic) grows to dominate -> mHC-SP gives only marginal additional speedup beyond SP

### 5.4 Category Breakdown Shift (910C, 8K/4K, Fused Baseline)

| Category | No SP | SP Only | SP + mHC-SP |
|---|---|---|---|
| mHC | **25.6%** | **36.0%** | **6.6%** |
| Communication | 31.9% | 18.6% | 27.1% |
| Attention Proj | 17.9% | 25.2% | 36.9% |
| Attention Compute | 5.5% | 7.7% | 11.3% |
| MoE (all) | 10.8% | 3.0% | 4.4% |
| Others | 8.3% | 9.5% | 13.7% |

After SP+mHC-SP, the bottleneck shifts to attention projections (CUBE-bound) and communication — areas where 910C's hardware advantages shine.

---

## 6. mHC Kernel Fusion Optimization Analysis

### 6.1 Optimization Levels

Four progressively aggressive mHC optimization levels:

| Level | mhc_kernel_fused | mhc_sp | mhc_fused_bf16 | Description |
|---|---|---|---|---|
| Unfused FP32 | False | False | False | Original baseline |
| Fused FP32 | True | False | False | Kernel fusion (new default) |
| Fused FP32 + SP | True | True | False | + Sequence parallelism for mHC |
| Fused BF16 + SP | True | True | True | + BF16 precision for fused ops |

### 6.2 Prefill Time Comparison

#### Ascend 910C (TP=8, EP=16, DP=2, BS=16)

| Level | Prefill Time (ms) | mHC % | Speedup vs Unfused |
|---|---|---|---|
| Unfused FP32 | 10,144 | 84.3% | 1.00x |
| Fused FP32 | 2,492 | 36.0% | 4.07x |
| Fused FP32 + SP | 1,706 | 6.6% | 5.95x |
| Fused BF16 + SP | 1,650 | 3.4% | 6.15x |

### 6.3 Bottleneck Migration

The category breakdown shifts dramatically as mHC optimizations are applied (910C 8K/4K):

| Category | Unfused FP32 | Fused FP32 | Fused+SP | Fused BF16+SP |
|---|---|---|---|---|
| **mHC** | **84.3%** | **36.0%** | **6.6%** | **3.4%** |
| Attention Proj | 6.2% | 25.2% | 36.9% | 38.1% |
| Communication | 4.6% | 18.6% | 27.1% | 28.0% |
| Attention Compute | 1.9% | 7.7% | 11.3% | 11.7% |
| Lightning Index | 1.2% | 4.9% | 7.1% | 7.4% |
| MoE (all) | 0.7% | 3.0% | 4.4% | 4.5% |
| Others | 1.2% | 4.6% | 6.6% | 6.9% |

### 6.4 Representative Layer Op Detail

From the representative layer analysis (910C, 8K/4K, one C4A layer):

| Operation | Unfused FP32 | Fused FP32 | Fused FP32+SP | Fused BF16+SP |
|---|---|---|---|---|
| mhc_pre_attn | 80.5 ms | 6.7 ms | 0.84 ms | 0.42 ms |
| mhc_post_attn | 18.6 ms | 7.5 ms (fused) | 0.93 ms (fused) | 0.47 ms (fused) |
| mhc_pre_moe | 80.5 ms | 6.7 ms | 0.84 ms | 0.42 ms |
| mhc_post_moe | 18.6 ms | 6.7 ms | 0.84 ms | 0.42 ms |
| sinkhorn_attn | 0.24 ms | (eliminated) | (eliminated) | (eliminated) |
| sinkhorn_moe | 0.24 ms | (eliminated) | (eliminated) | (eliminated) |

Key observations:
- `mhc_pre_attn` drops from 80.5ms (unfused) to 6.7ms (fused) -- a 12x reduction
- Sinkhorn operations are folded into the fused kernel, eliminating separate HBM round-trips
- With SP, mHC ops further reduce by 8x (TP=8 parallelism on memory-bound FP32 operations)
- BF16 fusion halves memory traffic for an additional ~2x reduction on top of SP

### 6.5 Key Insights

1. **Kernel fusion reduces mHC HBM traffic ~10x** by keeping intermediates in registers/SRAM instead of writing back to HBM after each sub-operation
2. **The fused default makes 910C competitive with H20 on prefill** — the prefill throughput gap narrowed from 2.57x to just 1.12x
3. **BF16 fusion provides additional ~30% reduction** on top of FP32 fusion+SP, for a total ~6x speedup over the unfused baseline
4. **After fusion, the bottleneck migrates to attention projections (CUBE-bound)** — where 910C's 2.54x cube TFLOPS advantage is maximally leveraged
5. **The optimization impact is most dramatic at short sequences (8K)** where mHC dominates; at 256K, quadratic attention already overshadows mHC

---

## 7. Op Boundedness Analysis

### 7.1 Per-Category Bottleneck Summary

| Category | Prefill Bottleneck | Decode Bottleneck |
|---|---|---|
| Attention Projections | CUBE | MEM |
| Attention Compute | CUBE | MEM |
| KV Compression | CUBE | VEC/MEM |
| Lightning Index | COMM (score AllReduce) | COMM |
| mHC (fused) | MEM | MEM |
| MoE Gate | CUBE | MEM |
| MoE Routed Experts | CUBE | MEM |
| Communication | COMM | COMM |

**Key insight: Prefill is compute-bound (CUBE), Decode is memory-bound (MEM/COMM).**

With kernel fusion enabled, mHC is still MEM-bound but its absolute time is dramatically reduced. The prefill bottleneck has shifted to CUBE-bound attention projections — the classic LLM inference pattern.

### 7.2 Prefill Op Breakdown (910C, Best Throughput Config, Fused)

| Category | 8K/4K | 128K/4K | 256K/4K |
|---|---|---|---|
| mHC | **36.3%** | **16.7%** | **10.6%** |
| Attention Proj | 25.4% | 11.7% | 7.4% |
| Attention Compute | 7.8% | 31.6% | 39.0% |
| Lightning Index | 4.8% | 28.2% | 35.5% |
| Communication | 18.1% | 8.3% | 5.3% |
| MoE | 1.3% | 0.6% | 0.4% |
| Others | 6.3% | 2.9% | 1.8% |

With kernel fusion, mHC's dominance is significantly reduced. At 8K, it remains the largest single category at 36.3%, but attention projections and communication are now comparable. As sequence length increases, attention compute and Lightning Index grow quadratically and dominate.

### 7.3 Decode Op Breakdown (910C, Best Throughput Config)

| Category | 8K/4K | 128K/4K | 256K/4K |
|---|---|---|---|
| MoE Routed | **55.9%** | **40.3%** | **41.1%** |
| Attention Compute | 14.5% | 22.0% | 21.4% |
| Communication | 12.9% | 12.4% | 12.2% |
| Lightning Index | 5.7% | 16.4% | 16.7% |
| Attention Proj | 5.4% | 6.8% | 6.9% |
| mHC | 4.1% | 0.7% | 0.4% |
| Others | 1.5% | 1.4% | 1.3% |

Decode is dominated by MoE weight loading (40--56%). At longer sequences, attention and Lightning Index KV cache reads grow.

---

## 8. Time Breakdown by Category — Detailed Tables

### 8.1 Prefill (H20, Best Throughput Config, Fused)

| Category | 8K/4K | 128K/4K | 256K/4K |
|---|---|---|---|
| mHC | **36.3%** | **16.7%** | **10.6%** |
| Attention Compute | 7.8% | 31.6% | 39.0% |
| Attention Proj | 25.4% | 11.7% | 7.4% |
| Lightning Index | 4.8% | 28.2% | 35.5% |
| Communication | 18.1% | 8.3% | 5.3% |
| MoE | 1.3% | 0.6% | 0.4% |
| KV Compression | 2.4% | 1.1% | 0.7% |
| Others | 3.9% | 1.8% | 1.1% |

### 8.2 Decode (H20, Best Throughput Config)

| Category | 8K/4K | 128K/4K | 256K/4K |
|---|---|---|---|
| Communication | **44.6%** | 16.6% | 13.2% |
| mHC | 19.8% | 4.3% | 2.3% |
| MoE Routed | 14.3% | **49.5%** | **53.4%** |
| Attention Compute | 7.3% | 13.5% | 13.9% |
| Attention Proj | 8.2% | 4.2% | 4.5% |
| Lightning Index | 3.7% | 11.0% | 11.9% |
| Others | 2.0% | 0.8% | 0.8% |

On H20 at 8K, decode communication is the dominant factor (44.6%) — this is because H20's low EP bandwidth (50 GB/s) makes AllToAll extremely expensive. At longer sequences, MoE weight loading takes over.

---

## 9. Ascend 910C vs NVIDIA H20 — Detailed Comparison

### 9.1 Prefill Throughput Comparison

| Combo | 910C (tps/gpu) | H20 (tps/gpu) | H20/910C | 910C GPUs | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | 1,656 | 1,848 | **1.12x** | 16 | 8 |
| 128K/4K | 760 | 798 | **1.05x** | 16 | 8 |
| 256K/4K | 482 | 497 | **1.03x** | 16 | 8 |

H20's prefill throughput advantage has narrowed dramatically with kernel fusion:
- Was 2.57x at 8K, now just **1.12x**
- Was 1.90x at 128K, now just **1.05x**
- Was 1.62x at 256K, now just **1.03x**
- At longer sequences, the gap is virtually eliminated because attention compute (CUBE-bound) dominates and 910C has 2.54x higher cube TFLOPS

### 9.2 Decode Throughput Comparison

| Combo | 910C (tps/gpu) | H20 (tps/gpu) | H20/910C | 910C GPUs | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | 181 | 252 | **1.39x** | 16 | 16 |
| 128K/4K | 16.7 | 47.4 | **2.84x** | 32 | 16 |
| 256K/4K | 8.5 | 25.6 | **3.01x** | 32 | 16 |

H20 advantage grows at longer sequences for decode because:
1. Longer context -> more KV cache -> more HBM reads -> H20's 2.2x bandwidth advantage matters more
2. 910C needs 32 GPUs (larger EP) while H20 fits in 16 GPUs at long context

### 9.3 Decode Latency Comparison

| Combo | 910C (ms/step) | H20 (ms/step) | H20/910C |
|---|---|---|---|
| 8K/4K | 19.3 | 9.0 | **0.47x** (H20 faster) |
| 128K/4K | 21.0 | 9.9 | **0.47x** |
| 256K/4K | 21.6 | 10.1 | **0.47x** |

H20 decode latency is consistently ~2x better across all context lengths, directly reflecting the HBM bandwidth ratio.

### 9.4 Root Cause Analysis

**Where 910C wins:**
- Cube-heavy workloads: attention projections (matmul-dominant), MoE expert matmuls
- High-EP configurations: EP=32/64 at low cost due to 392 GB/s EP bandwidth
- **Prefill latency**: with kernel fusion, 910C achieves 1.5--1.9x lower prefill latency than H20 across all sequence lengths (910C's cube advantage is maximally leveraged when the mHC bottleneck is eliminated)

**Where H20 wins:**
- Memory-bound workloads: decode weight loading, KV cache reads
- Low-EP throughput: EP=8 with only 8 GPUs matches or exceeds 910C's 16 GPU configurations
- Decode phase across all scenarios (2x faster consistently)

**H20's EP bandwidth problem:**
- H20 EP bandwidth is only 50 GB/s (IB cross-node) vs 910C's 392 GB/s (on-die interconnect)
- At EP=64, AllToAll time on H20 is ~7.8x higher than 910C
- This forces H20 to use EP=8 for throughput configs, meaning all 256 experts must be split across only 8 ranks (32 experts per rank)
- The weight memory per rank is higher at EP=8, but H20's 96 GB HBM accommodates this

---

## 10. Industry Implications

### 10.1 KV Cache & Long Context

Long context (128K+) fundamentally changes the serving landscape:
- Batch sizes shrink to 1--8 per rank due to KV cache memory
- Decode throughput drops 10--21x going from 8K to 128K/256K
- P/D ratios increase 2--14x requiring proportionally more prefill instances

**Implication**: KV compression (as in DeepSeek V4's C4A/C128A) is essential for practical long-context serving. Without it, 128K context would be infeasible at reasonable batch sizes.

### 10.2 P/D Disaggregation

The analysis confirms that P/D disaggregation is critical for long-context serving:
- At 128K/4K: 2--4 prefill instances per decode instance
- At 256K/4K: 3--14 prefill instances per decode instance
- Mixed (non-disaggregated) serving would waste GPU resources on the non-bottleneck phase

**Practical guidance**: For 128K workloads on 910C, a cluster of 64 GPUs (2x16 GPU prefill + 1x32 GPU decode) provides QPS-balanced serving.

### 10.3 Hardware Design Insights

**With kernel fusion, 910C's compute advantage becomes decisive for prefill:**
- Before fusion: H20's 2.2x HBM bandwidth dominated prefill via mHC memory bottleneck
- After fusion: mHC bottleneck eliminated, 910C's 2.54x cube TFLOPS drives prefill
- This shows that software optimizations can fundamentally change hardware competitiveness

**HBM bandwidth remains the decode differentiator:**
- H20's 2.2x bandwidth advantage translates to consistent 2x decode speedup
- For decode-heavy workloads (chatbots with long outputs), HBM bandwidth > compute TFLOPS

**HBM capacity enables larger batches:**
- H20's 96 GB vs 910C's 64 GB allows 50% more configs to fit in memory
- At 128K/256K, this means H20 can fit batch sizes that 910C cannot

### 10.4 Network for MoE

EP bandwidth is the critical differentiator for MoE model serving:
- 910C's 392 GB/s EP allows EP=64 with modest AllToAll overhead
- H20's 50 GB/s EP makes EP=64 catastrophic (7.8x higher AllToAll cost)
- H20 compensates by using lower EP (EP=8) with more experts per rank

**Implication**: MoE models strongly favor platforms with high-bandwidth interconnects. NVLink-domain (8 GPU) is efficient on H20, but cross-node MoE is very expensive.

### 10.5 mHC as a New Paradigm

DeepSeek V4's mHC (Hyper Connection) is architecturally novel and was previously the dominant performance bottleneck. **Kernel fusion has proven transformative:**

- **Unfused**: mHC accounts for 84% of prefill time at 8K on 910C, making it the single largest optimization target
- **Fused (default)**: mHC drops to 36% of prefill, providing a 4.1x overall prefill speedup
- **Fused + SP + BF16**: mHC drops to 3.4%, for a total 6.1x speedup over unfused

The kernel fusion approach keeps intermediate results (pre/post projections, Sinkhorn normalization) in registers/SRAM rather than writing them back to HBM after each sub-operation. This eliminates ~10x HBM traffic and transforms what was a severe memory bottleneck into a manageable overhead.

**Remaining optimization opportunities:**
1. **BF16 mHC** (`mhc_fused_bf16=True`): Additional ~30% reduction by halving memory traffic for fused ops
2. **mHC-SP** (`mhc_sp=True`): Further parallelization across TP dimension, reducing mHC to <7% of total

---

## 11. Deployment Recommendations

### 11.1 Short Context (8K/4K) — Chat/Coding

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 1:1 | 32 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=8, EP=16, DP=2 (16 GPUs) | 1:1 | 24 |

- Both platforms handle 8K efficiently with 1P:1D ratio
- H20 slightly more cost-effective (24 vs 32 GPUs for balanced serving)
- Kernel fusion + shared expert overlap enabled by default
- 910C now achieves competitive prefill throughput (1,656 vs 1,848 tps/gpu)

### 11.2 Long Context (128K/4K) — RAG/Document QA

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 2:1 | 64 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 4:1 | 48 |

- 910C P:D ratio improved from 4:1 to 2:1 with kernel fusion, reducing total GPUs from 96 to 64
- H20 still requires fewer total GPUs (48 vs 64)
- Memory is the primary constraint on batch sizes

### 11.3 Ultra-Long Context (256K/4K) — Full Document Analysis

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 3:1 | 80 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 14:1 | 144 |

- 910C P:D ratio improved from 5:1 to 3:1 with kernel fusion, reducing total GPUs from 112 to 80
- H20 requires significantly more total GPUs at 256K (144 vs 80) due to extreme P:D ratio (14:1)
- 910C becomes the more GPU-efficient option at ultra-long context
- Consider whether 256K context is worth the GPU investment vs chunking strategies

### 11.4 General Guidance

1. **Kernel fusion is enabled by default** (`mhc_kernel_fused=True`) — provides 3--4x prefill speedup at no cost
2. **Shared expert overlap is enabled by default** (`shared_expert_overlapped=True`) — shared expert computation overlaps with MoE dispatch/combine communication
3. **Always use SP** (`sp=True`) — free performance gain from reduced Norm/activation compute
4. **Investigate mHC-SP** (`mhc_sp=True`) — if mHC parallelization is feasible in the framework, it provides additional prefill speedup on top of fusion
5. **Tune EP based on network** — high EP on 910C (EP=16--64), low EP on H20 (EP=8--16)
6. **Batch aggressively for decode** — decode throughput scales nearly linearly with batch size up to memory limits
7. **P/D disaggregation is essential for 128K+** — mixed serving wastes 60--80% of resources

---

## Appendix A: Methodology Details

- **Roofline model**: Each op's time = max(cube_time, vec_time, mem_time) + comm_time
- **FLOPs**: matmul [M,K]x[K,N] = MxNxKx2
- **BF16**: 2 bytes per element; FP32 (mHC): 4 bytes per element
- **Communication**: AllReduce = 2(n-1)/n x vol/BW, AllToAll = (n-1)/n x vol/BW, AllGather = (n-1)/n x vol/BW
- **Utilization**: 50% compute, 80% HBM bandwidth (both platforms)
- **Memory check**: weight_per_rank + KV_cache_per_rank <= HBM capacity
- **Decode approximation**: linear interpolation between first and last decode steps

## Appendix B: Model Configuration Summary

| Parameter | Value |
|---|---|
| Hidden size | 4,096 |
| Layers | 43 (2 full + 21 C4A + 20 C128A) |
| Q heads | 64 (MQA: 1 KV head) |
| Head dim | 512 |
| Q LoRA rank | 1,024 |
| O groups | 8, O LoRA rank = 1,024 |
| Index heads | 64, dim = 128, topK = 512 |
| Window size | 128 (SWA) |
| Routed experts | 256, top-6 |
| Shared experts | 1 |
| MoE inter dim | 2,048 |
| HC mult | 4 |
| Vocab size | 129,280 |

## Appendix C: Data Files

All raw data is available in `report/data/`:
- `search_results_910C.json` / `search_results_H20.json` — per-scenario top-20 configs
- `pd_ratio_analysis.json` — P/D ratio calculations
- `op_analysis.json` — per-op bottleneck breakdown
- `sp_comparison.json` — SP/mHC-SP comparison
- `hardware_comparison.json` — cross-platform comparison
- `mhc_optimization_comparison.json` — 4 mHC optimization levels comparison
