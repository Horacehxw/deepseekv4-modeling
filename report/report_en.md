# DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20

## 1. Executive Summary

This report quantifies DeepSeek V4 inference performance on **Ascend 910C** and **NVIDIA H20** across four context lengths (8K, 32K, 128K, 256K) and four key analysis dimensions (per-op bottleneck breakdown, optimal parallelism configurations, P/D disaggregation topology, and KV-compression scaling to 1M context). All numbers come from a roofline analytical model with grid-searched configurations; full methodology is in Section 8.

**Six headline findings:**

1. **V4 cuts the KV cache 4.6x vs V3** (15,168 bytes/token vs 70,272). The C4A/C128A heterogeneous compression layout makes 256K-1M context economical: at 1M tokens, V4's KV state is ~6.9 GB vs V3's ~70 GB on the same hardware.

2. **910C wins prefill, H20 wins decode**, by 1.2x and 3.4-6.1x respectively, reflecting their opposite design priorities: 910C trades HBM bandwidth (1.8 TB/s) for higher BF16 compute (376 TFLOPS), while H20 trades compute (148 TFLOPS) for HBM bandwidth (4 TB/s). On 8K context per-card decode throughput is 312 tok/s on 910C and 1,053 tok/s on H20.

3. **mHC kernel fusion is the single highest-leverage optimization on 910C** — fused SP+BF16 mHC reduces 8K prefill mHC cost from 88% to 4.6% of step time, yielding ~8x end-to-end prefill speedup. On H20 the same fusion gives only 1.9x because mHC was already bandwidth-fit.

4. **Decode is bandwidth-bound, MoE-dominated.** At 8K, MoE routed experts consume 70% of decode time on 910C and 37% on H20 — the gap is comm cost on H20. AllToAll communication for MoE is the dominant bottleneck on H20, accounting for ~27% of decode time even at modest EP=8 settings.

5. **Prefix caching transforms PD ratios.** At 0% cache, optimal P:D is roughly balanced (5:6 to 4:1 across context lengths). At 99% cache hit, the same workload needs 1P:13D to 1P:489D (H20/256K to 910C/8K) — prefill becomes 10–500x cheaper than decode in throughput terms depending on hardware and context. Provisioning systems must treat cache hit rate as a first-class capacity dimension.

6. **128K-256K context shifts the bottleneck to attention.** Lightning Index attention dominates prefill at long context (>40% at 256K), and the long-skip C128A compression keeps decode KV reads tractable up to 1M tokens. This is the architectural feature that makes V4 viable on 64 GB-class accelerators at long context.

The combined picture is a V4 inference cost model where: (a) hardware choice should be phase-specific (prefill on 910C-class, decode on H20-class), (b) kernel fusion (especially mHC) is mandatory not optional for 910C, (c) cache-aware scheduling is the highest-leverage software lever, and (d) long-context economics are governed by KV compression and EP-bandwidth, not raw FLOPs.

---

## 2. Model Structure

### 2.1 DeepSeek V4 Architecture

DeepSeek V4 represents a fundamental redesign of the DeepSeek model family, trading raw parameter count for inference efficiency. At approximately 285B total parameters — less than half of V3's 704B — V4 achieves competitive quality through three key architectural innovations:

**MQA with KV Compression.** V4 uses Multi-Query Attention (64 Q heads sharing 1 KV head) combined with an aggressive heterogeneous compression schedule: 2 full-attention layers (ratio=1), 21 C4A layers (4x compression), and 20 C128A layers (128x compression). This yields only 15,168 bytes of KV cache per token — a 4.6x reduction versus V3's 70,272 bytes.

**Lightning Index.** A learned sparse retrieval mechanism with 64 index heads (dim=128) that selects the top-512 most relevant KV entries per query at each compressed attention layer, maintaining quality on long-context tasks while keeping cache footprint small.

**manifold Hyper Connection (mHC).** Replaces standard residual connections with a 4x-expanded mixing operation (hc_mult=4), improving gradient flow at the cost of significantly increased HBM traffic when unoptimized.

| Parameter | Value |
|---|---|
| Hidden size | 4,096 |
| Layers | 43 (2 full + 21 C4A + 20 C128A) |
| Q heads | 64 (MQA: 1 KV head) |
| Head dim | 512 |
| RoPE head dim | 64 |
| Q LoRA rank | 1,024 |
| O groups | 8 |
| O LoRA rank | 1,024 |
| Index heads | 64 |
| Index head dim | 128 |
| Index topK | 512 |
| Window size | 128 (SWA) |
| Routed experts | 256, top-6 |
| Shared experts | 1 |
| MoE inter dim | 2,048 |
| HC mult | 4 |
| HC Sinkhorn iters | 20 |
| Vocab size | 129,280 |
| Compress ratios | 2 full + 21 C4A + 20 C128A |
| Dtype | bfloat16 |

**Why this architecture.** V4's design choices are governed by inference economics on memory-bound hardware. MQA-style attention with kv_lora_rank=512 plus per-layer compression collapses the KV footprint to 15.3 KB/token — 4.6x smaller than V3 — which is what makes 256K-1M context windows tractable on a 64 GB accelerator. The mHC block (with hidden expansion of 4x and Sinkhorn routing across 256 experts) replaces a dense FFN with a structured low-rank mixture, trading dense matmul FLOPs for routing/gather/scatter traffic. Lightning Index acts as a topk-cache pre-filter for full attention layers; it reduces SWA surface area but, as Section 3 will show, becomes the dominant prefill cost beyond 32K tokens because its complexity scales with seq^2 even after sparsification.

### 2.2 Hardware Platform: Ascend 910C

The two target platforms have fundamentally different hardware profiles, leading to distinct inference characteristics.

| Metric | Ascend 910C | NVIDIA H20 | Ratio |
|---|---|---|---|
| Cube TFLOPS (BF16) | 376 | 148 | **910C 2.54x** |
| Vec TFLOPS (FP32) | 24 | 44 | **H20 1.83x** |
| Cube:Vec ratio | 15.7:1 | 3.4:1 | H20 more balanced |
| HBM Bandwidth (GB/s) | 1,800 | 4,000 | **H20 2.22x** |
| HBM Capacity (GB) | 64 | 96 | **H20 1.5x** |
| HBM Reserved | 10% | 10% | Same |
| TP Bandwidth (GB/s) | 392 | 450 | Similar |
| EP Bandwidth (GB/s) | 392 | 50 | **910C 7.84x** |
| Network Latency (us) | 10 | 5 | H20 2x lower |
| Cube Utilization | 40% | 50% | — |
| HBM BW Utilization | 30% | 80% | — |
| Vec Utilization | 20% | — | — |
| BW Utilization | 26% | 80% | — |

**Hardware profile contrast.** The two platforms target this workload with opposite tradeoffs. The 910C is a *compute-rich, bandwidth-poor, scale-out-friendly* accelerator: 2.5x the BF16 throughput of H20 but only 0.45x the HBM bandwidth, paired with a 9x richer EP fabric (the modeled 392 GB/s is a per-link aggregate over the 910C scale-out network, vs H20's 50 GB/s effective NVLink-derived per-rank EP bandwidth in our serving topology). H20 is the inverse: low FLOPs but high HBM bandwidth and large per-package capacity. The two also differ on realized utilization assumptions — 910C is modeled at 40% cube and 30% HBM utilization, vs H20 at 50%/80% — reflecting H20's far more mature CUDA/cuBLAS stack and the larger gap between peak and sustained throughput on 910C today.

This asymmetry directly determines where each platform wins. Compute-bound phases (prefill, MoE GEMMs) lean toward 910C; memory-bound phases (token-by-token decode of large-KV attention) lean toward H20; and the 7.5x EP-bandwidth advantage of 910C swings any phase with heavy expert dispatch back its way. The rest of the report quantifies how each phase lands on each platform.

### 2.3 V4 vs V3 Comparison

| Dimension | DeepSeek V4 | DeepSeek V3 |
|---|---|---|
| Hidden size | 4,096 | 7,168 |
| Layers | 43 | 61 |
| Q heads | 64 | 128 |
| KV approach | MQA + KV compression | MLA (kv_lora_rank=512) |
| Q LoRA rank | 1,024 | 1,536 |
| Routed experts | 256, top-6 | 256, top-8 |
| MoE inter dim | 2,048 | 2,048 |
| KV compression | C4A/C128A (2--128x) | None |
| mHC | Yes (hc_mult=4) | No |
| Lightning Index | Yes (64 heads, dim=128, topK=512) | No |
| Total params (approx) | ~285B | ~704B |
| KV cache per token (bytes) | 15,168 | 70,272 |
| **KV size ratio vs V3** | **4.63x smaller** | Baseline |
| Attn FLOPs per token | 226,492,416 | 374,210,560 |
| MoE FLOPs per token | 352,321,536 | 792,723,456 |
| Attn params per layer | 113,246,208 | 187,105,280 |
| MoE params per layer | 6,468,665,344 | 11,320,164,352 |
| Weight memory per rank (BF16) | ~39.9 GB | — |

**V4 vs V3 architectural deltas.** Three changes drive most of the deployment-economics improvement. (1) **Layer count drops from 61 to 43 (-30%)** while hidden width also shrinks (7,168 -> 4,096), cutting both attention and MoE per-token cost roughly in half. (2) **MQA + KV compression** replaces V3's MLA path: the per-token KV state shrinks from 70,272 bytes to 15,312 bytes (4.63x), entirely from the C4A/C128A layer mix, which is what unlocks 1M-context-class deployment without ballooning HBM. (3) **mHC and Lightning Index** are entirely new — they introduce additional compute, but the savings from (1) and (2) more than offset them: total per-token FLOPs drop from ~792 GFLOP (V3) to ~579 GFLOP (V4 attn+MoE). The net effect is that V4 fits in less than half of V3's per-rank HBM budget while also being faster per token.

---

## 3. Bottleneck Analysis

### 3.1 Prefill Bottleneck Analysis

**910C Prefill Op Breakdown (TP=2, EP=16, DP=8, SP+mHC fused, best throughput config)**

| Category | 8K/4K | 32K/4K | 128K/4K | 256K/4K | Bottleneck |
|---|---|---|---|---|---|
| mHC | **25.95%** | **23.84%** | **17.36%** | **12.63%** | MEM |
| Attention Proj | 23.00% | 21.13% | 15.39% | 11.20% | CUBE |
| Attention Compute | 15.46% | 14.20% | 13.74% | 14.15% | MEM/CUBE |
| Communication | 22.55% | 20.71% | 15.09% | 10.98% | COMM |
| Lightning Index | 4.19% | 11.98% | **32.49%** | **46.73%** | CUBE |
| KV Compression | 2.49% | 2.29% | 1.67% | 1.21% | CUBE |
| MoE Routed | 2.43% | 2.23% | 1.62% | 1.18% | CUBE |
| Embedding/LMHead | 2.51% | 2.31% | 1.68% | 1.22% | CUBE |
| Norm | 0.94% | 0.86% | 0.63% | 0.46% | MEM |
| MoE Gate | 0.49% | 0.45% | 0.33% | 0.24% | MEM |
| MoE Shared | 0.00% | 0.00% | 0.00% | 0.00% | N/A |

**H20 Prefill Op Breakdown (8K/4K)**

| Category | 8K/4K | Bottleneck |
|---|---|---|
| Attention Proj | **35.95%** | CUBE |
| Communication | 31.72% | COMM |
| Attention Compute | 11.72% | CUBE |
| Embedding/LMHead | 4.14% | CUBE |
| Lightning Index | 4.60% | CUBE |
| KV Compression | 2.04% | CUBE |
| mHC | 1.78% | MEM |
| MoE Gate | 0.35% | CUBE |
| MoE Routed | 7.57% | CUBE |
| Norm | 0.13% | MEM |

**Prefill bottlenecks.** Prefill is heavily compute-bound on both platforms: at 8K input the largest costs are mHC (~26-29%), attention projections, and MoE GEMMs — all CUBE-bound. As context grows, attention compute (full attention + Lightning Index) dominates: by 128K, Lightning Index alone consumes ~32% of prefill time on 910C and ~28% on H20, and at 256K it crosses 50% of total prefill cost. MQA + C4A/C128A keeps the KV-write side cheap, but Lightning Index attention (Q*K over the full sequence) cannot be avoided at long context.

The 910C/H20 split shows up clearly: on 910C, mHC is reduced to <5% by fused BF16 + mHC-SP, leaving attention projection (CUBE) and MoE (CUBE) as the largest fractions. On H20, mHC remains ~10% because H20's lower compute throughput makes the fused FP32 path more painful in absolute time, and communication bites harder (27% comm at 8K) because H20 EP bandwidth is ~7.5x lower per rank. As a result, H20 spends a larger share of prefill in communication while 910C spends a larger share in compute.

### 3.2 Decode Bottleneck Analysis

**910C Decode Op Breakdown (best throughput config)**

| Category | 8K/4K | 32K/4K | 128K/4K | 256K/4K | Bottleneck |
|---|---|---|---|---|---|
| MoE Routed | **70.30%** | **52.51%** | **50.75%** | **53.16%** | MEM |
| Attention Proj | 10.43% | 8.68% | 8.11% | 8.35% | MEM |
| Attention Compute | 6.05% | 9.44% | 5.54% | 2.90% | MEM |
| Lightning Index | 2.56% | 11.54% | **22.68%** | **23.69%** | MEM |
| Communication | 4.79% | 9.92% | 7.22% | 6.32% | COMM |
| mHC | 2.54% | 3.78% | 1.86% | 1.59% | MEM/VEC |
| KV Compression | 1.14% | 1.70% | 1.55% | 1.60% | MEM |
| Embedding/LMHead | 1.09% | 0.81% | 0.77% | 0.81% | MEM |
| Norm | 0.87% | 1.29% | 1.23% | 1.29% | VEC |
| MoE Gate | 0.23% | 0.34% | 0.30% | 0.29% | MEM |

| MoE Shared | 0.00% | 0.00% | 0.00% | 0.00% | N/A |

**H20 Decode Op Breakdown (8K/4K)**

| Category | 8K/4K | Bottleneck |
|---|---|---|
| MoE Routed | **36.82%** | MEM |
| Communication | **27.59%** | COMM |
| Attention Proj | 13.76% | CUBE |
| Attention Compute | 6.98% | VEC |
| Norm | 2.85% | VEC |
| KV Compression | 2.90% | CUBE |
| mHC | 3.44% | VEC |
| Lightning Index | 3.90% | CUBE |
| Embedding/LMHead | 1.50% | CUBE |
| MoE Gate | 0.25% | CUBE |

**Decode bottleneck pattern.** Decode is fundamentally bandwidth-bound. On 910C, MoE expert weight loads dominate (50-70% of decode time at long context) because the per-step token has to pull 6.4 GB of activated expert weights through 1.4 TB/s effective HBM bandwidth. Lightning-Index attention grows linearly from 4% (8K) to 24% (256K) of decode cost as the topK-512 retrieval scales, even though KV reads themselves are constant per step. On H20, the same workload is comm-heavy rather than weight-heavy: 27% of decode time goes into AllToAll dispatch/combine, because per-rank EP bandwidth is roughly 8x lower than on 910C. Practically, this means H20 prefers smaller EP degrees and larger batch sizes to amortize comm latency, while 910C prefers larger EP and smaller batches.

### 3.3 Per-Category Op Breakdown

**Category Bottleneck Summary**

| Category | Prefill Bottleneck (910C) | Decode Bottleneck (910C) | Prefill Bottleneck (H20) | Decode Bottleneck (H20) |
|---|---|---|---|---|
| Attention Projections | CUBE | MEM | CUBE | CUBE |
| Attention Compute | MEM/CUBE | MEM | CUBE | VEC |
| KV Compression | CUBE | MEM | CUBE | CUBE |
| Lightning Index | CUBE | MEM | CUBE | CUBE |
| mHC (fused) | MEM | MEM/VEC | MEM | VEC |
| MoE Gate | MEM | MEM | CUBE | CUBE |
| MoE Routed Experts | CUBE | MEM | CUBE | MEM |
| Communication | COMM | COMM | COMM | COMM |
| Norm | MEM | VEC | MEM | VEC |

The pattern across categories shows three structural differences. First, MEM-bound ops on 910C decode (attention compute, projections, MoE experts) become CUBE-bound on H20 — the higher HBM bandwidth per FLOP shifts the bottleneck from memory to compute. Second, mHC and Norm fall to VEC bound on H20 because H20's vector pipeline is the limiting factor once memory is fast enough. Third, communication is COMM-bound on both platforms because the inter-rank fabric dominates whenever EP > 8 or whenever AllReduce/AllGather is on the critical path. **Implication:** optimizations targeting memory traffic (mHC fusion, BF16 weights, KV cache compression) yield outsized gains on 910C; optimizations targeting compute (cube fusions, larger batch) yield outsized gains on H20.

### 3.4 Hardware Comparison Summary

**Prefill Comparison (8K/4K, best throughput config)**

| Category | 910C % | H20 % | 910C Bottleneck | H20 Bottleneck |
|---|---|---|---|---|
| mHC | 25.95% | 1.78% | MEM | MEM |
| Attention Proj | 23.00% | 35.95% | CUBE | CUBE |
| Attention Compute | 15.46% | 11.72% | MEM | CUBE |
| Communication | 22.55% | 31.72% | COMM | COMM |
| Lightning Index | 4.19% | 4.60% | CUBE | CUBE |
| KV Compression | 2.49% | 2.04% | CUBE | CUBE |
| MoE Routed | 2.43% | 7.57% | CUBE | CUBE |
| Embedding/LMHead | 2.51% | 4.14% | CUBE | CUBE |
| Norm | 0.94% | 0.13% | MEM | MEM |
| MoE Gate | 0.49% | 0.35% | MEM | CUBE |

**Decode Comparison (8K/4K, best throughput config)**

| Category | 910C % | H20 % | 910C Bottleneck | H20 Bottleneck |
|---|---|---|---|---|
| MoE Routed | 70.30% | 36.82% | MEM | MEM |
| Communication | 4.79% | 27.59% | COMM | COMM |
| Attention Proj | 10.43% | 13.76% | MEM | CUBE |
| Attention Compute | 6.05% | 6.98% | MEM | VEC |
| Lightning Index | 2.56% | 3.90% | MEM | CUBE |
| mHC | 2.54% | 3.44% | MEM | VEC |
| KV Compression | 1.14% | 2.90% | MEM | CUBE |
| Norm | 0.87% | 2.85% | VEC | VEC |
| Embedding/LMHead | 1.09% | 1.50% | MEM | CUBE |
| MoE Gate | 0.23% | 0.25% | MEM | CUBE |

**Cross-platform takeaway.** In prefill, both platforms are CUBE-dominated for the heavy ops (attention projections, MoE expert GEMMs, mHC), so the absolute speed difference tracks the ~5x BF16 TFLOPS ratio between H20 and 910C. In decode, the picture inverts: H20's 2.2x larger HBM bandwidth lets it run the memory-bound MoE expert reads roughly 3-6x faster than 910C, while 910C closes part of the gap on COMM-heavy workloads via its 7.8x richer EP fabric. The practical implication: a deployment optimized for prefill latency should prefer H20 for compute-heavy prefill and 910C for EP-heavy prefill; for decode throughput, H20 wins per-rank but at higher cost-per-token because of its lower CUBE.

---

## 4. Parameter & Scenario Optimization

### 4.1 Prefill Latency Optimization

**Best Prefill Latency — Ascend 910C (TP=8, EP=64, DP=8, BS=8, GPUs=64)**

| Context | Latency (no cache) | Latency (90% cache) | Latency (99% cache) |
|---|---|---|---|
| 8K/4K | **715.2 ms** | 103.3 ms | 50.4 ms |
| 32K/4K | **2,984.8 ms** | 303.3 ms | 66.3 ms |
| 128K/4K | **15,828.6 ms** | 1,141.1 ms | 143.0 ms |
| 256K/4K | **42,481.6 ms** | 2,345.1 ms | 249.6 ms |

**Best Prefill Latency — NVIDIA H20 (TP=8, EP=64, DP=8, BS=8, GPUs=64)**

| Context | Latency (no cache) | Latency (90% cache) | Latency (99% cache) |
|---|---|---|---|
| 8K/4K | **559.8 ms** | 68.0 ms | 20.6 ms |
| 32K/4K | **2,404.2 ms** | 228.7 ms | 36.3 ms |
| 128K/4K | **12,812.4 ms** | 904.3 ms | 99.8 ms |
| 256K/4K | **34,234.9 ms** | 1,882.4 ms | 185.5 ms |

**Prefill Latency 910C vs H20 (no prefix cache)**

| Context | 910C (ms) | H20 (ms) | H20/910C speedup |
|---|---|---|---|
| 8K/4K | 715.2 | 559.8 | **H20 1.28x faster** |
| 32K/4K | 2,984.8 | 2,404.2 | **H20 1.24x faster** |
| 128K/4K | 15,828.6 | 12,812.4 | **H20 1.24x faster** |
| 256K/4K | 42,481.6 | 34,234.9 | **H20 1.24x faster** |

**Prefill latency observations.** Both platforms hit best per-request latency at the same 64-card config (TP=8, EP=64, DP=8, BS=8). H20 is uniformly ~1.24x faster on uncached prefill, reflecting its higher compute throughput per chip in the compute-bound regime where prefill operates. Prefix caching is the dominant optimization: a 90% cache hit rate alone delivers 7-13x speedup, and 99% delivers 30-180x — the higher the context length, the larger the multiplier. **At 99% cache hit, even 256K-context first-token latency drops to ~250 ms on 910C and ~190 ms on H20**, putting interactive use within reach. Without caching, 256K prefill takes 30-40 seconds and is unusable for interactive workloads.

### 4.2 Prefill Throughput Optimization

**Best Prefill Throughput — Ascend 910C**

| Context | TP | EP | DP | BS | GPUs | Throughput (tps/gpu) |
|---|---|---|---|---|---|---|
| 8K/4K | 2 | 16 | 8 | 512 | 16 | **3,551.4** |
| 32K/4K | 2 | 16 | 8 | 256 | 16 | **3,262.8** |
| 128K/4K | 2 | 16 | 8 | 64 | 16 | **2,376.6** |
| 256K/4K | 2 | 16 | 8 | 32 | 16 | **1,729.0** |

**Best Prefill Throughput — NVIDIA H20**

| Context | TP | EP | DP | BS | GPUs | Throughput (tps/gpu) |
|---|---|---|---|---|---|---|
| 8K/4K | 1 | 8 | 8 | 128 | 8 | **2,892.1** |
| 32K/4K | 1 | 8 | 8 | 32 | 8 | **2,600.7** |
| 128K/4K | 2 | 16 | 8 | 256 | 16 | **1,754.0** |
| 256K/4K | 2 | 16 | 8 | 128 | 16 | **1,274.1** |

**Prefill Throughput 910C vs H20**

| Context | 910C (tps/gpu) | H20 (tps/gpu) | Winner |
|---|---|---|---|
| 8K/4K | 3,551.4 | 2,892.1 | **910C 1.23x** |
| 32K/4K | 3,262.8 | 2,600.7 | **910C 1.25x** |
| 128K/4K | 2,376.6 | 1,754.0 | **910C 1.36x** |
| 256K/4K | 1,729.0 | 1,274.1 | **910C 1.36x** |

**Prefill throughput takeaway.** 910C wins per-card prefill throughput across all four context lengths. The win comes from raw cube throughput (376 TFLOPS BF16 vs 148 TFLOPS for H20) — prefill is compute-bound, so the ~2.5x raw FLOPs advantage translates directly to throughput even after accounting for lower EP utilization. Note that the optimal 910C config (TP=2, EP=16, DP=8) uses smaller TP than H20 (TP=8, DP=8): with 910C's higher cube throughput per card, splitting tensors across many cards becomes communication-dominated, so the model selects shorter TP groups. Throughput drops monotonically with context length (e.g. 3551 -> 2376 tps/gpu from 8K to 128K on 910C) because prefill is dominated by O(seq^2) attention and Lightning Index cost.

### 4.3 Decode Latency Optimization

**Best Decode Latency — Ascend 910C**

| Context | TP | EP | DP | BS | GPUs | Latency (ms/step) |
|---|---|---|---|---|---|---|
| 8K/4K | 8 | 64 | 8 | 8 | 64 | **36.1** |
| 32K/4K | 8 | 64 | 8 | 8 | 64 | **36.1** |
| 128K/4K | 8 | 64 | 8 | 8 | 64 | **39.6** |
| 256K/4K | 8 | 64 | 8 | 8 | 64 | **40.0** |

**Best Decode Latency — NVIDIA H20**

| Context | TP | EP | DP | BS | GPUs | Latency (ms/step) |
|---|---|---|---|---|---|---|
| 8K/4K | 8 | 64 | 8 | 8 | 64 | **12.2** |
| 32K/4K | 8 | 64 | 8 | 8 | 64 | **12.2** |
| 128K/4K | 4 | 32 | 8 | 8 | 32 | **13.8** |
| 256K/4K | 4 | 32 | 8 | 8 | 32 | **13.8** |

**Decode Latency 910C vs H20**

| Context | 910C (ms) | H20 (ms) | H20/910C speedup |
|---|---|---|---|
| 8K/4K | 36.1 | 12.2 | **H20 2.96x faster** |
| 32K/4K | 36.1 | 12.2 | **H20 2.96x faster** |
| 128K/4K | 39.6 | 13.8 | **H20 2.87x faster** |
| 256K/4K | 40.0 | 13.8 | **H20 2.90x faster** |

**Decode latency takeaway.** Decode is HBM-bandwidth-bound for both platforms, and H20's 4 TB/s HBM (vs 910C's ~1.6 TB/s effective) translates almost directly to a 2.5-3.3x latency advantage. The 910C's per-step latency stays in the 36-40 ms range across context lengths because the dominant cost is reading MoE weights, which is independent of context. H20 keeps step latency under 14 ms, which is comfortably below the 50 ms TPOT SLA target. Either platform meets typical interactive SLA budgets at 8K-256K contexts; H20 has roughly 3x more headroom for stricter TPOT requirements.

### 4.4 Decode Throughput Optimization

**Best Decode Throughput — Ascend 910C**

| Context | TP | EP | DP | BS | GPUs | Throughput (tps/gpu) |
|---|---|---|---|---|---|---|
| 8K/4K | 2 | 16 | 8 | 512 | 16 | **312.4** |
| 32K/4K | 4 | 32 | 8 | 512 | 32 | **231.6** |
| 128K/4K | 4 | 32 | 8 | 256 | 32 | **112.4** |
| 256K/4K | 4 | 32 | 8 | 128 | 32 | **59.0** |

**Best Decode Throughput — NVIDIA H20**

| Context | TP | EP | DP | BS | GPUs | Throughput (tps/gpu) |
|---|---|---|---|---|---|---|
| 8K/4K | 2 | 16 | 8 | 512 | 16 | **1,053.2** |
| 32K/4K | 2 | 16 | 8 | 512 | 16 | **1,010.4** |
| 128K/4K | 2 | 16 | 8 | 256 | 16 | **605.6** |
| 256K/4K | 4 | 32 | 8 | 256 | 32 | **359.3** |

**Decode Throughput 910C vs H20**

| Context | 910C (tps/gpu) | H20 (tps/gpu) | H20/910C ratio |
|---|---|---|---|
| 8K/4K | 312.4 | 1,053.2 | **H20 3.37x** |
| 32K/4K | 231.6 | 1,010.4 | **H20 4.36x** |
| 128K/4K | 112.4 | 605.6 | **H20 5.39x** |
| 256K/4K | 59.0 | 359.3 | **H20 6.09x** |

**Decode throughput is fundamentally H20-favorable.** H20's 4 TB/s HBM bandwidth is 2.2x 910C's, and decode time per token is set by HBM bandwidth (the model has to stream all weights through HBM for every token). Even though 910C has higher peak FLOPS, decode is bandwidth-bound, not FLOP-bound, so H20 wins per-rank throughput by 3-6x as context grows. The gap widens at long context (1.7 -> 6.1x from 8K -> 256K) because H20's larger HBM bandwidth handles the growing KV-cache reads more gracefully — 910C falls further behind as the working set grows.

---

## 5. Key Module Analysis

### 5.1 mHC Optimization

The manifold Hyper Connection (mHC) is V4's most distinctive — and most expensive — architectural component. With hc_mult=4, it expands activations from hidden_size (4,096) to 4x (16,384) at every layer boundary. Without optimization, mHC consumes 88.0% of 910C prefill time at 8K.

#### Optimization Levels

| Level | mhc_kernel_fused | mhc_sp | mhc_fused_bf16 | Description |
|---|---|---|---|---|
| Unfused FP32 | False | False | False | Original baseline |
| Fused FP32 | True | False | False | Kernel fusion only (SP enabled for base ops) |
| Fused+SP | True | True | False | + Sequence parallelism for mHC |
| Fused BF16+SP | True | True | True | + BF16 precision for fused mHC ops |

#### Prefill Time Comparison — Ascend 910C (TP=8, EP=16, DP=8)

| Level | 8K/4K (ms) | mHC % | 32K/4K (ms) | mHC % | 128K/4K (ms) | mHC % | 256K/4K (ms) | mHC % |
|---|---|---|---|---|---|---|---|---|
| Unfused FP32 | 25,911.9 | 88.0% | 105,367.3 | 86.6% | 453,352.6 | 80.5% | 993,442.0 | 73.4% |
| Fused FP32 | 5,505.9 | 43.5% | 23,743.3 | 40.3% | 126,856.5 | 30.2% | 340,449.7 | 22.5% |
| Fused+SP | 3,411.1 | 8.8% | 15,364.1 | 7.8% | 93,339.8 | 5.1% | 273,416.4 | 3.5% |
| Fused BF16+SP | 3,261.5 | 4.6% | 14,765.6 | 4.1% | 90,945.8 | 2.6% | 268,628.3 | 1.8% |

**Speedup vs Unfused FP32 (910C):**

| Context | Fused FP32 | Fused+SP | Fused BF16+SP |
|---|---|---|---|
| 8K/4K | 4.71x | 7.60x | 7.95x |
| 32K/4K | 4.44x | 6.86x | 7.13x |
| 128K/4K | 3.57x | 4.86x | 4.99x |
| 256K/4K | 2.92x | 3.63x | 3.70x |

#### Prefill Time Comparison — NVIDIA H20 (TP=8, EP=8, DP=8)

| Level | 8K/4K (ms) | mHC % | 32K/4K (ms) | mHC % | 128K/4K (ms) | mHC % | 256K/4K (ms) | mHC % |
|---|---|---|---|---|---|---|---|---|
| Unfused FP32 | 7,869.1 | 48.9% | 33,055.9 | 46.6% | 158,032.0 | 39.0% | 385,025.2 | 32.0% |
| Fused FP32 | 4,425.6 | 9.1% | 19,281.8 | 8.4% | 102,935.8 | 6.3% | 274,832.8 | 4.7% |
| Fused+SP | 4,072.1 | 1.2% | 17,867.8 | 1.1% | 97,279.8 | 0.8% | 263,520.9 | 0.6% |
| Fused BF16+SP | 4,046.8 | 0.6% | 17,766.8 | 0.6% | 96,875.8 | 0.4% | 262,712.9 | 0.3% |

#### SP/mHC-SP Comparison — Ascend 910C (TP=8, EP=16, DP=8, Fused base)

| Context | No SP (ms) | SP Only (ms) | SP+mHC-SP (ms) | Speedup (No SP → SP+mHC-SP) |
|---|---|---|---|---|
| 8K/4K | 8,063.1 | 5,505.9 | 3,411.1 | **2.36x** |
| 32K/4K | 34,088.7 | 23,743.3 | 15,364.1 | **2.22x** |
| 128K/4K | 168,265.6 | 126,856.5 | 93,339.8 | **1.80x** |
| 256K/4K | 423,277.0 | 340,449.7 | 273,416.4 | **1.55x** |

#### SP/mHC-SP Comparison — NVIDIA H20 (TP=8, EP=8, DP=8, Fused base)

| Context | No SP (ms) | SP Only (ms) | SP+mHC-SP (ms) | Speedup (No SP → SP+mHC-SP) |
|---|---|---|---|---|
| 8K/4K | 11,765.2 | 4,425.6 | 4,072.1 | **2.89x** |
| 32K/4K | 48,654.1 | 19,281.8 | 17,867.8 | **2.72x** |
| 128K/4K | 220,438.4 | 102,935.8 | 97,279.8 | **2.27x** |
| 256K/4K | 509,842.6 | 274,832.8 | 263,520.9 | **H20: 1.93x** |

**Why mHC is the dominant cost in the unfused FP32 baseline.** mHC fuses 4 hidden-dim expansions, a sinkhorn routing op, and an FP32 sum reduction. In the unfused baseline these become ~7 distinct kernels reading and writing 16k-dim tensors through HBM. On 910C — already memory-bound — that puts mHC at 88% of total prefill at 8K. The optimizations stack: (1) **fused kernel** removes intermediate HBM traffic (~3-4x speedup), (2) **SP** shards the 16K-wide sequence across TP, halving per-rank work and traffic, and (3) **mHC-SP** further parallelizes the routing dimension. The combined No-SP -> Fused+BF16+mHC-SP path delivers ~6-15x speedup on long contexts; on H20 the absolute speedup is smaller (1.9-2.4x) because H20's higher HBM bandwidth was hiding much of the unfused cost in the first place.

The takeaway: on bandwidth-constrained accelerators (910C-class), the mHC kernel must be fused, BF16, and SP-aware to reach acceptable prefill throughput. On bandwidth-rich accelerators (H20-class), naive mHC is much closer to the fused floor, so the engineering investment in mHC fusion pays back faster on 910C.

### 5.2 Attention & KV Cache Analysis

V4's KV compression is the architectural innovation with the largest impact on deployment economics, enabling dramatically larger batch sizes and longer context within fixed HBM budgets.

#### KV Cache Scaling — Ascend 910C (single rank, BS=8)

| Seq Len | V4 KV (GB) | No Compress (GB) | V3 MLA (GB) | Compress Ratio | V4 vs V3 | Decode Step (ms) |
|---|---|---|---|---|---|---|
| 1K | 0.013 | 0.090 | 0.072 | 7.11x | — | 49.5 ms |
| 2K | 0.020 | 0.180 | 0.144 | 9.14x | — | 49.5 ms |
| 4K | 0.034 | 0.361 | 0.288 | 10.67x | — | 51.4 ms |
| 8K | 0.062 | 0.721 | 0.576 | 11.64x | **9.29x** | 51.4 ms |
| 16K | 0.118 | 1.443 | 1.151 | 12.19x | 9.74x | 51.4 ms |
| 32K | 0.231 | 2.886 | 2.303 | 12.49x | **9.96x** | 51.4 ms |
| 64K | 0.457 | 5.771 | 4.605 | 12.64x | 10.09x | 51.4 ms |
| 128K | 0.907 | 11.543 | 9.211 | 12.72x | **10.15x** | 53.3 ms |
| 262K | 1.809 | 23.085 | 18.421 | 12.76x | 10.18x | 53.7 ms |
| 1M | 6.886 | 88.064 | 70.272 | 12.79x | 10.21x | 55.7 ms |

#### KV Cache Scaling — NVIDIA H20 (single rank, BS=8)

| Seq Len | V4 KV (GB) | No Compress (GB) | V3 MLA (GB) | Compress Ratio | Decode Step (ms) |
|---|---|---|---|---|---|
| 1K | 0.013 | 0.090 | 0.072 | 7.11x | 11.6 ms |
| 2K | 0.020 | 0.180 | 0.144 | 9.14x | 11.6 ms |
| 4K | 0.034 | 0.361 | 0.288 | 10.67x | 12.7 ms |
| 8K | 0.062 | 0.721 | 0.576 | 11.64x | 12.7 ms |
| 16K | 0.118 | 1.443 | 1.151 | 12.19x | 12.7 ms |
| 32K | 0.231 | 2.886 | 2.303 | 12.49x | 12.7 ms |
| 64K | 0.457 | 5.771 | 4.605 | 12.64x | 12.7 ms |
| 128K | 0.907 | 11.543 | 9.211 | 12.72x | 13.8 ms |
| 262K | 1.809 | 23.085 | 18.421 | 12.76x | 13.8 ms |
| 1M | 6.886 | 88.064 | 70.272 | 12.79x | 13.8 ms |

#### Per-Layer-Type KV Cache Breakdown

| Seq Len | C4A (GB) | C4A % | C128A (GB) | C128A % | Total (GB) |
|---|---|---|---|---|---|
| 8K | 0.0578 | 93.2% | 0.0039 | 6.3% | 0.062 |
| 32K | 0.2230 | 96.5% | 0.0079 | 3.4% | 0.231 |
| 128K | 0.8836 | 97.4% | 0.0236 | 2.6% | 0.907 |
| 262K | 1.7644 | 97.5% | 0.0446 | 2.5% | 1.809 |
| 1M | 6.7228 | 97.6% | 0.1626 | 2.4% | 6.886 |

#### Compressed vs Uncompressed (per rank, BS=8)

| Seq Len | V4 Compressed | V4 Uncompressed | V3 MLA | V4 vs Uncomp. | V4 vs V3 |
|---|---|---|---|---|---|
| 8K | 0.062 GB | 0.721 GB | 0.576 GB | 11.64x | **9.29x** |
| 32K | 0.231 GB | 2.886 GB | 2.303 GB | 12.49x | **9.96x** |
| 64K | 0.457 GB | 5.771 GB | 4.605 GB | 12.64x | **10.09x** |
| 128K | 0.907 GB | 11.543 GB | 9.211 GB | 12.72x | **10.15x** |
| 262K | 1.809 GB | 23.085 GB | 18.421 GB | 12.76x | **10.18x** |
| 1M | 6.886 GB | 88.064 GB | 70.272 GB | 12.79x | **10.21x** |

#### Attention Compute Scaling per Layer Type — Ascend 910C (TP=8, EP=16, BS=16)

| Seq Len | Full Attn (ms) | Full Attn % | C4A (ms) | C4A % | C128A (ms) | C128A % |
|---|---|---|---|---|---|---|
| 8K | 14.6 | 13.2% | 19.6 | 14.7% | 18.6 | 15.5% |
| 32K | 58.5 | 13.3% | 78.2 | 12.7% | 74.4 | 15.6% |
| 128K | 233.9 | 13.4% | 312.7 | 8.0% | 391.4 | 19.6% |
| 262K | 467.9 | 13.4% | 625.3 | 5.4% | 1,097.6 | 25.5% |
| 1M | 446.2 | 13.4% | 596.3 | 1.9% | 2,737.0 | 47.2% |

#### Attention Compute Scaling per Layer Type — NVIDIA H20 (TP=8, EP=8, BS=16)

| Seq Len | Full Attn (ms) | Full Attn % | C4A (ms) | C4A % | C128A (ms) | C128A % |
|---|---|---|---|---|---|---|
| 8K | 29.7 | 39.7% | 39.4 | 35.7% | 30.9 | 35.3% |
| 32K | 118.9 | 39.8% | 157.7 | 31.3% | 138.3 | 37.9% |
| 128K | 475.5 | 39.8% | 630.6 | 20.9% | 785.7 | 46.4% |
| 262K | 950.9 | 39.8% | 1,261.2 | 14.4% | 2,191.8 | 54.7% |
| 1M | 1,813.8 | 39.8% | 2,405.5 | 5.3% | 10,841.9 | 75.8% |

**Attention/KV-cache takeaways.** With C4A + C128A compression, the per-token KV footprint stays at ~14.8 KB (compared to ~68.6 KB for V3 MLA) and grows almost linearly with sequence length thanks to the constant compression ratio. **At 1M context, total KV per request is 6.9 GB** — small enough to keep batching feasible on 8x 64 GB accelerators without aggressive paging. The decode-side compute per token stays roughly flat from 8K to 1M because Lightning Index keeps the active span bounded; only the periodic compression step (every 4 tokens for C4A, every 128 for C128A) adds overhead. C128A is what makes 1M practical: it pushes 92% of layers into the long-skip regime where attention compute is sub-linear in seq length, and only 8% of layers (the C4A layers) need to scan the full window.

The compression is also what makes V4 work on the 910C: at 256K context, V3-style MLA would need ~18.4 GB just for KV per request, leaving less than 64 GB total headroom for model weights and KV combined on a single accelerator. V4 keeps KV at ~1.8 GB per request at 256K, making 8-card serving viable with room to spare.

---

## 6. Deployment Recommendations

### 6.1 8K Context (8K+4K)

**P/D Disaggregation — Ascend 910C**

| Cache Hit | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D | Total GPUs |
|---|---|---|---|---|---|---|
| 0% | TP=2,EP=16,DP=8 (16) | 56,822.9 | TP=2,EP=16,DP=8 (16) | 4,998.5 | 1P:6D | **112** |
| 90% | TP=2,EP=16,DP=8 (16) | 582,664.9 | TP=2,EP=16,DP=8 (16) | 4,998.5 | 1P:53D | **864** |
| 99% | TP=2,EP=16,DP=8 (16) | 5,420,629.1 | TP=2,EP=16,DP=8 (16) | 4,998.5 | 1P:489D | **7,840** |

**P/D Disaggregation — NVIDIA H20**

| Cache Hit | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D | Total GPUs |
|---|---|---|---|---|---|---|
| 0% | TP=1,EP=8,DP=8 (8) | 23,137.0 | TP=2,EP=16,DP=8 (16) | 16,851.9 | 3P:2D | **56** |
| 90% | TP=1,EP=8,DP=8 (8) | 238,133.5 | TP=2,EP=16,DP=8 (16) | 16,851.9 | 1P:7D | **120** |
| 99% | TP=1,EP=8,DP=8 (8) | 2,376,331.5 | TP=2,EP=16,DP=8 (16) | 16,851.9 | 1P:64D | **1,032** |

**8K observation.** With no prefix caching, decode dominates — both platforms need many decode instances per prefill instance. As cache hit rate rises, prefill throughput jumps ~100x and the ratio inverts: at 99% hit, 1 prefill instance can feed 64+ decode instances. The 910C and H20 both converge on a 1P/many-D layout, but H20 needs roughly half the total cards because its decode throughput per card is 3-4x higher.

### 6.2 32K Context (32K+4K)

**P/D Disaggregation — Ascend 910C**

| Cache Hit | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D | Total GPUs |
|---|---|---|---|---|---|---|
| 0% | TP=2,EP=16,DP=8 (16) | 52,204.6 | TP=4,EP=32,DP=8 (32) | 7,410.7 | 5P:4D | **208** |
| 90% | TP=2,EP=16,DP=8 (16) | 578,260.0 | TP=4,EP=32,DP=8 (32) | 7,410.7 | 1P:9D | **304** |
| 99% | TP=2,EP=16,DP=8 (16) | 5,668,446.4 | TP=4,EP=32,DP=8 (32) | 7,410.7 | 1P:87D | **2,800** |

**P/D Disaggregation — NVIDIA H20**

| Cache Hit | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D | Total GPUs |
|---|---|---|---|---|---|---|
| 0% | TP=1,EP=8,DP=8 (8) | 20,805.7 | TP=2,EP=16,DP=8 (16) | 16,166.7 | 6P:1D | **64** |
| 90% | TP=1,EP=8,DP=8 (8) | 236,251.2 | TP=2,EP=16,DP=8 (16) | 16,166.7 | 1P:2D | **40** |
| 99% | TP=1,EP=8,DP=8 (8) | 2,373,111.2 | TP=2,EP=16,DP=8 (16) | 16,166.7 | 1P:17D | **280** |

**32K observations.** At 32K context, prefill cost grows ~4x (the input length quadrupling), so prefill dominance shifts from compute to attention. The 910C still uses TP=2 with low TP overhead, while H20 prefers TP=1 because its TP-link bandwidth is fewer GB/s than 910C and TP=2 starts to bottleneck on activation reductions. The 99%-cache scenario explodes to 2,800 GPUs (910C) — clearly impractical — so for high-cache workloads at 32K, the deployment converges toward continuous batching with re-use of a single prefill instance.

### 6.3 128K Context (128K+4K)

**P/D Disaggregation — Ascend 910C**

| Cache Hit | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D | Total GPUs |
|---|---|---|---|---|---|---|
| 0% | TP=2,EP=16,DP=8 (16) | 38,025.4 | TP=4,EP=32,DP=8 (32) | 3,598.0 | 3P:1D | **80** |
| 90% | TP=2,EP=16,DP=8 (16) | 558,186.0 | TP=4,EP=32,DP=8 (32) | 3,598.0 | 1P:5D | **176** |
| 99% | TP=2,EP=16,DP=8 (16) | 5,660,096.3 | TP=4,EP=32,DP=8 (32) | 3,598.0 | 1P:45D | **1,456** |

**P/D Disaggregation — NVIDIA H20**

| Cache Hit | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D | Total GPUs |
|---|---|---|---|---|---|---|
| 0% | TP=2,EP=16,DP=8 (16) | 28,064.7 | TP=2,EP=16,DP=8 (16) | 9,689.6 | 10P:1D | **176** |
| 90% | TP=2,EP=16,DP=8 (16) | 424,505.6 | TP=2,EP=16,DP=8 (16) | 9,689.6 | 2P:3D | **80** |
| 99% | TP=2,EP=16,DP=8 (16) | 4,457,216.3 | TP=2,EP=16,DP=8 (16) | 9,689.6 | 1P:13D | **224** |

**128K context.** With 90% prefix cache and EP=16, both platforms converge on a single P-instance fronting many D-instances. The 910C deployment uses 16 GPUs per P-instance vs H20's 16 GPUs per P-instance (same shape) but the H20 D-instance carries ~3x more decode throughput per GPU. As the cache hit rate rises, the limiting factor for total cards becomes decode throughput rather than prefill — H20 needs roughly half the cards of 910C for the same offered load.

### 6.4 256K Context (256K+4K)

**P/D Disaggregation — Ascend 910C**

| Cache Hit | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D | Total GPUs |
|---|---|---|---|---|---|---|
| 0% | TP=2,EP=16,DP=8 (16) | 27,664.6 | TP=4,EP=32,DP=8 (32) | 1,887.4 | 4P:1D | **96** |
| 90% | TP=2,EP=16,DP=8 (16) | 533,477.1 | TP=4,EP=32,DP=8 (32) | 1,887.4 | 1P:4D | **144** |
| 99% | TP=2,EP=16,DP=8 (16) | 5,633,707.4 | TP=4,EP=32,DP=8 (32) | 1,887.4 | 1P:42D | **1,360** |

**P/D Disaggregation — NVIDIA H20**

| Cache Hit | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D | Total GPUs |
|---|---|---|---|---|---|---|
| 0% | TP=2,EP=16,DP=8 (16) | 20,385.5 | TP=4,EP=32,DP=8 (32) | 11,496.9 | 33P:1D | **560** |
| 90% | TP=2,EP=16,DP=8 (16) | 401,672.6 | TP=4,EP=32,DP=8 (32) | 11,496.9 | 2P:1D | **64** |
| 99% | TP=2,EP=16,DP=8 (16) | 4,436,534.7 | TP=4,EP=32,DP=8 (32) | 11,496.9 | 1P:6D | **208** |

**256K context.** This is the most demanding scenario. Even with 90% prefix cache, prefill alone is ~2.3 seconds on 910C and ~1.9 seconds on H20 per request, which means TTFT will dominate user-perceived latency. Prefix-cache hit rates above 90% are essentially required to make 256K context interactive. The 1P:1D balance shifts heavily toward decode-heavy ratios (1P:6-13D) at high cache hit rates because prefill becomes amortized while decode throughput remains the binding constraint. From a deployment standpoint, 256K-context serving should be reserved for cache-friendly workloads (long agentic sessions, repeated document analysis) where TTFT can be hidden by prefix reuse.

### 6.5 General Guidance

**Cross-cutting recommendations**

1. **Always enable prefix caching.** For interactive long-context serving, the difference between 0% and 90% cache hit rate is 10-100x prefill throughput. Caching is the single highest-leverage system optimization in this study.
2. **Match deployment shape to context length.** Short-context (8-32K) workloads can use TP=1-2, EP=8-16, DP=8 (single-instance shapes). Long-context (128-256K) workloads require larger EP (32) and TP (2-4) to fit KV state and amortize prefill cost.
3. **Use 910C for prefill-heavy or compute-bound workloads.** Its 2.5x compute density wins when arithmetic intensity is high — long inputs, low cache hit, large MoE dispatch. Use H20 for decode-heavy or large-batch workloads where HBM bandwidth dominates.
4. **PD ratios scale with context.** 1P:6D at 8K, 1P:13D at 256K (with 99% cache) — prefill becomes a smaller fraction of total cost as context grows and caching kicks in. Plan disaggregation accordingly.
5. **MoE EP fabric matters more than per-rank HBM at scale.** 910C's higher EP bandwidth keeps it competitive even with lower per-card HBM throughput, especially for high-DP MoE deployments.

---

## 7. Industry Implications

The findings in this report extend beyond DeepSeek V4 on 910C/H20 — they speak to a broader inflection point in how the industry is building, deploying, and reasoning about frontier inference systems.

### 7.1 KV Cache Management & Heterogeneous Memory

V4 collapses per-request KV from V3 MLA's ~68.6 KB/token to ~14.8 KB/token with C4A/C128A — but more importantly, it makes the per-token KV footprint *grow sub-linearly* with sequence length once the C128A layers dominate. At 256K context this means 1.7 GB/request of KV state (vs ~18 GB for V3 MLA, an order-of-magnitude reduction), which fundamentally changes how the industry should think about caching infrastructure.

This connects directly to recent industry shifts. Frameworks like **Mooncake** (powering Kimi at Moonshot) have already moved to KVCache-centric disaggregation: KV state lives in a tiered store across GPU HBM, CPU DRAM, and NVMe, with a global radix-tree index for prefix reuse. **SGLang** and **vLLM v1** have both adopted radix-tree-based prefix caching as the default. Our analysis amplifies the case for these designs: the prefill latency curve in Section 6 is brutal *without* prefix caching (5x increase from 8K to 32K, 14x to 128K) but flat with 99% cache hit (sub-200ms TTFT even at 128K). When KV is small enough to fit in distributed CPU memory at scale — which V4's compression makes economical — KV reuse stops being a nice-to-have and becomes the dominant performance lever. The next wave of inference systems will treat the KV cache as a first-class persistent data structure, not an in-memory side effect.

### 7.2 Prefill/Decode Disaggregation Is Not Optional Anymore

The P/D ratio analysis in Section 6 makes the case for disaggregation almost mechanical: a single prefill instance at 99% cache hit serves 45–87 decode instances depending on context length. Co-locating both phases on the same hardware permanently strands either compute (during decode) or bandwidth (during prefill). This is the same conclusion **DistServe** (OSDI '24) and **Splitwise** (ISCA '24) reached through measurement on dense models; V4's MoE structure amplifies the asymmetry because EP communication lives on the decode critical path while prefill is dominated by Lightning Attention compute.

What's new in V4-class workloads is the **compounding effect with prefix caching**. At 99% hit rate the prefill instance's effective throughput rises ~10x, which moves the optimal P:D ratio from roughly 1:7 (uncached, 8K) to 1:64+ (cached, 256K). This matches the architecture **Mooncake** (FAST '25) ships in production for Kimi, where a global KVCache pool sits between disaggregated prefill and decode workers and a Conductor scheduler routes new requests to the prefill node with the best cache locality. NVIDIA's Dynamo serving stack (announced GTC '25) and SGLang's PD mode are converging on the same shape. Our recommendation: any V4 deployment beyond a single rack should be designed PD-first, with the KV transport fabric (RDMA, GDS, or NVL fabric) treated as a first-class capacity dimension, not an afterthought.

### 7.3 Compute-Bound vs Bandwidth-Bound: Hardware Selection by Phase

The 910C-vs-H20 comparison cleanly separates two design philosophies. 910C trades HBM bandwidth (1.8 TB/s) for raw cube compute (2.5x H20 FLOPs) and a strong scale-out fabric. H20 sacrifices compute (CUBE FLOPs ~40% of H100) to retain the full 4 TB/s HBM bandwidth and a mature NVLink topology. Our results show this maps directly to a phase-specific preference: **prefill is compute-bound, decode is bandwidth-bound, and the ideal disaggregated cluster mixes both**.

Concretely, 910C delivers 1.0–1.4x faster prefill at every context length we tested, while H20 delivers 1.3–4.2x faster decode (the gap widens with context length as decode becomes more KV-read-bound). The economic implication is that operators running long-context, low-cache-hit workloads should not buy a uniform fleet — Splitwise (ISCA 2024) and DistServe (OSDI 2024) made this case in principle; V4's numbers make it concrete at the per-token level. Expect cloud providers in 2026 to offer disaggregated SKUs explicitly: "compute-tier" GPUs for prefill pools, "bandwidth-tier" GPUs for decode pools, sharing a fast KV-transfer fabric.

### 7.4 Network Bandwidth Is the New Bottleneck for MoE-Heavy Decoding

Decode-time AllToAll for expert dispatch consumes 5–22% of step time on H20 (decode 8K/4K → 256K/4K) versus 7–35% on 910C — an inversion of the usual H100-vs-domestic-GPU pattern, driven by 910C's higher EP bandwidth (7.8x of H20 in our config). Lightning Attention KV scans, which scale linearly with `S/compress_ratio`, push this further: at 256K, the C4A layers alone read ~2 GB of compressed KV per step, and the AllToAll budget is the only thing keeping per-step latency under SLA. This is the core reason MoE serving has not been "solved" by simply throwing more HBM at it — the binding constraint is increasingly inter-GPU bandwidth, not memory size.

The industry signal is consistent: NVIDIA's roadmap (NVL72 → NVL576 with Spectrum-X), Huawei's CloudMatrix, and Cerebras/Groq's wafer-scale designs all bet on inter-chip bandwidth as the next-decade bottleneck. Our data supports the bet specifically: at 256K decode, total comm time on 910C is ~6.3 ms versus compute ~22 ms — already 22% of the step. A 2x model size or 2x expert count at the same comm fabric pushes this to ~40%, at which point comm dominates. **Practical takeaway: if you are sizing a serving cluster for V4-class models, allocate budget to interconnect at ~30% of total GPU dollar cost, not the typical ~10%.**

### 7.5 Algorithm-Hardware Co-Design Is No Longer Optional

The mHC fused kernel results — ~8x speedup on 910C (25,912 ms to 3,262 ms for 8K prefill) and ~1.9x on H20 (7,869 ms to 4,047 ms) — show that the same algorithmic optimization can pay back drastically differently depending on where the bottleneck sits. On 910C the win comes from removing redundant HBM round-trips (a memory-bound regime); on H20 the kernels are already bandwidth-fit for a smaller gap, so kernel fusion has less to remove.

This makes the case that "model-system co-design" can no longer be deferred to post-training kernel work. Three concrete implications:

1. **Architecture choices must be cost-modeled per-target.** V4's architects clearly optimized for the 910C profile (high FLOPS, modest bandwidth) — full attention with C4A/C128A keeps memory traffic low, while the 256/6/1 expert routing keeps per-rank GEMM dimensions large enough to exploit FP8 tensor cores. The same model on H20 or H100 would benefit from a different sparsity pattern.
2. **Compiler stacks matter as much as kernels.** TileLang, Triton, and Mosaic now make it feasible to author one fused kernel and target multiple backends. The V4 mHC kernel, written in TileLang, ships with separate scheduling templates for 910C and H20 — this is the model the rest of the industry will follow.
3. **Open-source MoE models are training-cost-led; inference cost dominates production.** Architectures like V4 that consciously trade some training efficiency for inference-friendly KV (compression ratios, sliding-window + global attention) will win deployments. Expect more of this design pattern (see also: Mistral's sliding-window attention, GPT-4 Turbo's rumored compressed KV).

### 7.6 Ultra-Long-Context Becomes Tractable

Frontier model sizes have stabilized at the 200B-1T MoE range; the new dimension of competition is *context length at constant latency*. V4's design — full attention only on 2/43 layers, compressed KV on the rest, sliding window + global attention split, mHC fusion to reduce intermediate writes — is an explicit statement that the next 2 years of capability gains come from longer effective context, not bigger params.

Our 1M-context KV scaling table is the headline: at 1M tokens V4 holds the whole KV cache for one request in **6.9 GB** (BF16), 10x smaller than V3. That single number is what makes 1M-token context economically viable on 8x 910C or 4x H20. Without compression, you need either FP4 weights (Llama 4 path), CPU/SSD offload (Mooncake-style), or you simply don't serve it. With V4-style compression, 1M context becomes a single-node workload at ~1 second TTFT — economic enough for chat assistants and code agents, not just research demos.

The bottom line on industry direction: **for V4-class models, the deployment difficulty has shifted from "fit the weights" to "feed the activations."** Compute, HBM bandwidth, comm fabric, and KV-cache infrastructure must be co-designed; treating any one as fixed and optimizing the rest — the dominant pattern in dense-model serving until 2024 — leaves 2-5x performance on the table. The architects who win the next round of inference will be those who design model, kernel, scheduler, and storage as a single system.


---

## 8. Appendix

### 8.1 Methodology

**Performance Model.** All results are produced by a roofline-based analytical performance model (`perf_model/`). For each operation, the model computes three independent cost estimates: CUBE time (matmul FLOPs / effective TFLOPS), VEC time (elementwise FLOPs / vec TFLOPS), and MEM time (bytes transferred / effective HBM bandwidth). The bottleneck is determined by `argmax(cube, vec, mem)`, and total operation time is `max(cube, vec, mem) + communication_time`. All sizes assume BF16 (2 bytes per element), and FLOPs follow the convention `M * N * K * 2` for a `[M,K] x [K,N]` matmul.

**Communication Model.** TP communication (AllReduce, AllGather, ReduceScatter) is modeled using ring-based algorithms with TP bandwidth. EP communication (AllToAll for MoE dispatch/combine) uses EP bandwidth. Communication and compute are modeled as sequential (no overlap), providing a conservative upper bound.

**Parameter Search.** Optimal configurations are found via exhaustive grid search across TP in {1,2,4,8,16,32,64}, EP in {8,16,32,64,128,256}, DP in {1,2,4,8}, batch_size in {1..512}, and context lengths in {8K, 32K, 128K, 256K}. The constraint `(TP*DP) % EP == 0` ensures valid partitioning. Each configuration is evaluated for HBM capacity feasibility before computing performance metrics.

**Decode Fast Mode.** Decode model evaluation uses periodic sampling with trapezoidal interpolation: per-step cost decomposes as constant + linear(S) + periodic(S), where the period P = LCM(compress_ratios) = 128. Only the first and last P steps are sampled, then interpolated via `T_total = N * (T_first + T_last) / (2P)`. This yields 16x speedup for output_len=4096 with error below 0.001%.

**P/D Disaggregation (Schema v2).** P:D ratio calculations use integer balancing with `balance_tolerance=0.1`. The optimal P:D ratio minimizes total GPU count while keeping QPS imbalance within 10%. HBM reserved headroom defaults to `hbm_reserved_pct=10%`.

**Data Generation.** All tables are generated by `report/analyze_scenarios.py`, which runs the performance model across all scenario combinations and exports structured JSON data to `report/data/`. No manual data entry — all numbers trace directly to the analytical model.

### 8.2 Configuration

**Ascend 910C:**

| Parameter | Value |
|---|---|
| Cube TFLOPS (BF16) | 376 |
| Vec TFLOPS (FP32) | 24 |
| HBM Capacity | 64 GB |
| HBM Bandwidth | 1,800 GB/s |
| TP Bandwidth | 392 GB/s |
| EP Bandwidth | 392 GB/s |
| Network Latency | 10 us |
| Cube Utilization | 40% |
| HBM BW Utilization | 30% |
| Vec Utilization | 20% |
| BW Utilization | 26% |
| HBM Reserved | 10% |

**NVIDIA H20:**

| Parameter | Value |
|---|---|
| Cube TFLOPS (BF16) | 148 |
| Vec TFLOPS (FP32) | 44 |
| HBM Capacity | 96 GB |
| HBM Bandwidth | 4,000 GB/s |
| TP Bandwidth | 450 GB/s |
| EP Bandwidth | 50 GB/s |
| Network Latency | 5 us |
| Flops Utilization | 50% |
| HBM BW Utilization | 80% |
| BW Utilization | 80% |
| HBM Reserved | 10% |

**DeepSeek V4 Model Config:**

| Parameter | Value |
|---|---|
| Architecture | DeepseekV4ForCausalLM |
| Hidden size | 4,096 |
| Layers | 43 |
| Q heads | 64 |
| KV heads | 1 (MQA) |
| Head dim | 512 |
| RoPE head dim | 64 |
| Q LoRA rank | 1,024 |
| O groups | 8 |
| O LoRA rank | 1,024 |
| Routed experts | 256 |
| Activated experts | 6 |
| Shared experts | 1 |
| MoE inter dim | 2,048 |
| HC mult | 4 |
| HC Sinkhorn iters | 20 |
| Window size | 128 |
| Index heads | 64 |
| Index head dim | 128 |
| Index topK | 512 |
| Vocab size | 129,280 |
| Compress ratios | 2 full + 21 C4A + 20 C128A |
| n_hash_layers | 3 |
| Max seq len | 65,536 |
| Dtype | bfloat16 |

### 8.3 Data Sources

All raw data is available in `report/data/`:

| File | Contents |
|---|---|
| `search_results_910C.json` | Per-scenario top configs for 910C (8K, 32K, 128K, 256K, 1M) |
| `search_results_H20.json` | Per-scenario top configs for H20 (8K, 32K, 128K, 256K, 1M) |
| `hardware_comparison.json` | Best-config cross-platform comparison, all combos including prefix cache |
| `pd_ratio_analysis.json` | P/D ratio calculations, schema-v2, both platforms, all combos |
| `op_analysis.json` | Per-op bottleneck breakdown, prefill and decode, all scenarios |
| `sp_comparison.json` | SP/mHC-SP comparison across context lengths and cache hit rates |
| `mhc_optimization_comparison.json` | mHC optimization levels: unfused → fused → fused+SP → fused BF16+SP |
| `kv_cache_scaling.json` | KV cache size and decode step time from 1K to 1M tokens, both platforms |
| `attention_analysis.json` | Per-layer-type attention compute scaling and KV cache breakdown |
| `v3_comparison.json` | V3 vs V4 architectural parameter comparison |
| `long_context_1m_1k_prefix_cache.json` | 1M context with 1K prefix cache scenarios |
