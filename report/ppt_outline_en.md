# DeepSeek V4 Inference Performance Analysis — PPT Outline

---

## Section 1: Executive Summary

### Slide 1: Title Slide

- **Title**: DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20
- Roofline-based performance modeling across four serving scenarios
- Platforms: Ascend 910C, NVIDIA H20
- Scenarios: 8K/4K, 32K/4K, 128K/4K, 256K/4K (input/output token lengths)
- Structure: 8 sections — Architecture, Bottlenecks, Optimization, Advanced, Deployment, Implications, Appendix
- **Visualization**: Title text with platform logos and scenario icons

---

### Slide 2: Executive Summary — Key Findings

- mHC kernel fusion (default) reduces prefill time ~4x; mHC drops from 84.6% to 36.5% of prefill on 910C at 8K
- 910C achieves near-parity on prefill throughput: 1,679 vs 1,872 tps/gpu at 8K (H20 only 1.12x higher)
- Decode is MoE-weight-bound on 910C (46.6% at 8K) and communication-bound on H20 (57.3% at 8K)
- V4's KV compression saves 4.63x memory vs V3 — enabling practical long-context serving
- P/D disaggregation ratios on 910C: 1P:1D (8K), 2P:1D (32K), 3P:1D (128K), 3P:1D (256K)
- 910C's EP bandwidth advantage (7.84x) enables high-EP MoE configs prohibitive on H20
- **Visualization**: Numbered highlight boxes or icon grid summarizing six key findings

---

## Section 2: Model Structure

### Slide 3: DeepSeek V4 Architecture Overview

- 43 transformer layers: 2 full-attention (ratio=1), 21 C4A (4x compression), 20 C128A (128x compression)
- MQA: 64 Q heads, 1 KV head, head_dim=512, Q LoRA rank=1,024
- MoE: 256 routed experts (top-6 routing), 1 shared expert, inter_dim=2,048
- mHC (manifold Hyper Connection): FP32 pre/post projections + Sinkhorn normalization at every sub-layer
- Lightning Index: 64 heads, dim=128, topK=512 for compressed KV entry selection
- KV Compression: group projections compress K and V caches by ratio (C4A=4x, C128A=128x)
- **Visualization**: Architecture block diagram showing layer types, MoE routing, mHC data flow, and KV compression

---

### Slide 4: V4 vs V3 Comparison

| Feature | V4 | V3 |
|---|---|---|
| Total params | ~285B | ~704B |
| Hidden size | 4,096 | 7,168 |
| Layers | 43 | 61 |
| Q heads | 64 | 128 |
| KV heads | 1 | 1 |
| KV approach | MQA + KV compression (C4A/C128A) | MLA (kv_lora_rank=512) |
| KV bytes/token | 15,168 | 70,272 |
| Routed experts | 256 (top-6) | 256 (top-8) |
| MoE inter_dim | 2,048 | 2,048 |
| mHC | Yes (hc_mult=4) | No |
| Lightning Index | Yes (64 heads, dim=128, topK=512) | No |
| Attn FLOPs/token | 226M | 374M |
| MoE FLOPs/token | 352M | 793M |

- KV compression ratio: **4.63x** savings vs V3
- V4 weight memory per rank: 39.92 GB (TP=8, EP=16)
- **Visualization**: Side-by-side comparison table with arrows highlighting key architectural differences

---

### Slide 5: Hardware Platforms — 910C vs H20

| Metric | Ascend 910C | NVIDIA H20 | Advantage |
|---|---|---|---|
| Cube TFLOPS (BF16) | 376 | 148 | 910C 2.54x |
| HBM Bandwidth | 1,800 GB/s | 4,000 GB/s | H20 2.22x |
| EP Bandwidth | 392 GB/s | 50 GB/s | 910C 7.84x |
| HBM Capacity | 64 GB | 96 GB | H20 1.5x |
| Cube:Vec Ratio | 15.7:1 | 3.4:1 | H20 more balanced |

- **Visualization**: Side-by-side comparison table with color-coded advantage indicators

---

## Section 3: Bottleneck Analysis

### Slide 6: Per-Category Bottleneck Summary

| Category | Prefill Bottleneck | Decode Bottleneck |
|---|---|---|
| Attention Proj | CUBE | MEM |
| Attention Compute | CUBE | MEM |
| KV Compression | CUBE | MEM |
| Lightning Index | CUBE | COMM |
| mHC | MEM | MEM |
| MoE Gate | CUBE | MEM |
| MoE Routed | CUBE | MEM |
| Communication | COMM | COMM |

- Prefill is compute-bound (CUBE): attention projections, MoE expert matmuls are CUBE-limited
- Decode is memory-bound (MEM/COMM): MoE weight loading and KV cache reads dominate
- **Visualization**: Table with per-category bottleneck indicator color-coded for prefill and decode

---

### Slide 7: Prefill Op Breakdown — 910C

| Category | 8K | 32K | 128K | 256K |
|---|---|---|---|---|
| mHC | 36.8% | 34.0% | 25.8% | 19.6% |
| Attn Proj | 23.2% | 21.4% | 16.3% | 12.3% |
| Communication | 18.3% | 16.9% | 12.9% | 9.7% |
| KV Compression | 7.4% | 6.9% | 5.2% | 4.0% |
| Lightning Index | 4.4% | 11.2% | 30.0% | 44.4% |
| Attn Compute | 4.5% | 4.7% | 6.1% | 7.2% |
| MoE Routed | 1.8% | 1.6% | 1.2% | 0.9% |
| MoE Gate | 1.3% | 1.2% | 0.9% | 0.7% |
| Embedding/LMHead | 1.9% | 1.8% | 1.3% | 1.0% |
| Norm | 0.3% | 0.3% | 0.2% | 0.2% |

- Trend: mHC dominates at short context; Lightning Index grows quadratically and dominates at 128K+
- **Visualization**: Stacked bar chart showing category % breakdown across 8K/32K/128K/256K

---

### Slide 8: Decode Op Breakdown — 910C

| Category | 8K | 32K | 128K | 256K |
|---|---|---|---|---|
| MoE Routed | 46.6% | 34.7% | 44.7% | 46.4% |
| Communication | 21.6% | 26.2% | 14.8% | 14.3% |
| Lightning Index | 10.1% | 20.9% | 26.3% | 27.2% |
| mHC | 6.8% | 5.0% | 1.6% | 0.8% |
| Attn Compute | 6.6% | 5.9% | 2.5% | 1.3% |
| Attn Proj | 4.3% | 3.8% | 7.2% | 7.4% |
| KV Compression | 3.0% | 2.6% | 1.8% | 1.6% |
| MoE Gate | 0.2% | 0.3% | 0.3% | 0.3% |
| Embedding/LMHead | 0.4% | 0.3% | 0.7% | 0.7% |
| Norm | 0.5% | 0.4% | 0.1% | 0.1% |

- Decode dominated by MoE weight loading (35--47%); at longer sequences Lightning Index KV reads grow
- **Visualization**: Stacked bar chart showing decode category % breakdown across 8K/32K/128K/256K

---

### Slide 9: 910C vs H20 Bottleneck Comparison

**Prefill 8K (top categories):**

| Category | 910C % | 910C Bottleneck | H20 % | H20 Bottleneck |
|---|---|---|---|---|
| mHC | 36.8% | MEM | 9.2% | MEM |
| Attn Proj | 23.2% | CUBE | 32.9% | CUBE |
| Communication | 18.3% | COMM | 27.0% | COMM |
| KV Compression | 7.4% | CUBE | 10.5% | CUBE |

**Decode 8K (top categories):**

| Category | 910C % | 910C Bottleneck | H20 % | H20 Bottleneck |
|---|---|---|---|---|
| MoE Routed | 46.6% | MEM | 18.4% | MEM |
| Communication | 21.6% | COMM | 57.3% | COMM |
| Lightning Index | 10.1% | COMM | 5.0% | COMM |
| mHC | 6.8% | MEM | 2.7% | MEM |

- Root cause: 910C lower HBM BW makes MEM-bound ops worse; H20 50 GB/s EP makes AllToAll dominant
- **Visualization**: Paired bars comparing 910C vs H20 bottleneck profiles for prefill and decode

---

## Section 4: Parameter & Scenario Optimization

### Slide 10: Optimal Throughput Configs — 910C

| Scenario | Phase | TP | EP | DP | BS | GPUs | Throughput |
|---|---|---|---|---|---|---|---|
| 8K/4K | Prefill | 8 | 16 | 2 | 512 | 16 | 1,679 tps/gpu |
| 8K/4K | Decode | 8 | 16 | 2 | 512 | 16 | 307 tps/gpu |
| 32K/4K | Prefill | 8 | 16 | 2 | 128 | 16 | 1,551 tps/gpu |
| 32K/4K | Decode | 8 | 32 | 4 | 512 | 32 | 115 tps/gpu |
| 128K/4K | Prefill | 8 | 16 | 2 | 32 | 16 | 1,179 tps/gpu |
| 128K/4K | Decode | 4 | 32 | 8 | 256 | 32 | 37.2 tps/gpu |
| 256K/4K | Prefill | 8 | 16 | 2 | 16 | 16 | 893 tps/gpu |
| 256K/4K | Decode | 4 | 32 | 8 | 128 | 32 | 19.3 tps/gpu |

- **Visualization**: Table with throughput values and color gradient indicating efficiency

---

### Slide 11: Hardware Comparison — Throughput

| Scenario | Metric | 910C | H20 | Ratio (H20/910C) |
|---|---|---|---|---|
| 8K/4K | Prefill tps/gpu | 1,679 | 1,872 | 1.12x |
| 8K/4K | Decode tps/gpu | 307 | 269 | 0.88x |
| 32K/4K | Prefill tps/gpu | 1,551 | 1,726 | 1.11x |
| 32K/4K | Decode tps/gpu | 115 | 205 | 1.78x |
| 128K/4K | Prefill tps/gpu | 1,179 | 1,314 | 1.11x |
| 128K/4K | Decode tps/gpu | 37.2 | 91.1 | 2.45x |
| 256K/4K | Prefill tps/gpu | 893 | 997 | 1.12x |
| 256K/4K | Decode tps/gpu | 19.3 | 53.5 | 2.77x |

- Prefill throughput near-parity (910C within 12%) due to CUBE TFLOPS advantage
- Decode gap widens with context length: H20's HBM bandwidth dominates
- **Visualization**: Grouped bar chart comparing 910C vs H20 throughput across scenarios

---

### Slide 12: Latency Comparison Across Scenarios

| Scenario | 910C Prefill (ms) | H20 Prefill (ms) | 910C Decode (ms/step) | H20 Decode (ms/step) |
|---|---|---|---|---|
| 8K/4K | 325 | 547 | 19.4 | 9.1 |
| 32K/4K | 1,327 | 2,345 | 19.4 | 9.1 |
| 128K/4K | 6,914 | 12,328 | 20.8 | 9.8 |
| 256K/4K | 18,255 | 32,581 | 21.0 | 9.9 |

- 910C achieves 1.68--1.78x lower prefill latency (CUBE TFLOPS advantage)
- Decode latency stable on both platforms (KV compression caps cost growth)
- H20 decode consistently ~2x faster (HBM bandwidth advantage)
- **Visualization**: Dual-axis chart — prefill latency and decode latency vs sequence length

---

### Slide 13: P/D Disaggregation Ratios

| Scenario | 910C P:D Ratio | 910C Total GPUs | H20 P:D Ratio | H20 Total GPUs |
|---|---|---|---|---|
| 8K/4K | 1P:1D | 32 | 1P:1D | 24 |
| 32K/4K | 2P:1D | 64 | 2P:1D | 32 |
| 128K/4K | 3P:1D | 80 | 5P:1D | 56 |
| 256K/4K | 3P:1D | 80 | 14P:1D | 144 |

- Formula: P/D ratio = (decode_tps_instance x input_len) / (prefill_tps_instance x output_len)
- At 256K, 910C needs 80 GPUs vs H20's 144 GPUs — 910C 44% fewer due to H20's extreme P:D ratio
- **Visualization**: Stacked bar chart showing prefill GPUs + decode GPUs per scenario/platform

---

## Section 5: Advanced Optimization (mHC + Attention/KV)

### Slide 14: mHC Optimization — Four Levels (910C 8K)

| Level | Prefill (ms) | mHC % | Speedup vs Unfused |
|---|---|---|---|
| Unfused FP32 | 10,110 | 84.6% | 1.00x |
| Fused FP32 (default) | 2,458 | 36.5% | 4.11x |
| Fused FP32 + mHC-SP | 1,673 | 6.7% | 6.04x |
| Fused BF16 + mHC-SP | 1,618 | 3.4% | 6.25x |

- Fusion eliminates separate sinkhorn/pre/post ops, ~10x less HBM traffic
- mHC-SP parallelizes mHC across TP ranks (8x further reduction)
- **Visualization**: Waterfall chart showing mHC optimization stages with time reduction at each level

---

### Slide 15: mHC Bottleneck Migration (910C 8K)

| Category | Unfused | Fused | Fused+mHC-SP | Fused BF16+mHC-SP |
|---|---|---|---|---|
| mHC | 84.6% | 36.5% | 6.7% | 3.4% |
| Attn Proj | 5.6% | 23.0% | 33.8% | 35.0% |
| Communication | 4.6% | 18.8% | 27.6% | 28.6% |
| KV Compression | 1.8% | 7.4% | 10.8% | 11.2% |
| Lightning Index | 1.1% | 4.5% | 6.6% | 6.9% |
| Attn Compute | 1.1% | 4.5% | 6.6% | 6.8% |
| MoE | 0.7% | 3.0% | 4.5% | 4.6% |

- Insight: fusion shifts bottleneck from MEM (mHC) to CUBE (attention) — where 910C's 2.54x advantage applies
- **Visualization**: Stacked bar chart showing category % at each optimization level

---

### Slide 16: SP/mHC-SP Comparison

**910C — Prefill time (ms) by SP configuration:**

| Config | 8K | 32K | 128K | 256K |
|---|---|---|---|---|
| No SP | 3,474 | 14,668 | 71,981 | 179,522 |
| SP only | 2,458 | 10,579 | 55,599 | 146,748 |
| SP + mHC-SP | 1,673 | 7,437 | 43,030 | 121,611 |
| Speedup (No SP -> SP+mHC-SP) | 2.08x | 1.97x | 1.67x | 1.48x |

**H20 — Prefill time (ms) by SP configuration:**

| Config | 8K | 32K | 128K | 256K |
|---|---|---|---|---|
| No SP | 11,681 | 48,196 | 216,574 | 496,622 |
| SP only | 4,341 | 18,824 | 99,072 | 261,612 |
| SP + mHC-SP | 3,988 | 17,410 | 93,416 | 250,300 |
| Speedup (No SP -> SP+mHC-SP) | 2.93x | 2.77x | 2.32x | 1.98x |

- H20 benefits more from SP because it reduces expensive EP AllToAll (50 GB/s)
- Speedup diminishes at long context — quadratic attention compute unaffected by SP/mHC-SP
- **Visualization**: Grouped bar chart showing prefill time for No SP / SP / SP+mHC-SP

---

### Slide 17: KV Cache & Compression Analysis

**KV cache size comparison (per batch element):**

| Seq Length | V4 Compressed | V4 Uncompressed | V3 MLA | Compression Ratio |
|---|---|---|---|---|
| 8K | 0.062 GB | 0.721 GB | 0.576 GB | 11.6x |
| 32K | 0.231 GB | 2.886 GB | 2.303 GB | 12.5x |
| 64K | 0.457 GB | 5.771 GB | 4.605 GB | 12.6x |
| 128K | 0.907 GB | 11.543 GB | 9.211 GB | 12.7x |

**Decode latency vs context length (910C, TP=4, EP=32, BS=8):**

| Seq Length | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|---|---|
| Decode (ms) | 18.0 | 18.0 | 19.4 | 19.4 | 19.4 | 19.4 | 19.5 | 20.8 |

- C4A layers dominate KV cache: 93.2% at 8K, 97.4% at 128K
- Decode latency remarkably stable — KV compression caps attention cost growth
- **Visualization**: Dual chart — KV cache size comparison and decode latency vs sequence length

---

### Slide 18: Attention Compute Scaling by Layer Type

**910C — Attention time per layer at 128K (BS=16):**

| Layer Type | Ratio | Attn Time (ms) | Layer Total (ms) | Attn % |
|---|---|---|---|---|
| Full Attention | 1 | 187.2 | 744.3 | 25.1% |
| C4A | 4 | 233.9 | 1,677.7 | 13.9% |
| C128A | 128 | 280.7 | 895.1 | 31.4% |

- C128A layers: despite 128x compression, attention is a larger share because other ops (index, compression) are minimal
- C4A layers: Lightning Index dominates layer time (not attention itself)
- Multi-ratio strategy keeps total decode latency growth sublinear across context lengths
- **Visualization**: Bar chart comparing layer types at 128K

---

## Section 6: Deployment Recommendations

### Slide 19: Deployment Recommendations by Scenario

| Scenario | Use Case | 910C Config | 910C GPUs | H20 Config | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | Chat/Coding | TP=8,EP=16,DP=2 | 32 (1P:1D) | TP=8,EP=8,DP=1 + TP=8,EP=16,DP=2 | 24 (1P:1D) |
| 32K/4K | Document | TP=8,EP=16,DP=2 + TP=8,EP=32,DP=4 | 64 (2P:1D) | TP=8,EP=8,DP=1 + TP=4,EP=16,DP=4 | 32 (2P:1D) |
| 128K/4K | RAG/Doc QA | TP=8,EP=16,DP=2 + TP=4,EP=32,DP=8 | 80 (3P:1D) | TP=8,EP=8,DP=1 + TP=4,EP=16,DP=4 | 56 (5P:1D) |
| 256K/4K | Full Document | TP=8,EP=16,DP=2 + TP=4,EP=32,DP=8 | 80 (3P:1D) | TP=8,EP=8,DP=1 + TP=4,EP=32,DP=8 | 144 (14P:1D) |

- General: enable kernel fusion + SP (default), tune EP by network topology, batch aggressively for decode
- At 256K, 910C is more GPU-efficient (80 vs 144 GPUs) due to H20's extreme P:D ratio
- **Visualization**: Decision flowchart or recommendation table by use case

---

## Section 7: Industry Implications

### Slide 20: Industry Implications

- Software optimization changes hardware competitiveness: mHC fusion moved 910C from 4x behind to near parity
- KV compression is essential for long context: without it, 128K requires 11.5 GB KV per batch element
- HBM bandwidth remains the decode differentiator: for chatbot workloads, HBM BW > compute TFLOPS
- EP bandwidth critical for MoE: 910C's 7.84x advantage enables EP=64; H20 limited to EP<=16
- P/D disaggregation architecture required for 128K+ (mixed serving wastes prefill or decode resources)
- 256K serving is expensive on both platforms (18s/33s prefill latency) — chunked prefill strategies needed
- **Visualization**: Matrix chart mapping implications to impact level and platform relevance

---

## Section 8: Appendix

### Slide 21: Appendix — Methodology & Data Sources

- Roofline model: each op's time = max(cube, vec, mem) + comm; utilization: 50% compute, 80% HBM BW
- FLOPs: matmul [M,K]x[K,N] = MxNxKx2; BF16 = 2 bytes, FP32 = 4 bytes
- Search space: TP in {1..64}, EP in {1..256}, DP in {1..8}, BS in {1..512}; constraint (TP x DP) % EP == 0
- Comm models: AllReduce = 2(n-1)/n x vol/BW, AllToAll = (n-1)/n x vol/BW, AllGather = (n-1)/n x vol/BW
- Limitations: comm modeled as additive (no overlap with compute), flash attention memory model assumed
- Raw data: 10 JSON files in report/data/ — search_results, pd_ratio_analysis, op_analysis, sp_comparison, mhc_optimization, hardware_comparison, v3_comparison, kv_cache_scaling, attention_analysis
- **Visualization**: Roofline diagram showing compute vs memory intensity with example op placements
