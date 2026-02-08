# DeepSeek V4 Inference Performance Analysis — PPT Outline

---

## Slide 1: Title Slide

- **Title**: DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20
- Roofline-based performance modeling across three serving scenarios
- Platforms: Ascend 910C, NVIDIA H20
- Scenarios: 8K/4K, 128K/4K, 256K/4K (input/output token lengths)
- **Visualization**: Title text with platform logos and scenario icons

---

## Slide 2: Executive Summary — Key Findings

- mHC kernel fusion (enabled by default) reduces prefill 3--4x; mHC drops from 84% to 36% of prefill on 910C
- Decode is MoE-weight-bound: routed expert weight loading = 40--56% of decode on 910C
- H20 prefill throughput advantage narrowed from 2.57x to 1.12x (kernel fusion eliminates mHC memory bottleneck)
- P/D disaggregation ratios: 1P:1D (8K), 2P:1D (128K), 3P:1D (256K) on 910C
- 910C EP bandwidth advantage (7.8x) enables high-EP MoE; 910C now achieves lower prefill latency than H20
- **Visualization**: Numbered highlight boxes or icon grid summarizing six key findings

---

## Slide 3: DeepSeek V4 Architecture Overview

- 43 transformer layers: 2 full-attention, 21 C4A (4x compression), 20 C128A (128x compression)
- MQA: 64 Q heads, 1 KV head, head_dim=512 with Q LoRA rank 1,024
- MoE: 256 routed experts (top-6 routing), 1 shared expert, inter_dim=2,048
- mHC (Hyper Connection): FP32 pre/post projections + Sinkhorn normalization at every sub-layer
- Lightning Index: 64 heads, dim=128, topK=512 for compressed KV entry selection
- **Visualization**: Architecture block diagram showing layer types, MoE routing, and mHC data flow

---

## Slide 4: Hardware Comparison — Ascend 910C vs NVIDIA H20

- Cube TFLOPS (BF16): 910C = 376, H20 = 148 (910C 2.54x advantage)
- HBM Bandwidth: 910C = 1,800 GB/s, H20 = 4,000 GB/s (H20 2.22x advantage)
- EP Bandwidth: 910C = 392 GB/s, H20 = 50 GB/s (910C 7.84x advantage)
- HBM Capacity: 910C = 64 GB, H20 = 96 GB (H20 1.5x advantage)
- Cube:Vec ratio: 910C = 15.7:1 (imbalanced), H20 = 3.4:1 (more balanced)
- **Visualization**: Side-by-side comparison table with color-coded advantage indicators (green = winner per metric)

---

## Slide 5: Methodology — Roofline Performance Model

- Each op characterized by three resource dimensions: Cube time, Vec time, Mem time
- Bottleneck = argmax(cube, vec, mem); Total time = max(cube, vec, mem) + comm
- FLOPs convention: matmul [M,K] x [K,N] = M x N x K x 2; BF16 = 2 bytes, FP32 = 4 bytes
- Utilization assumptions: 50% compute, 80% HBM bandwidth (both platforms)
- Search space: TP in {1..64}, EP in {1..256}, DP in {1..8}, BS in {1..512}; constraint: (TP x DP) % EP == 0
- **Visualization**: Roofline diagram showing compute vs memory intensity with example op placements

---

## Slide 6: Parameter Search Results — 8K/4K (Both Platforms)

- 910C Prefill Throughput: 1,656 tps/gpu (TP=8, EP=16, DP=2, BS=256, 16 GPUs)
- H20 Prefill Throughput: 1,848 tps/gpu (TP=8, EP=8, DP=1, BS=128, 8 GPUs) — only 1.12x higher (was 2.57x)
- 910C Decode Throughput: 181 tps/gpu (TP=4, EP=16, DP=4, BS=512, 16 GPUs)
- H20 Decode Throughput: 252 tps/gpu (TP=8, EP=16, DP=2, BS=512, 16 GPUs) — 1.39x higher
- 910C Prefill Latency: 330 ms; H20: 554 ms — 910C 1.68x faster (cube advantage after mHC fusion)
- **Visualization**: Grouped bar chart comparing 910C vs H20 for prefill and decode throughput/latency

---

## Slide 7: Parameter Search Results — 128K/4K (Both Platforms)

- 910C Prefill Throughput: 760 tps/gpu (16 GPUs); H20: 798 tps/gpu (8 GPUs) — H20 1.05x higher
- 910C Decode Throughput: 16.7 tps/gpu (32 GPUs); H20: 47.4 tps/gpu (16 GPUs) — H20 2.84x higher
- Max batch per rank drops to single digits due to KV cache memory at 128K
- 910C Prefill Latency: 10,747 ms; H20: 20,396 ms — 910C 1.90x faster
- 910C Decode Latency: 21.0 ms/step; H20: 9.9 ms/step — H20 2.1x faster
- **Visualization**: Grouped bar chart with 910C vs H20 metrics; annotation for memory constraints

---

## Slide 8: Parameter Search Results — 256K/4K (Both Platforms)

- 910C Prefill Throughput: 482 tps/gpu (16 GPUs); H20: 497 tps/gpu (8 GPUs) — H20 1.03x higher
- 910C Decode Throughput: 8.5 tps/gpu (32 GPUs); H20: 25.6 tps/gpu (32 GPUs) — H20 3.01x higher
- 910C Prefill Latency: 33,923 ms; H20: 65,685 ms — 910C 1.94x faster
- Max batch halves again: BS=8 on 910C, BS=4 on H20 at EP=8
- 256K serving is expensive on both platforms; consider chunking strategies
- **Visualization**: Grouped bar chart; callout box highlighting memory-constrained batch sizes

---

## Slide 9: Long-Context Scaling Trends

- Prefill latency scales super-linearly: 330ms -> 10,747ms -> 33,923ms (8K -> 128K -> 256K) on 910C
- Decode throughput drops dramatically: 181 -> 16.7 -> 8.5 tps/gpu on 910C (8K -> 128K -> 256K)
- Decode latency is relatively stable: 19.3 -> 21.0 -> 21.6 ms/step (compression caps attention cost)
- H20 prefill advantage virtually eliminated at all sequence lengths (~1.0--1.1x throughput)
- H20 advantage widens for decode at longer sequences (more KV cache reads favor 2.2x HBM BW)
- **Visualization**: Dual-axis line chart — prefill latency and decode throughput vs sequence length for both platforms

---

## Slide 10: P/D Disaggregation Concept

- Prefill/Decode disaggregated serving: P instances process input tokens, D instances generate output tokens
- QPS balance equation: N_p x P_rps >= N_d x D_rps
- P:D ratio formula: N_p/N_d >= (D_tps_instance x input_len) / (P_tps_instance x output_len)
- Ratio driven by input/output length imbalance: 2:1 at 8K/4K -> 64:1 at 256K/4K
- Mixed (non-disaggregated) serving wastes 60--80% of GPU resources at long context
- **Visualization**: Conceptual diagram showing P instances feeding D instances with QPS flow arrows

---

## Slide 11: P/D Ratio Results Table

- 910C 8K/4K: 1P:1D (P=16 GPUs, D=16 GPUs) — balanced prefill and decode
- 910C 128K/4K: 2P:1D (P=2x16 GPUs, D=1x32 GPUs) — improved from 4P:1D with kernel fusion
- 910C 256K/4K: 3P:1D (P=3x16 GPUs, D=1x32 GPUs) — improved from 5P:1D with kernel fusion
- H20 128K/4K: 4P:1D; 256K/4K: 14P:1D — H20 needs many more P instances at long context
- Kernel fusion reduced 910C P:D ratios by eliminating the prefill mHC bottleneck
- **Visualization**: Table with color gradient showing P:D ratio escalation across scenarios and platforms

---

## Slide 12: P/D GPU Budget Analysis

- 8K/4K: 32 GPUs (910C) vs 24 GPUs (H20) — H20 25% fewer GPUs
- 128K/4K: 64 GPUs (910C) vs 48 GPUs (H20) — H20 25% fewer GPUs
- 256K/4K: 80 GPUs (910C) vs 144 GPUs (H20) — 910C 44% fewer GPUs
- At 256K, 910C is more GPU-efficient due to H20's extreme 14P:1D ratio
- Kernel fusion dramatically improved 910C's GPU efficiency (was 96/112 at 128K/256K, now 64/80)
- **Visualization**: Stacked bar chart showing prefill GPUs + decode GPUs for each scenario/platform

---

## Slide 13: SP and mHC-SP Comparison (Fused Baseline)

- Three configurations tested: No SP (baseline), SP only, SP + mHC-SP (all with kernel fusion)
- SP alone: significant benefit on H20 (2.67x at 8K, 1.72x at 128K) due to EP AllToAll reduction
- SP alone: modest benefit on 910C (1.41x at 8K) due to fast EP bandwidth
- SP + mHC-SP on 910C at 8K: 2.06x overall speedup (mHC drops from 25.6% to 6.6%)
- Speedup is smaller than on unfused baseline because fusion already addressed the main bottleneck
- **Visualization**: Grouped bar chart showing prefill time (ms) for three configs across 8K/128K/256K

---

## Slide 14: SP+mHC-SP Speedup Analysis

- 910C speedups (No SP -> SP+mHC-SP, fused): 2.06x (8K), 1.39x (128K), 1.23x (256K)
- H20 speedups: 2.90x (8K), 1.78x (128K), 1.48x (256K)
- Speedup diminishes because quadratic attention compute grows and is unaffected by mHC-SP
- After SP+mHC-SP on fused baseline, bottleneck shifts to attention projections (36.9%) and communication (27.1%) at 8K
- For long-context workloads, attention optimization becomes more important than mHC optimization
- **Visualization**: Line chart showing speedup factor vs sequence length for both platforms; annotation for bottleneck shift

---

## Slide 15: mHC Kernel Fusion Optimization

- Four optimization levels: Unfused FP32 -> Fused FP32 -> Fused FP32+SP -> Fused BF16+SP
- Kernel fusion reduces prefill 4x by eliminating mHC HBM round-trips
- On 910C 8K/4K: 10,144ms (unfused) -> 2,492ms (fused) -> 1,706ms (fused+SP) -> 1,650ms (fused BF16+SP)
- mHC drops from 84% to 36% (fused) to 7% (fused+SP) to 3% (fused BF16+SP) of prefill time
- After fusion, bottleneck migrates to attention projections (CUBE-bound) — where 910C excels
- **Visualization**: Waterfall chart showing mHC optimization stages with time reduction

---

## Slide 16: Op Boundedness Analysis — Prefill

- Attention Projections: CUBE-bound (compute-intensive matmuls)
- Attention Compute: CUBE-bound (quadratic in sequence length)
- mHC (fused): MEM-bound but dramatically reduced in absolute time
- MoE Gate and Routed Experts: CUBE-bound during prefill
- With fusion, prefill bottleneck shifts from MEM (mHC) to CUBE (attention) — classic LLM pattern
- **Visualization**: Table with per-category bottleneck indicator (CUBE / VEC / MEM / COMM) with color coding

---

## Slide 17: Op Boundedness Analysis — Decode

- MoE Routed Experts: MEM-bound (weight loading dominates at 40--56% of decode on 910C)
- Attention Projections: MEM-bound (single-token decode -> low arithmetic intensity)
- Attention Compute: MEM-bound (KV cache reads dominate)
- mHC: MEM-bound (small fraction after fusion)
- Communication: COMM-bound (AllToAll for MoE, AllReduce for attention)
- **Visualization**: Table with per-category bottleneck indicator; contrasted side-by-side with prefill slide

---

## Slide 18: Prefill Time Breakdown by Category (%, Fused Baseline)

- 910C 8K: mHC = 36.3%, Attention Proj = 25.4%, Communication = 18.1%, Attention Compute = 7.8%
- 910C 128K: Attention Compute = 31.6%, Lightning Index = 28.2%, mHC = 16.7%
- 910C 256K: Attention Compute = 39.0%, Lightning Index = 35.5%, mHC = 10.6%
- As sequence length increases, attention compute grows quadratically and dominates
- mHC share reduces with sequence length even on fused baseline
- **Visualization**: Stacked bar chart (or stacked area) showing category % breakdown for each scenario and platform

---

## Slide 19: 910C vs H20 Comparison Summary

- Prefill throughput: nearly equal at 1.03--1.12x (H20 advantage eliminated by kernel fusion)
- Prefill latency: 910C wins 1.68--1.94x (cube TFLOPS advantage after mHC bottleneck removed)
- Decode latency: H20 wins consistently at ~2x (directly reflects 2.2x HBM BW ratio)
- 910C EP bandwidth (7.84x) enables EP=32/64 configs that are prohibitive on H20
- At 256K, 910C needs fewer total GPUs (80 vs 144) due to H20's extreme P:D ratio
- **Visualization**: Summary comparison table with green/red arrows indicating platform advantage per metric

---

## Slide 20: Industry Implications and Optimization Opportunities

- Kernel fusion is transformative: 3--4x prefill speedup by eliminating mHC HBM round-trips
- KV compression (C4A/C128A) is essential: without it, 128K context is infeasible at reasonable batch sizes
- Software optimizations can fundamentally change hardware competitiveness (910C went from 2.57x behind to near parity)
- HBM bandwidth remains the decode differentiator: for chatbot workloads, HBM BW > compute TFLOPS
- MoE models strongly favor high-bandwidth interconnects: NVLink-domain efficient, cross-node MoE expensive
- **Visualization**: Matrix chart mapping optimization opportunities to impact level (high/medium/low) and effort

---

## Slide 21: Deployment Recommendations

- 8K/4K (Chat/Coding): 910C = 32 GPUs (1P:1D), H20 = 24 GPUs (1P:1D); both efficient with kernel fusion
- 128K/4K (RAG/Doc QA): 910C = 64 GPUs (2P:1D), H20 = 48 GPUs (4P:1D); P/D disaggregation essential
- 256K/4K (Full Document): 910C = 80 GPUs (3P:1D), H20 = 144 GPUs (14P:1D); 910C more GPU-efficient
- General guidance: kernel fusion + shared expert overlap enabled by default; always use SP; tune EP by network
- Batch aggressively for decode (linear throughput scaling up to memory limit); P/D disaggregation essential for 128K+
- **Visualization**: Decision flowchart or recommendation table by use case with platform-specific configs
