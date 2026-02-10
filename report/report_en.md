# DeepSeek V4 Inference Performance Analysis: Ascend 910C vs NVIDIA H20

## 1. Executive Summary

This report presents a roofline-based performance analysis of DeepSeek V4 inference on Ascend 910C and NVIDIA H20 hardware, covering bottleneck profiling, parameter optimization, P/D disaggregation, and cross-platform comparison across four context-length scenarios (8K, 32K, 128K, 256K input with 4K output).

**Key findings:**

1. **KV compression is V4's defining advantage.** The heterogeneous compression schedule (2 full + 21 C4A + 20 C128A layers) reduces KV cache to 0.062 GB per batch element at 8K -- 9.3x smaller than V3's MLA and up to 12.7x smaller than uncompressed at 128K. This enables 10x larger batch sizes within the same HBM budget.

2. **mHC kernel fusion is mandatory for practical deployment.** Without fusion, mHC consumes 84.6% of prefill time. Fused BF16 kernels with Sequence Parallelism reduce this to 3.5%, delivering a 6.25x end-to-end prefill speedup.

3. **The 910C and H20 have complementary strengths.** The 910C achieves 1.68--1.78x faster prefill latency (compute advantage), while the H20 delivers 2.13--2.14x faster decode (memory bandwidth advantage). For decode throughput, the 910C leads at 8K (1.14x) thanks to its 7.84x EP bandwidth advantage, but the H20 dominates at 32K+ (up to 2.77x at 256K).

4. **P/D disaggregation becomes essential at long context.** The P:D GPU ratio grows from 1:1 at 8K to 3:1 (910C) and 14:1 (H20) at 256K. The H20's extreme 14:1 ratio at 256K requires 144 total GPUs versus the 910C's 80, making the 910C more cost-effective for ultra-long context.

5. **Lightning Index emerges as the dominant cost at long context.** Rising from 4.4% of prefill time at 8K to 44.4% at 256K on the 910C, Lightning Index overtakes mHC as the primary bottleneck -- the next frontier for optimization.

6. **EP network bandwidth determines MoE serving economics.** The 910C's 392 GB/s EP bandwidth (7.84x the H20's 50 GB/s) transforms the MoE communication profile: communication is 57.3% of H20 decode time versus 21.6% on the 910C, making datacenter network design a first-order concern for MoE deployment.

---

## 2. Model Structure

### 2.1 DeepSeek V4 Architecture

DeepSeek V4 represents a fundamental redesign of the DeepSeek model family, trading raw parameter count for inference efficiency. At approximately 285B total parameters -- less than half of V3's 704B -- V4 achieves competitive quality through three key architectural innovations:

**MQA with KV Compression.** V4 abandons V3's Multi-head Latent Attention (MLA) in favor of Multi-Query Attention (MQA, 64 Q heads sharing 1 KV head) combined with aggressive KV compression. The 43 layers use a heterogeneous compression schedule: 2 full-attention layers (ratio=1), 21 C4A layers (4x compression), and 20 C128A layers (128x compression). This yields only 15,168 bytes of KV cache per token -- a 4.6x reduction compared to V3's 70,272 bytes per token.

**Lightning Index.** To compensate for the information loss from aggressive KV compression, V4 introduces a learned sparse retrieval mechanism with 64 index heads (dim=128) that selects the top-512 most relevant KV entries per query at each compressed attention layer. This enables the model to maintain quality on long-context tasks while keeping cache footprint small.

**manifold Hyper Connection (mHC).** V4 replaces standard residual connections with a hyper-connection mechanism (hc_mult=4), which expands hidden states by 4x at layer boundaries to improve gradient flow and cross-layer information sharing. While powerful, this creates a 4x amplification of intermediate activation memory, making mHC a significant bottleneck as discussed in Section 5.1.

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

### 2.2 V4 vs V3 Comparison

V4's design philosophy is "smaller model, smarter inference." The following comparison illustrates the tradeoffs:

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
| mHC (manifold Hyper Connection) | Yes (hc_mult=4) | No |
| Lightning Index | Yes (64 heads, dim=128) | No |
| Total params (approx) | ~285B | ~704B |
| KV cache per token | 15,168 bytes | 70,272 bytes |
| **KV compression ratio** | **4.6x less than V3** | Baseline |
| Attn FLOPs per token | 226,492,416 | 374,210,560 |
| MoE FLOPs per token | 352,321,536 | 792,723,456 |
| Attn params per layer | 113,246,208 | 187,105,280 |
| MoE params per layer | 6,468,665,344 | 11,320,164,352 |

The most striking difference is the 2.5x reduction in total parameters (285B vs 704B), which directly translates to lower weight memory requirements -- approximately 40 GB per rank (BF16) for V4 on a TP=8, EP=16 configuration. V4's attention FLOPs per token are 1.65x lower (226M vs 374M), and MoE FLOPs are 2.25x lower (352M vs 793M), driven by fewer activated experts (top-6 vs top-8) and smaller hidden dimensions. The KV cache reduction is even more dramatic: V4 stores only 15,168 bytes per token compared to V3's 70,272 bytes -- a 4.6x improvement -- enabling V4 to serve far larger batch sizes within the same HBM budget. This is the key enabler for V4's superior throughput-per-GPU in memory-constrained deployments.

### 2.3 Hardware Platforms

The two target platforms have fundamentally different hardware profiles, leading to distinct inference characteristics:

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

**910C strengths: compute and EP interconnect.** The 910C delivers 2.54x the BF16 matmul throughput of the H20 (376 vs 148 TFLOPS), making it significantly faster for compute-bound prefill workloads. Critically, the 910C provides 392 GB/s EP bandwidth -- 7.84x the H20's 50 GB/s -- because its inter-node network uses the same high-bandwidth interconnect as intra-node TP. This makes MoE expert-parallel communication dramatically cheaper on the 910C, directly benefiting the all-to-all dispatching required by 256-expert MoE.

**H20 strengths: memory bandwidth and capacity.** The H20's 4,000 GB/s HBM bandwidth (2.22x the 910C's 1,800 GB/s) gives it a decisive advantage in memory-bound operations that dominate decode: loading MoE expert weights, reading KV caches, and executing mHC activations. The H20 also has 50% more HBM capacity (96 GB vs 64 GB), supporting larger batch sizes or longer sequences per GPU. Its Cube:Vec ratio of 3.4:1 (vs 15.7:1 on 910C) indicates a more balanced architecture where vectorized operations are relatively less penalized.

**Net effect.** The 910C wins on prefill latency (1.68--1.78x faster, Table 4.4) due to its compute advantage on matmul-heavy prefill. The H20 wins on decode latency (2.13--2.14x faster) because decode is dominated by MEM-bound weight and KV cache reads. For throughput workloads, the picture is more nuanced and depends heavily on the context length, as detailed in Sections 4 and 6.

---

## 3. Bottleneck Analysis

### 3.1 Per-Category Bottleneck Summary

The bottleneck profile between prefill and decode is sharply divided. Prefill is dominated by CUBE (matmul) operations across attention projections, KV compression, and MoE, since the large sequence dimension creates high arithmetic intensity. Decode, processing a single token at a time, shifts to MEM-bound (memory bandwidth) for nearly every category -- with the exception of communication, which remains COMM-bound in both phases. The only category that is MEM-bound in both phases is mHC, because the fused kernel's main cost is reading and writing the 4x-expanded activations rather than performing matmuls.

| Category | Prefill Bottleneck | Decode Bottleneck |
|---|---|---|
| Attention Projections | CUBE | MEM |
| Attention Compute | CUBE | MEM |
| KV Compression | CUBE | MEM |
| Lightning Index | CUBE/COMM | COMM/MEM |
| mHC (fused) | MEM | MEM |
| MoE Gate | CUBE | MEM |
| MoE Routed Experts | CUBE | MEM |
| Communication | COMM | COMM |

### 3.2 Prefill Op Breakdown (910C, Best Throughput Config, Fused)

On the 910C with the best throughput configuration (TP=8, EP=16, DP=2) and fused mHC kernels, the prefill bottleneck landscape evolves significantly with context length:

| Category | 8K/4K | 32K/4K | 128K/4K | 256K/4K |
|---|---|---|---|---|
| mHC | **36.8%** | **34.0%** | **25.8%** | **19.6%** |
| Attention Proj | 23.2% | 21.4% | 16.3% | 12.3% |
| Lightning Index | 4.4% | 11.2% | 30.0% | 44.4% |
| Communication | 18.3% | 16.9% | 12.9% | 9.7% |
| KV Compression | 7.4% | 6.9% | 5.2% | 4.0% |
| Attention Compute | 4.5% | 4.7% | 6.1% | 7.2% |
| MoE (Gate+Routed) | 3.1% | 2.8% | 2.2% | 1.6% |
| Norm | 0.3% | 0.3% | 0.2% | 0.2% |
| Embedding/LMHead | 1.9% | 1.8% | 1.3% | 1.0% |

At 8K context, mHC dominates at 36.8%, followed by attention projections (23.2%) and communication (18.3%). Even with kernel fusion, the 4x activation expansion in mHC generates substantial HBM traffic that the 910C's 1,800 GB/s bandwidth cannot fully hide. As context length grows, a critical shift occurs: **Lightning Index rises from 4.4% at 8K to 44.4% at 256K**, overtaking mHC as the dominant cost. This is because Lightning Index involves matmuls whose size scales with sequence length (computing index scores over the full KV cache), while mHC cost is sequence-length-independent. Conversely, mHC's share drops from 36.8% to 19.6% as it stays constant in absolute terms. MoE remains a small fraction (1.6--3.1%) because the shared expert overlaps with dispatch communication, and EP=16 distributes the 256 routed experts across enough ranks to keep per-rank expert weight loads manageable.

### 3.3 Decode Op Breakdown (910C, Best Throughput Config)

Decode exhibits a fundamentally different profile from prefill, driven by single-token processing that collapses arithmetic intensity:

| Category | 8K/4K | 32K/4K | 128K/4K | 256K/4K |
|---|---|---|---|---|
| MoE Routed | **46.6%** | **34.7%** | **44.7%** | **46.4%** |
| Communication | 21.6% | 26.2% | 14.8% | 14.3% |
| Lightning Index | 10.1% | 20.9% | 26.3% | 27.2% |
| mHC | 6.8% | 5.0% | 1.6% | 0.8% |
| Attention Compute | 6.6% | 5.9% | 2.5% | 1.3% |
| Attention Proj | 4.3% | 3.8% | 7.2% | 7.4% |
| KV Compression | 3.0% | 2.6% | 1.8% | 1.7% |
| MoE Gate | 0.3% | 0.3% | 0.3% | 0.3% |
| Norm | 0.5% | 0.4% | 0.1% | 0.1% |
| Embedding/LMHead | 0.4% | 0.3% | 0.7% | 0.7% |

MoE routed experts dominate decode at 34.7--46.6% across all context lengths. With batch=512 and DP=2, each rank processes 256 requests but only loads 6 experts per token from the 16 local experts (EP=16), generating 24 MB of weight reads per step per layer -- pure memory bandwidth load. Communication is the second-largest component (14.3--26.2%), comprising MoE all-to-all dispatch/combine plus SP all-gather operations. Lightning Index grows from 10.1% at 8K to 27.2% at 256K as the index key reads scale with total sequence length. Notably, mHC drops to under 1% at 256K in decode -- the per-token mHC cost is tiny because it only processes a single token through the 4x expansion, unlike prefill which must process the entire batch-times-sequence.

### 3.4 910C vs H20 Bottleneck Comparison

The 910C and H20 show the same bottleneck types per category (both are CUBE-bound on prefill matmuls, MEM-bound on decode weight loads), but the **relative proportions differ dramatically** due to their different compute-to-bandwidth ratios:

**Prefill bottleneck comparison (8K/4K):**

| Category | 910C | H20 | 910C Bottleneck | H20 Bottleneck |
|---|---|---|---|---|
| mHC | 36.8% | 9.2% | MEM | MEM |
| Attention Proj | 23.2% | 32.9% | CUBE | CUBE |
| Communication | 18.3% | 27.0% | COMM | COMM |
| KV Compression | 7.4% | 10.5% | CUBE | CUBE |
| Lightning Index | 4.4% | 5.1% | CUBE | CUBE |
| Attention Compute | 4.5% | 5.8% | CUBE | CUBE |
| MoE Routed | 1.8% | 4.9% | CUBE | CUBE |
| MoE Gate | 1.3% | 1.8% | CUBE | CUBE |
| Embedding/LMHead | 1.9% | 2.7% | CUBE | CUBE |

**Decode bottleneck comparison (8K/4K):**

| Category | 910C | H20 | 910C Bottleneck | H20 Bottleneck |
|---|---|---|---|---|
| MoE Routed | 46.6% | 18.4% | MEM | MEM |
| Communication | 21.6% | 57.3% | COMM | COMM |
| Lightning Index | 10.1% | 5.0% | COMM | COMM |
| mHC | 6.8% | 2.7% | MEM | MEM |
| Attention Compute | 6.6% | 2.6% | MEM | MEM |
| Attention Proj | 4.3% | 9.5% | CUBE | CUBE |
| KV Compression | 3.0% | 3.1% | MEM | CUBE |

**Prefill:** On the 910C, mHC is the top cost at 36.8% versus only 9.2% on the H20. This 4x discrepancy arises because mHC is MEM-bound, and the H20's 2.22x higher HBM bandwidth dispatches these reads much faster -- while the 910C's matmul-heavy ops (attention proj at 23.2% on 910C) run faster due to the 2.54x compute advantage, the MEM-bound mHC cannot be hidden. On the H20, attention projections (32.9%) and communication (27.0%) are the dominant costs. Communication is proportionally larger on the H20 because its slower EP bandwidth (50 GB/s vs 392 GB/s) inflates all-to-all times.

**Decode:** The most dramatic contrast is communication: 57.3% of H20 decode time versus 21.6% on the 910C. The H20's 50 GB/s EP bandwidth becomes the critical bottleneck when MoE dispatch/combine must transfer expert activations across 16 ranks each step. The 910C's 392 GB/s EP bandwidth (7.84x faster) makes MoE communication relatively cheap, allowing MoE weight reads to dominate instead (46.6% vs 18.4%). This EP bandwidth difference is the single most important factor explaining why the 910C achieves competitive decode throughput at short context lengths despite its 2.22x lower HBM bandwidth.

---

## 4. Parameter & Scenario Optimization

### 4.1 Optimal Configurations -- Ascend 910C

Grid search across TP in {1..64}, EP in {1..256}, DP in {1..8}, BS in {1..512} reveals a consistent pattern on the 910C: **latency-optimal configs maximize parallelism (TP=8, EP=64, DP=8)** to minimize per-rank work, while **throughput-optimal configs minimize GPU count (TP=8, EP=16, DP=2) and maximize batch size** to amortize fixed costs. Notably, prefill throughput configs remain stable at TP=8, EP=16, DP=2 across all context lengths, because the 910C's compute advantage means prefill scales linearly with batch size until HBM capacity limits the maximum batch. Decode latency configs consistently use TP=4, EP=32, DP=8 -- lower TP than prefill because decode is memory-bound rather than compute-bound, so TP=4 provides sufficient parallelism without the communication overhead of TP=8.

#### 8K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **325 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 512 | 16 | **1,679 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **19.4 ms/step** |
| **Decode Throughput** | 8 | 16 | 2 | 512 | 16 | **307 tps/gpu** |

#### 32K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **1,327 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 128 | 16 | **1,551 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **19.4 ms/step** |
| **Decode Throughput** | 8 | 32 | 4 | 512 | 32 | **115 tps/gpu** |

#### 128K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **6,914 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 32 | 16 | **1,179 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **20.8 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 256 | 32 | **37.2 tps/gpu** |

#### 256K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **18,255 ms** |
| **Prefill Throughput** | 8 | 16 | 2 | 16 | 16 | **893 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **21.0 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 128 | 32 | **19.3 tps/gpu** |

### 4.2 Optimal Configurations -- NVIDIA H20

The H20's optimal configurations reveal a markedly different strategy driven by its bandwidth-rich, compute-limited profile and narrow EP interconnect. Prefill throughput configs consistently settle at TP=8, EP=8, DP=1 (just 8 GPUs) -- the minimum EP that fits model weights in the H20's 96 GB HBM (74.5 GB for weights at EP=8). This is optimal because the H20's 50 GB/s EP bandwidth makes wider EP splits prohibitively expensive in communication overhead; keeping EP=8 minimizes all-to-all transfers. For decode throughput, the H20 tends toward TP=4 with moderate EP (16--32) as context length grows, since the 4,000 GB/s HBM bandwidth efficiently serves MoE weight reads even with fewer EP ranks. Decode latency is remarkably consistent at 9.1--9.9 ms across 8K--256K, reflecting that the H20's memory bandwidth handles KV cache growth efficiently.

#### 8K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **547 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 256 | 8 | **1,872 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.1 ms/step** |
| **Decode Throughput** | 8 | 16 | 2 | 512 | 16 | **269 tps/gpu** |

#### 32K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **2,345 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 64 | 8 | **1,726 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.1 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 512 | 16 | **205 tps/gpu** |

#### 128K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **12,328 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 16 | 8 | **1,314 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.8 ms/step** |
| **Decode Throughput** | 4 | 16 | 4 | 128 | 16 | **91.1 tps/gpu** |

#### 256K Input / 4K Output

| Scenario | TP | EP | DP | BS | GPUs | Metric |
|---|---|---|---|---|---|---|
| **Prefill Latency** | 8 | 64 | 8 | 8 | 64 | **32,581 ms** |
| **Prefill Throughput** | 8 | 8 | 1 | 8 | 8 | **997 tps/gpu** |
| **Decode Latency** | 4 | 32 | 8 | 8 | 32 | **9.9 ms/step** |
| **Decode Throughput** | 4 | 32 | 8 | 256 | 32 | **53.5 tps/gpu** |

### 4.3 P/D Disaggregated Serving

P/D disaggregation separates prefill and decode into independently scaled clusters, each using its own optimal configuration. The P:D ratio (number of prefill instances per decode instance) is determined by matching their throughputs: `P:D = ceil(prefill_tps_instance / decode_tps_instance * output_len / input_len)`.

On the 910C, the P:D ratio grows moderately from 1:1 at 8K to 3:1 at 256K. This reflects the 910C's balanced EP bandwidth: prefill throughput degrades from 26,861 to 14,291 tps/instance as context grows, while decode throughput drops from 4,913 to 618 tps/instance. The ratio stays manageable because the 910C's fast EP interconnect keeps both prefill and decode communication costs proportional.

On the H20, the P:D ratio escalates far more aggressively: from 1:1 at 8K to **14:1 at 256K** (144 total GPUs). This extreme ratio arises because H20 decode throughput at 256K (1,711 tps/instance on 32 GPUs) is surprisingly high relative to prefill (7,974 tps/instance on 8 GPUs), but the input:output ratio of 64:1 means each decoded token was "paid for" by 64 prefilled tokens. The H20 needs 14 prefill instances to keep one decode instance saturated. This has major implications for cluster sizing: ultra-long context on H20 requires a heavily prefill-skewed deployment.

#### Ascend 910C

| Combo | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D Ratio | Total GPUs |
|---|---|---|---|---|---|---|
| **8K/4K** | TP=8,EP=16,DP=2 (16) | 26,861 | TP=8,EP=16,DP=2 (16) | 4,913 | **1P:1D** | 32 |
| **32K/4K** | TP=8,EP=16,DP=2 (16) | 24,818 | TP=8,EP=32,DP=4 (32) | 3,673 | **2P:1D** | 64 |
| **128K/4K** | TP=8,EP=16,DP=2 (16) | 18,863 | TP=4,EP=32,DP=8 (32) | 1,190 | **3P:1D** | 80 |
| **256K/4K** | TP=8,EP=16,DP=2 (16) | 14,291 | TP=4,EP=32,DP=8 (32) | 618 | **3P:1D** | 80 |

#### NVIDIA H20

| Combo | P Config (GPUs) | P tps/inst | D Config (GPUs) | D tps/inst | P:D Ratio | Total GPUs |
|---|---|---|---|---|---|---|
| **8K/4K** | TP=8,EP=8,DP=1 (8) | 14,980 | TP=8,EP=16,DP=2 (16) | 4,303 | **1P:1D** | 24 |
| **32K/4K** | TP=8,EP=8,DP=1 (8) | 13,806 | TP=4,EP=16,DP=4 (16) | 3,272 | **2P:1D** | 32 |
| **128K/4K** | TP=8,EP=8,DP=1 (8) | 10,512 | TP=4,EP=16,DP=4 (16) | 1,457 | **5P:1D** | 56 |
| **256K/4K** | TP=8,EP=8,DP=1 (8) | 7,974 | TP=4,EP=32,DP=8 (32) | 1,711 | **14P:1D** | 144 |

### 4.4 910C vs H20 Comparative Analysis

The cross-platform comparison reveals a clear pattern: the performance gap widens with context length, and the direction depends on the phase.

#### Prefill Throughput (tps/gpu)

| Combo | 910C | H20 | H20/910C | 910C GPUs | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | 1,679 | 1,872 | **1.12x** | 16 | 8 |
| 32K/4K | 1,551 | 1,726 | **1.11x** | 16 | 8 |
| 128K/4K | 1,179 | 1,314 | **1.11x** | 16 | 8 |
| 256K/4K | 893 | 997 | **1.12x** | 16 | 8 |

#### Decode Throughput (tps/gpu)

| Combo | 910C | H20 | H20/910C | 910C GPUs | H20 GPUs |
|---|---|---|---|---|---|
| 8K/4K | 307 | 269 | **910C 1.14x** | 16 | 16 |
| 32K/4K | 115 | 205 | **H20 1.78x** | 32 | 16 |
| 128K/4K | 37.2 | 91.1 | **H20 2.45x** | 32 | 16 |
| 256K/4K | 19.3 | 53.5 | **H20 2.77x** | 32 | 32 |

#### Prefill & Decode Latency

| Combo | 910C Prefill (ms) | H20 Prefill (ms) | 910C/H20 | 910C Decode (ms) | H20 Decode (ms) | H20/910C |
|---|---|---|---|---|---|---|
| 8K/4K | 325 | 547 | **1.68x** | 19.4 | 9.1 | **2.14x** |
| 32K/4K | 1,327 | 2,345 | **1.77x** | 19.4 | 9.1 | **2.14x** |
| 128K/4K | 6,914 | 12,328 | **1.78x** | 20.8 | 9.8 | **2.13x** |
| 256K/4K | 18,255 | 32,581 | **1.78x** | 21.0 | 9.9 | **2.13x** |

**Prefill throughput:** The H20 leads by a remarkably consistent 1.11--1.12x across all context lengths, despite using only 8 GPUs versus 16 for the 910C. This means the H20 achieves roughly 2.2x the throughput *per GPU cluster* -- driven by its ability to run the throughput-optimal config at EP=8 (minimizing communication) while the 910C requires EP=16. The H20's higher HBM bandwidth also helps amortize memory-bound mHC costs more effectively during prefill.

**Decode throughput:** The 910C holds a narrow 1.14x lead at 8K/4K -- the only context length where it wins. This is directly attributable to the 910C's EP bandwidth advantage: at short context with large batch sizes, MoE communication dominates H20 decode (57.3%, Section 3.4), and the 910C's 7.84x faster EP link turns this disadvantage into a net compute win. However, at 32K and beyond, the H20 pulls ahead decisively (1.78--2.77x) as growing KV cache reads and Lightning Index loads shift the bottleneck toward pure HBM bandwidth, where the H20's 2.22x advantage compounds.

**Latency:** The 910C achieves 1.68--1.78x faster prefill latency (compute-bound matmul advantage) while the H20 delivers 2.13--2.14x faster decode (memory-bandwidth-bound advantage). Both ratios are stable across context lengths, confirming that the fundamental hardware characteristics -- not the workload shape -- determine the latency gap.

---

## 5. Key Module Analysis

### 5.1 mHC Optimization

The manifold Hyper Connection (mHC) is V4's most distinctive -- and most expensive -- architectural component. With hc_mult=4, it expands activations from hidden_size (4,096) to 4x (16,384) at every layer boundary, generating enormous HBM traffic for the pre-attention and post-MoE mixing operations. Without optimization, mHC consumes **84.6%** of prefill time. We analyze four progressively aggressive optimization levels:

#### Optimization Levels

| Level | mhc_kernel_fused | mhc_sp | mhc_fused_bf16 | Description |
|---|---|---|---|---|
| Unfused FP32 | False | False | False | Original baseline |
| Fused FP32 | True | False | False | Kernel fusion (new default) |
| Fused FP32 + SP | True | True | False | + Sequence parallelism for mHC |
| Fused BF16 + SP | True | True | True | + BF16 precision for fused ops |

#### Prefill Time Comparison (910C, TP=8, EP=16, DP=2, BS=16)

| Level | 8K/4K | mHC % | 32K/4K | mHC % | Speedup vs Unfused (8K) |
|---|---|---|---|---|---|
| Unfused FP32 | 10,110 ms | 84.6% | 41,188 ms | 83.0% | 1.00x |
| Fused FP32 | 2,458 ms | 36.5% | 10,579 ms | 33.9% | 4.11x |
| Fused FP32 + SP | 1,673 ms | 6.7% | 7,437 ms | 6.0% | 6.04x |
| Fused BF16 + SP | 1,617 ms | 3.5% | 7,213 ms | 3.1% | 6.25x |

The impact of kernel fusion alone is dramatic: a **4.11x speedup** at 8K, reducing mHC from 84.6% to 36.5% of total prefill time. Fusion eliminates the separate sinkhorn iteration kernel (20 iterations of softmax-like normalization) and pre/post mixing kernels, consolidating them into a single kernel pass that reduces HBM traffic by roughly 10x. Adding Sequence Parallelism for mHC (mHC-SP) distributes the 4x-expanded activations across TP ranks, yielding a further 1.47x speedup (6.04x total). The final BF16 optimization halves the bytes-per-element for fused operations, though the incremental gain is modest (6.25x total, only 3.5% improvement over Fused+SP) because the remaining mHC cost is already small at 3.5%.

#### Bottleneck Migration (910C, 8K/4K)

| Category | Unfused FP32 | Fused FP32 | Fused+SP | Fused BF16+SP |
|---|---|---|---|---|
| **mHC** | **84.6%** | **36.5%** | **6.7%** | **3.5%** |
| Attention Proj | 5.6% | 23.0% | 33.8% | 35.0% |
| Communication | 4.6% | 18.8% | 27.6% | 28.6% |
| Attention Compute | 1.1% | 4.5% | 6.6% | 6.8% |
| Lightning Index | 1.1% | 4.5% | 6.7% | 6.9% |
| KV Compression | 1.8% | 7.4% | 10.8% | 11.2% |
| MoE (Gate+Routed) | 0.7% | 3.0% | 4.5% | 4.6% |
| Norm | 0.1% | 0.3% | 0.5% | 0.5% |
| Embedding/LMHead | 0.5% | 1.9% | 2.8% | 2.8% |

The bottleneck migration table shows a textbook example of successive optimization exposing the next bottleneck. As mHC drops from 84.6% to 3.5%, attention projections rise from 5.6% to 35.0%, communication from 4.6% to 28.6%, and KV compression from 1.8% to 11.2%. With the fully optimized Fused BF16+SP configuration, the workload becomes balanced across multiple categories, indicating that further gains require optimizing attention projections or communication rather than mHC alone.

#### SP/mHC-SP Comparison (Fused Baseline, TP=8, EP=16, DP=2)

| Combo | No SP | SP Only | SP + mHC-SP | Speedup (No SP -> SP+mHC-SP) |
|---|---|---|---|---|
| **8K/4K** (910C) | 3,474 ms | 2,458 ms | 1,673 ms | **2.08x** |
| **32K/4K** (910C) | 14,668 ms | 10,579 ms | 7,437 ms | **1.97x** |
| **128K/4K** (910C) | 71,981 ms | 55,599 ms | 43,030 ms | **1.67x** |
| **256K/4K** (910C) | 179,522 ms | 146,748 ms | 121,611 ms | **1.48x** |
| **8K/4K** (H20) | 11,681 ms | 4,341 ms | 3,988 ms | **2.93x** |
| **32K/4K** (H20) | 48,196 ms | 18,824 ms | 17,410 ms | **2.77x** |
| **128K/4K** (H20) | 216,574 ms | 99,072 ms | 93,416 ms | **2.32x** |
| **256K/4K** (H20) | 496,622 ms | 261,612 ms | 250,300 ms | **1.98x** |

SP and mHC-SP jointly deliver substantial prefill speedups, but with diminishing returns at longer sequences. At 8K on the 910C, the combined SP+mHC-SP achieves 2.08x speedup over no-SP; at 256K this drops to 1.48x because SP's benefit -- reducing per-rank activation memory -- is offset by increased AllGather communication at longer sequences. The H20 benefits more from SP (up to 2.93x at 8K) because its higher HBM bandwidth makes the freed memory bandwidth more impactful, and its faster TP link (450 GB/s) keeps AllGather overhead low. The gap between SP-only and SP+mHC-SP is 8.1% on H20 at 8K (4,341 vs 3,988 ms) versus 32% on 910C (2,458 vs 1,673 ms), confirming that the 910C benefits disproportionately from mHC-SP because its lower HBM bandwidth makes the mHC memory traffic a bigger relative bottleneck.

### 5.2 Attention & KV Cache Analysis

V4's KV compression is the architectural innovation with the largest impact on deployment economics, enabling dramatically larger batch sizes and longer context windows within fixed HBM budgets.

#### KV Cache Scaling (910C, TP=4, EP=32, DP=8, BS=8)

| Seq Len | V4 KV Cache | No Compression | V3 MLA | Savings vs Uncomp. | Decode Step |
|---|---|---|---|---|---|
| 1K | 0.013 GB | 0.090 GB | 0.072 GB | 7.1x | 18.0 ms |
| 2K | 0.020 GB | 0.180 GB | 0.144 GB | 9.1x | 18.0 ms |
| 4K | 0.034 GB | 0.361 GB | 0.288 GB | 10.7x | 19.4 ms |
| 8K | 0.062 GB | 0.721 GB | 0.576 GB | 11.6x | 19.4 ms |
| 16K | 0.118 GB | 1.443 GB | 1.151 GB | 12.2x | 19.4 ms |
| 32K | 0.231 GB | 2.886 GB | 2.303 GB | 12.5x | 19.4 ms |
| 64K | 0.457 GB | 5.771 GB | 4.605 GB | 12.6x | 19.5 ms |
| 128K | 0.907 GB | 11.543 GB | 9.211 GB | 12.7x | 20.8 ms |

#### KV Cache Scaling (H20, TP=4, EP=32, DP=8, BS=8)

| Seq Len | V4 KV Cache | No Compression | V3 MLA | Savings vs Uncomp. | Decode Step |
|---|---|---|---|---|---|
| 1K | 0.013 GB | 0.090 GB | 0.072 GB | 7.1x | 8.4 ms |
| 2K | 0.020 GB | 0.180 GB | 0.144 GB | 9.1x | 8.4 ms |
| 4K | 0.034 GB | 0.361 GB | 0.288 GB | 10.7x | 9.1 ms |
| 8K | 0.062 GB | 0.721 GB | 0.576 GB | 11.6x | 9.1 ms |
| 16K | 0.118 GB | 1.443 GB | 1.151 GB | 12.2x | 9.1 ms |
| 32K | 0.231 GB | 2.886 GB | 2.303 GB | 12.5x | 9.1 ms |
| 64K | 0.457 GB | 5.771 GB | 4.605 GB | 12.6x | 9.1 ms |
| 128K | 0.907 GB | 11.543 GB | 9.211 GB | 12.7x | 9.8 ms |

V4's KV cache grows sub-linearly with sequence length due to the heterogeneous compression schedule. At 8K, V4 requires only 0.062 GB per batch element -- compared to 0.721 GB uncompressed and 0.576 GB for V3's MLA -- a 11.6x compression ratio. This ratio improves to 12.7x at 128K because the C128A layers (128x compression) contribute proportionally more at longer sequences. The decode step time remains nearly constant on both platforms (19.4 ms on 910C, 9.1 ms on H20) from 1K through 32K, only increasing at 128K (20.8 ms / 9.8 ms) when Lightning Index cache reads become non-trivial. On the H20, decode remains at 9.1 ms even at 64K -- the 4,000 GB/s bandwidth absorbs the extra KV reads without visible latency impact.

#### Per-Layer-Type KV Cache Breakdown

| Seq Len | C4A (%) | C128A (%) | Total (GB) |
|---|---|---|---|
| 8K | 93.2% | 6.3% | 0.062 |
| 32K | 96.5% | 3.4% | 0.231 |
| 128K | 97.4% | 2.6% | 0.907 |

C4A layers dominate the KV cache (93.2--97.4%), despite C128A layers being nearly as numerous (20 vs 21). This is because C128A's 128x compression reduces each layer's KV contribution to a tiny fraction. At 128K, the 20 C128A layers contribute only 23.6 MB total versus 883.6 MB for the 21 C4A layers. The 2 full-attention layers (SWA with window_size=128) store no persistent KV cache in this model because their window is too small to require significant storage relative to compressed layers.

#### Compressed vs Uncompressed Comparison

| Seq Len | V4 Compressed | V4 Uncompressed | V3 MLA | V4 vs Uncomp. | V4 vs V3 |
|---|---|---|---|---|---|
| 8K | 0.062 GB | 0.721 GB | 0.576 GB | 11.6x | **9.3x smaller** |
| 32K | 0.231 GB | 2.886 GB | 2.303 GB | 12.5x | **10.0x smaller** |
| 64K | 0.457 GB | 5.771 GB | 4.605 GB | 12.6x | **10.1x smaller** |
| 128K | 0.907 GB | 11.543 GB | 9.211 GB | 12.7x | **10.2x smaller** |

V4's compressed KV cache is **9.3--10.2x smaller than V3 MLA** across sequence lengths. This is a transformative improvement: at 128K context with BS=8, V4 needs only 0.907 GB for KV cache versus 9.211 GB for V3 -- freeing 8.3 GB of HBM per rank for additional batch capacity or model weights. In practical terms, this enables V4 to serve 10x more concurrent requests within the same memory envelope, or extend to 10x longer contexts. The compression ratio improves slightly with sequence length (from 9.3x at 8K to 10.2x at 128K) as the highly compressed C128A layers make up a larger share of the effective cache.

#### Attention Compute Scaling (910C, per-layer, TP=8, EP=16, BS=16)

| Layer Type | 8K Attn (ms) | 8K Attn% | 32K Attn (ms) | 32K Attn% | 128K Attn (ms) | 128K Attn% |
|---|---|---|---|---|---|---|
| Full (ratio=1) | 11.7 | 25.0% | 46.8 | 25.1% | 187.2 | 25.1% |
| C4A (ratio=4) | 14.6 | 24.0% | 58.5 | 21.1% | 233.9 | 13.9% |
| C128A (ratio=128) | 12.4 | 24.3% | 52.6 | 25.5% | 280.7 | 31.4% |

#### Attention Compute Scaling (H20, per-layer, TP=8, EP=8, BS=16)

| Layer Type | 8K Attn (ms) | 8K Attn% | 32K Attn (ms) | 32K Attn% | 128K Attn (ms) | 128K Attn% |
|---|---|---|---|---|---|---|
| Full (ratio=1) | 29.7 | 40.0% | 118.9 | 40.1% | 475.5 | 40.1% |
| C4A (ratio=4) | 37.1 | 34.6% | 148.6 | 30.4% | 594.3 | 20.4% |
| C128A (ratio=128) | 30.6 | 35.3% | 133.7 | 37.3% | 713.2 | 44.3% |

The attention compute scaling analysis reveals a counterintuitive pattern: **C128A layers become the most expensive at long context despite 128x KV compression.** At 128K on the 910C, C128A attention takes 280.7 ms per layer (31.4% of layer time) versus 233.9 ms for C4A (13.9%) and 187.2 ms for full attention (25.1%). This happens because C128A layers use Lightning Index to select top-512 entries from the full compressed cache, and the index score computation (64 heads * 128 dim * full_seq_len) grows linearly with sequence length regardless of compression ratio. The C4A layers, which read a 4x-compressed cache directly, have lower attention cost because their effective cache is smaller without needing index-based retrieval.

On the H20, this effect is even more pronounced: C128A attention reaches 713.2 ms (44.3% of layer time) at 128K. The H20 takes 2.54x longer per attention operation than the 910C (713.2 vs 280.7 ms) because attention involves matmuls that are CUBE-bound during prefill, and the H20 has 2.54x less matmul throughput. Full attention layers maintain a stable 25.1% share on 910C and 40.1% on H20 across all sequence lengths -- their SWA window (128 tokens) caps their cost regardless of total context.

---

## 6. Deployment Recommendations

### 6.1 Short Context (8K/4K) -- Chat/Coding

Short-context workloads (chat, coding assistants) prioritize low latency and high throughput per GPU. At 8K/4K, the P:D ratio is 1:1 on both platforms, meaning prefill and decode demand equal GPU resources. This is the most GPU-efficient operating point.

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=8, EP=16, DP=2 (16 GPUs) | 1:1 | 32 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=8, EP=16, DP=2 (16 GPUs) | 1:1 | 24 |

**910C (32 GPUs):** Uses the same TP=8, EP=16, DP=2 config for both prefill and decode. This symmetric setup simplifies deployment -- a single cluster config handles both phases. The 910C delivers 307 tps/gpu decode throughput, 1.14x better than the H20, thanks to its EP bandwidth advantage keeping MoE dispatch fast. Prefill latency of 325 ms meets interactive chat requirements.

**H20 (24 GPUs):** Achieves a more GPU-efficient deployment by using only 8 GPUs for prefill (EP=8 minimizes communication) while scaling decode to 16 GPUs. Despite 25% fewer total GPUs than the 910C, the H20 delivers 1,872 tps/gpu prefill throughput (1.12x the 910C). The tradeoff is higher prefill latency (547 ms vs 325 ms), acceptable for throughput-oriented chat services.

### 6.2 Medium Context (32K/4K) -- Document Processing

At 32K context, the P:D ratio shifts to 2:1, doubling the prefill GPU requirement relative to decode. This reflects the 8x larger input (32K vs 4K output), making prefill the throughput bottleneck.

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=8, EP=32, DP=4 (32 GPUs) | 2:1 | 64 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 2:1 | 32 |

**910C (64 GPUs):** Decode shifts to TP=8, EP=32, DP=4 (32 GPUs) to handle the larger KV cache (0.231 GB at 32K vs 0.062 GB at 8K). The wider EP=32 spreads expert weights across more ranks, reducing per-rank memory pressure. Prefill config remains unchanged at TP=8, EP=16, DP=2. Decode throughput drops to 115 tps/gpu -- the growing KV cache and Lightning Index reads start to dominate.

**H20 (32 GPUs):** Prefill stays at the efficient EP=8 config. Decode uses TP=4, EP=16, DP=4 with 205 tps/gpu -- 1.78x the 910C's decode throughput -- because the H20's 4,000 GB/s HBM bandwidth efficiently serves the larger cache reads. Total GPU count is exactly half the 910C's (32 vs 64), making the H20 substantially more cost-effective at this context length.

### 6.3 Long Context (128K/4K) -- RAG/Document QA

Long-context RAG and document QA workloads push the P:D ratio further apart. Lightning Index becomes the dominant prefill cost (30% on 910C), and KV cache reaches 0.907 GB per batch element at 128K.

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 3:1 | 80 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=16, DP=4 (16 GPUs) | 5:1 | 56 |

**910C (80 GPUs, P:D=3:1):** Decode moves to TP=4, EP=32, DP=8 (32 GPUs) -- lower TP than prefill because decode's memory-bound profile gains little from additional TP parallelism. The 910C's decode throughput drops to 37.2 tps/gpu, yet its EP bandwidth advantage keeps the overall P:D ratio at a moderate 3:1.

**H20 (56 GPUs, P:D=5:1):** The H20's P:D ratio jumps to 5:1 at 128K, requiring 5 prefill instances (40 GPUs) per decode instance (16 GPUs). Despite the H20's superior decode throughput (91.1 tps/gpu, 2.45x the 910C), its smaller prefill instance (8 GPUs) cannot produce tokens fast enough to keep pace. This highlights a key deployment constraint: on the H20, long-context serving becomes increasingly prefill-bound.

### 6.4 Ultra-Long Context (256K/4K) -- Full Document Analysis

Ultra-long context represents the most challenging deployment scenario, with prefill times exceeding 18 seconds on the 910C and 32 seconds on the H20.

| Platform | Prefill | Decode | P:D | Total GPUs |
|---|---|---|---|---|
| **910C** | TP=8, EP=16, DP=2 (16 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 3:1 | 80 |
| **H20** | TP=8, EP=8, DP=1 (8 GPUs) | TP=4, EP=32, DP=8 (32 GPUs) | 14:1 | 144 |

**910C (80 GPUs, P:D=3:1):** The 910C maintains the same config as 128K with a still-manageable 3:1 P:D ratio. Prefill latency of 18.3 seconds is high but acceptable for batch document processing. The 910C's total GPU count stays at 80, unchanged from 128K, because the P:D ratio rounds to the same value (2.77 rounds to 3).

**H20 (144 GPUs, P:D=14:1):** The H20's P:D ratio explodes to 14:1, requiring 112 GPUs for prefill and 32 for decode. This extreme imbalance arises from the combination of long input (256K tokens), the H20's compute-limited prefill (32.6 second latency), and relatively efficient decode (53.5 tps/gpu on 32 GPUs). At this scale, the H20's GPU cost is 1.8x the 910C's (144 vs 80 GPUs), making the 910C more cost-effective for ultra-long context despite the H20's per-GPU advantages. Operators should carefully evaluate whether 256K context is truly required, or whether chunked processing at 128K can meet quality requirements.

### 6.5 General Guidance

**Platform selection by workload:**
- **Short context (up to 8K), decode-throughput-critical:** The 910C is competitive or better (1.14x decode throughput advantage at 8K) due to its EP bandwidth. Choose the 910C if MoE communication is the bottleneck.
- **Medium to long context (32K--128K), throughput-critical:** The H20 offers superior decode throughput (1.78--2.45x) and requires fewer GPUs. The HBM bandwidth advantage dominates once KV cache reads become the bottleneck.
- **Latency-critical prefill:** The 910C is consistently 1.68--1.78x faster on prefill. For interactive applications where time-to-first-token matters, the 910C is preferred.
- **Ultra-long context (256K):** The 910C is strongly preferred -- 80 GPUs vs 144 for the H20, with a manageable 3:1 P:D ratio vs 14:1.

**Key optimization levers:**
- **mHC kernel fusion is mandatory.** Without it, mHC consumes 84.6% of prefill time. Fused BF16 kernels + SP reduce this to 3.5%, yielding 6.25x end-to-end speedup.
- **SP and mHC-SP should be enabled.** Combined speedup of 1.5--2.1x on prefill. The benefit is larger at short context lengths.
- **KV compression is V4's defining advantage.** The 9.3--10.2x reduction vs V3 MLA enables 10x larger batch sizes or 10x longer context within the same HBM budget.
- **P/D disaggregation is essential for long context.** At 128K+, using separate prefill and decode clusters with different TP/EP configurations avoids resource waste from unified configs that compromise on both phases.

---

## 7. Industry Implications

### 7.1 KV Cache Management & Tiered Caching

V4's heterogeneous KV compression -- achieving 9.3--12.7x reduction versus uncompressed caches and 4.6x versus V3's MLA -- fundamentally changes the KV cache management landscape. At 128K context with BS=8, V4 needs only 0.907 GB of KV cache per rank versus 11.543 GB uncompressed, freeing over 10 GB of HBM for additional batch capacity.

This compression interacts powerfully with the tiered KV caching systems now entering production. Frameworks like LMCache and NVIDIA Dynamo extend KV storage from GPU HBM to CPU DRAM and SSDs, achieving up to 15x higher throughput and 2x lower latency by exploiting cache reuse across requests. V4's compressed KV representation makes tiered caching even more attractive: the 0.062 GB per batch element at 8K fits entirely in a modest GPU-side cache tier, while longer context entries (0.907 GB at 128K) can be efficiently staged across DRAM and SSD tiers with minimal bandwidth overhead. The heterogeneous compression schedule means that cache management policies should be compression-ratio-aware: C4A layers contribute 93.2--97.4% of the KV cache, making them the primary targets for eviction and prefetching decisions, while C128A entries are so small (2.6--6.3% of total) that they can be retained in the fastest tier indefinitely.

The industry trend toward KV-cache-centric serving architectures -- exemplified by Mooncake (FAST'25 best paper) and the llm-d project's cache-aware routing -- aligns naturally with V4's design. Models that compress KV cache at the architecture level, rather than relying on post-hoc quantization, can deliver predictable cache sizes and access patterns that tiered storage systems can optimize around.

### 7.2 P/D Disaggregation Architecture

Our P:D ratio analysis reveals a critical deployment planning insight: the GPU allocation between prefill and decode clusters varies dramatically with context length and hardware platform. On the 910C, the ratio grows moderately from 1:1 at 8K to 3:1 at 256K. On the H20, it escalates to 14:1 at 256K, requiring 112 prefill GPUs per 32 decode GPUs.

These findings validate the industry's rapid adoption of P/D disaggregation, which has moved from research (DistServe, OSDI'24) to production deployment in under 18 months. Major inference frameworks -- NVIDIA Dynamo, vLLM, and SGLang -- now support disaggregated serving natively, with Meta, LinkedIn, and Mistral running it in production. The core benefit is avoiding the resource waste of unified configurations that must compromise between compute-bound prefill and memory-bound decode.

V4's architecture makes disaggregation particularly valuable because its prefill and decode profiles diverge sharply. Prefill is dominated by CUBE-bound matmul operations (attention projections at 23.2%, Lightning Index growing to 44.4% at 256K), while decode is dominated by MEM-bound weight reads (MoE routed experts at 34.7--46.6%) and COMM-bound dispatch (14.3--57.3% depending on platform). Using the same hardware configuration for both phases wastes either the 910C's compute advantage during decode or the H20's bandwidth advantage during prefill.

The extreme 14:1 ratio on the H20 at 256K also highlights an emerging challenge: KV cache transfer between prefill and decode workers. At 256K context, the compressed KV cache is approximately 7.3 GB per request (0.907 GB * 8 batch elements), which must be transferred at request-routing time. With the H20's 50 GB/s EP bandwidth, this transfer takes approximately 146 ms -- adding meaningful overhead to the time-to-first-token. This motivates the "KV cache centric" disaggregation architectures where KV data locality drives request routing.

### 7.3 Hardware Design Tradeoffs

The 910C vs H20 comparison reveals a striking lesson about hardware design philosophy for inference accelerators. The 910C's 2.54x compute advantage (376 vs 148 TFLOPS BF16) yields a consistent 1.68--1.78x prefill latency advantage, while the H20's 2.22x memory bandwidth advantage (4,000 vs 1,800 GB/s) yields a 2.13--2.14x decode latency advantage. Neither platform dominates across all workloads.

The more nuanced insight lies in the Cube:Vec ratio. The 910C's 15.7:1 ratio (376 TFLOPS / 24 TFLOPS) means vectorized operations like mHC's elementwise mixing are severely penalized relative to matmul. This is why mHC consumes 36.8% of 910C prefill time but only 9.2% on the H20 (Cube:Vec = 3.4:1) -- the H20's more balanced architecture handles mixed matmul/vector workloads more efficiently. As models increasingly adopt non-standard connection patterns (hyper-connections, state-space mixers, MoE gating), hardware with a balanced compute profile may be more adaptable than peak-FLOPS-optimized designs.

The EP bandwidth disparity (392 vs 50 GB/s, 7.84x) is the single most impactful hardware difference. It transforms MoE from a communication bottleneck (57.3% of H20 decode) to a manageable overhead (21.6% on 910C). This suggests that future inference accelerators should prioritize inter-node bandwidth parity with intra-node bandwidth, particularly for MoE architectures with hundreds of experts. The 910C achieves this by using the same high-bandwidth interconnect for both TP and EP traffic, a design choice that pays dividends for 256-expert MoE.

### 7.4 Network Bandwidth for MoE

The 256-expert MoE dispatch in V4 generates substantial all-to-all communication every layer: with EP=16, each token must be routed to 6 of 16 ranks, generating dispatch and combine traffic proportional to batch_size * activated_experts * hidden_size. On the H20, this communication dominates decode at 57.3% of step time (8K/4K), compared to 21.6% on the 910C.

This gap has direct implications for datacenter network design. At EP=16, the H20 needs 800 GB/s of bisection bandwidth (16 ranks * 50 GB/s per link) to avoid EP communication becoming a bottleneck. Scaling to EP=32 doubles this requirement. Recent work on datacenter-scale high-bandwidth domains (InfiniteHBD, SIGCOMM 2025) demonstrates that optical circuit switching can provide Tbps-level bandwidth for EP traffic, but the cost of such infrastructure must be weighed against the alternative of using fewer, faster-interconnected GPUs like the 910C.

The MegaScale-MoE approach of combining SP+EP parallelism (achieving 14.9--32.9% higher MFU than TP+TP) aligns with our finding that the throughput-optimal 910C configuration uses EP=16 with SP enabled, while the H20 minimizes EP to 8 to avoid communication overhead. The practical takeaway for infrastructure planners: MoE models with 256+ experts require either (a) high-bandwidth inter-node links (>200 GB/s per GPU, as on the 910C) or (b) conservative EP configurations (EP=8) that sacrifice parallelism to stay within network bandwidth limits.

### 7.5 mHC as New Paradigm

V4's manifold Hyper Connection (mHC) replaces standard residual connections with a 4x-expanded mixing operation. Without kernel fusion, this single architectural choice consumes 84.6% of prefill time -- more than all other operations combined. The 4.11x speedup from fusion alone, and 6.25x with fusion + SP + BF16, demonstrates that mHC is only practical when paired with aggressive system-level optimization.

This creates a new pattern in model-system co-design: architectural innovations that provide training benefits (better gradient flow, improved cross-layer information sharing) may impose severe inference costs unless kernel-level optimizations are developed simultaneously. The mHC experience suggests that model architects should evaluate not just training loss but inference roofline characteristics when proposing new connection patterns.

The bottleneck migration analysis (Section 5.1) illustrates a general principle: optimizing one component to near-zero merely exposes the next bottleneck. As mHC dropped from 84.6% to 3.5%, attention projections rose from 5.6% to 35.0%, and communication from 4.6% to 28.6%. This "whack-a-mole" pattern is characteristic of mature inference optimization -- there are no silver bullets, only successive bottleneck elimination.

The recent mHC paper (Manifold-Constrained Hyper-Connections) addresses the stability and scalability concerns of the original approach by constraining the residual connection space onto a specific manifold. This suggests the architecture community is already iterating on the idea, and future variants may reduce the 4x expansion factor while preserving gradient flow benefits.

### 7.6 Ultra-Long Context Serving

Our analysis of 128K and 256K context scenarios reveals several practical challenges that the industry must address as long-context models become standard:

**Prefill latency becomes user-facing.** At 256K on the 910C, best-case prefill latency is 18.3 seconds (64 GPUs). On the H20, it reaches 32.6 seconds. These latencies are tolerable for batch document processing but unacceptable for interactive applications. The industry response -- chunked prefill, speculative decoding, and prefix caching -- only partially mitigates this, as V4's heterogeneous compression schedule means different chunks have different computation profiles.

**Lightning Index dominates at scale.** The index score computation grows linearly with sequence length (64 heads * 128 dim * full_seq_len per layer), causing Lightning Index to rise from 4.4% of prefill time at 8K to 44.4% at 256K on the 910C (42.5% on the H20). This is the most significant architectural bottleneck for long-context V4 serving. Unlike mHC, which was tamed with kernel fusion, Lightning Index involves full-precision matmul operations that are already running at the hardware roofline -- there is no easy 4x speedup waiting to be unlocked. Approximate top-K selection or hierarchical indexing may be necessary.

**P/D ratio explosion on bandwidth-limited hardware.** The H20's P:D ratio reaches 14:1 at 256K (144 total GPUs), compared to the 910C's manageable 3:1 (80 GPUs). This means ultra-long context on bandwidth-limited platforms requires a heavily prefill-skewed cluster, with 78% of GPUs dedicated to prefill. This inefficiency may push operators toward compute-rich platforms for long-context workloads, even when bandwidth-rich platforms offer better per-GPU decode throughput.

**Batch size constraints.** At 256K, even with V4's aggressive KV compression, maximum batch sizes drop to 16 (910C) and 8 (H20) for prefill throughput configurations. The 910C's 64 GB HBM accommodates 16 requests * 0.907 GB KV cache + 24 GB weights, leaving little headroom. Without V4's compression (which would require 11.5 GB per request at 128K), even a single request at 256K would consume 23 GB of KV cache -- nearly half the 910C's HBM. The 4.6x KV compression over V3 is what makes 256K serving feasible at all, but the batch sizes remain too small for efficient GPU utilization in throughput-oriented deployments.

---

## 8. Appendix

### 8.1 Hardware & Model Parameters

All hardware and model parameters are sourced from publicly available specifications and configuration files. The tables below document the exact values used in all performance model calculations throughout this report. Utilization rates (50% flops, 80% HBM bandwidth) represent conservative estimates for production inference workloads.

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
| Flops Utilization | 50% |
| HBM BW Utilization | 80% |
| BW Utilization | 80% |

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
| Dtype | bfloat16 |

**Runtime Defaults:**

| Parameter | Value |
|---|---|
| seq_len | 8,192 |
| batch_size | 8 |
| output_len | 4,096 |
| TP | 8 |
| EP | 64 |
| DP | 8 |
| SP | True |
| mhc_kernel_fused | True |
| shared_expert_overlapped | True |
| moe_load_balance_factor | 1.2 |

### 8.2 Methodology

**Performance Model.** All results are produced by a roofline-based analytical performance model (`perf_model/`). For each operation, the model computes three independent cost estimates: CUBE time (matmul FLOPs / effective TFLOPS), VEC time (elementwise FLOPs / vec TFLOPS), and MEM time (bytes transferred / effective HBM bandwidth). The bottleneck is determined by `argmax(cube, vec, mem)`, and total operation time is `max(cube, vec, mem) + communication_time`. All sizes assume BF16 (2 bytes per element), and FLOPs follow the convention `M * N * K * 2` for a `[M,K] x [K,N]` matmul.

**Communication Model.** TP communication (AllReduce, AllGather, ReduceScatter) is modeled using ring-based algorithms with TP bandwidth. EP communication (AllToAll for MoE dispatch/combine) uses EP bandwidth. Communication and compute are modeled as sequential (no overlap), providing a conservative upper bound.

**Parameter Search.** Optimal configurations are found via exhaustive grid search across TP in {1,2,4,8,16,32,64}, EP in {8,16,32,64,128,256}, DP in {1,2,4,8}, batch_size in {1..512}, and seq_len in {1K..32K}. The constraint `(TP*DP) % EP == 0` ensures valid partitioning. Each configuration is evaluated for HBM capacity feasibility before computing performance metrics.

**Decode Fast Mode.** Decode model evaluation uses periodic sampling with trapezoidal interpolation: per-step cost decomposes as constant + linear(S) + periodic(S), where the period P = LCM(compress_ratios) = 128. Only the first and last P steps are sampled, then interpolated via `T_total = N * (T_first + T_last) / (2P)`. This yields 16x speedup for output_len=4096 with error below 0.001%.

**Data Generation.** All tables and charts are generated by `report/analyze_scenarios.py`, which runs the performance model across all scenario combinations and exports structured JSON data to `report/data/`. No manual data entry is involved -- all numbers trace back to the analytical model.

### 8.3 Data Source Files

All raw data is available in `report/data/`:
- `search_results_910C.json` / `search_results_H20.json` -- per-scenario top-20 configs (4 combos: 8K, 32K, 128K, 256K)
- `pd_ratio_analysis.json` -- P/D ratio calculations
- `op_analysis.json` -- per-op bottleneck breakdown
- `sp_comparison.json` -- SP/mHC-SP comparison
- `mhc_optimization_comparison.json` -- 4 mHC optimization levels
- `hardware_comparison.json` -- cross-platform comparison
- `v3_comparison.json` -- V4 vs V3 architecture comparison
- `kv_cache_scaling.json` -- KV cache scaling across sequence lengths
- `attention_analysis.json` -- per-layer-type attention analysis
