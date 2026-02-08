"""All per-operation cost functions (~30 ops)."""

from typing import List

from .config import Config
from .roofline import OpProfile, roofline_time, allreduce_time, alltoall_time, allgather_time, bytes2


# --- Attention Projections ---

def op_q_proj_dq(T: int, cfg: Config) -> OpProfile:
    """W_dq: [H, q_lora_rank], replicated."""
    H = cfg.model.hidden_size
    qlr = cfg.model.q_lora_rank
    flops = T * H * qlr * 2
    weight_bytes = bytes2(H * qlr)
    act_in = bytes2(T * H)
    act_out = bytes2(T * qlr)
    return roofline_time("q_proj_dq", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_q_proj_uq(T: int, cfg: Config) -> OpProfile:
    """W_uq: [q_lora_rank, Nq/TP * (Dqc)]."""
    qlr = cfg.model.q_lora_rank
    Nq = cfg.model.num_attention_heads
    Dqc = cfg.model.head_dim
    Dr = cfg.model.rope_head_dim # NOTE: currently rope is contained in head dim, but we keep it here for clarity and future flexibility
    TP = cfg.rt.tp
    out_dim = (Nq // TP) * Dqc
    flops = T * qlr * out_dim * 2
    weight_bytes = bytes2(qlr * out_dim)
    act_in = bytes2(T * qlr)
    act_out = bytes2(T * out_dim)
    return roofline_time("q_proj_uq", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_k_proj(T: int, cfg: Config) -> OpProfile:
    """W_k: [H, k_dim], replicated (MQA, 1 KV head)."""
    H = cfg.model.hidden_size
    kd = cfg.model.k_dim
    flops = T * H * kd * 2
    weight_bytes = bytes2(H * kd)
    act_in = bytes2(T * H)
    act_out = bytes2(T * kd)
    return roofline_time("k_proj", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_v_proj(T: int, cfg: Config) -> OpProfile:
    """W_v: [H, v_dim], replicated (MQA, 1 KV head)."""
    H = cfg.model.hidden_size
    vd = cfg.model.v_dim
    flops = T * H * vd * 2
    weight_bytes = bytes2(H * vd)
    act_in = bytes2(T * H)
    act_out = bytes2(T * vd)
    return roofline_time("v_proj", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_wo_a(T: int, cfg: Config) -> OpProfile:
    """wo_a: block-diag, Ng/TP blocks of [Nq/Ng * Dqc, o_lora_rank].
    Total FLOPs = T * Nq/TP * Dqc * o_lora_rank * 2."""
    # TODO: review this part later by human
    Nq = cfg.model.num_attention_heads
    Dqc = cfg.model.head_dim
    olr = cfg.model.o_lora_rank
    TP = cfg.rt.tp
    flops = T * (Nq // TP) * Dqc * olr * 2
    # Weight: Ng/TP blocks, each [Nq/Ng * Dqc, o_lora_rank]
    Ng = cfg.model.o_groups
    weight_elems = (Ng // TP) * (Nq // Ng) * Dqc * olr
    weight_bytes = bytes2(weight_elems)
    act_in = bytes2(T * (Nq // TP) * Dqc)
    act_out = bytes2(T * (cfg.model.o_mid_dim // TP))
    return roofline_time("wo_a", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_wo_b(T: int, cfg: Config) -> OpProfile:
    """wo_b: [o_mid_dim/TP, H], RowParallel."""
    # TODO: review this part later by human
    H = cfg.model.hidden_size
    omd = cfg.model.o_mid_dim
    TP = cfg.rt.tp
    in_dim = omd // TP
    flops = T * in_dim * H * 2
    weight_bytes = bytes2(in_dim * H)
    act_in = bytes2(T * in_dim)
    act_out = bytes2(T * H)
    return roofline_time("wo_b", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_attn_tp_allreduce(T: int, cfg: Config) -> OpProfile:
    """TP AllReduce after wo_b: B*S*H*2 bytes."""
    H = cfg.model.hidden_size
    TP = cfg.rt.tp
    vol = bytes2(T * H)
    t = allreduce_time(vol, TP, cfg.net.tp_bandwidth_gbps,
                       cfg.net.latency_us, cfg.net.bandwidth_utilization)
    return OpProfile(name="attn_tp_allreduce", comm_bytes=vol, comm_time_s=t,
                     time_s=t, bottleneck="COMM" if t > 0 else "")


# --- Lightning Index ---

def op_index_iq_proj(T: int, cfg: Config) -> OpProfile:
    """W_iq: ColumnParallel [H, index_n_heads/TP * index_head_dim]."""
    in_dim = cfg.model.q_lora_rank
    TP = cfg.rt.tp
    nh = cfg.model.index_n_heads
    hd = cfg.model.index_head_dim
    out_dim = (nh // TP) * hd
    flops = T * in_dim * out_dim * 2
    weight_bytes = bytes2(in_dim * out_dim)
    act_in = bytes2(T * in_dim)
    act_out = bytes2(T * out_dim)
    return roofline_time("index_iq_proj", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_index_ik_proj(T: int, cfg: Config) -> OpProfile:
    """W_ik: replicated [H, index_head_dim] (1 head)."""
    H = cfg.model.hidden_size
    hd = cfg.model.index_head_dim
    flops = T * H * hd * 2
    weight_bytes = bytes2(H * hd)
    act_in = bytes2(T * H)
    act_out = bytes2(T * hd)
    return roofline_time("index_ik_proj", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_index_kv_compression_prefill(B: int, S: int, ratio: int, cfg: Config) -> OpProfile:
    """Index key compression for Lightning Index.
    4 projections [d', d'] + group compression (g tokens -> 1).
    Cube: 8*B*S*d'^2 + (8g-2)*B*d'*(S/g)
    Vec:  (4g+1)*d'*B*(S/g)
    Mem:  5*B*S*d' elements
    """
    d = cfg.model.index_head_dim   # d_c' = 128
    g = ratio                       # group size = compression ratio
    S_comp = S // g

    cube_flops = 8 * B * S * d**2 + (8 * g - 2) * B * d * S_comp
    vec_ops = (4 * g + 1) * d * B * S_comp
    mem_bytes = bytes2(5 * B * S * d)

    return roofline_time("index_kv_compress", cube_flops, vec_ops, mem_bytes, cfg.hw)

def op_index_kv_compression_decode(B: int, S_total: int, ratio: int, cfg: Config) -> OpProfile:
    """Index key compression for decode: exact per-step cost.
    When S_total % ratio == 0: projections + group compression.
    Otherwise: projections only (no group compression).
    """
    d = cfg.model.index_head_dim   # d_c' = 128
    g = ratio

    if S_total % g == 0:
        cube_flops = 8 * B * d**2 + (8 * g - 2) * B * d
        vec_ops    = (4 * g + 1) * B * d
        mem_elems  = 10 * B * d + 4 * d**2
    else:
        cube_flops = 8 * B * d**2
        vec_ops    = 0
        mem_elems  = 5 * B * d + 4 * d**2

    return roofline_time("index_kv_compress_decode", cube_flops, vec_ops, bytes2(mem_elems), cfg.hw)


def op_index_score(B: int, S: int, ratio: int, cfg: Config) -> OpProfile:
    """Score: Sigma_j w*ReLU(iq*ik^T).
    FLOPs = B * (index_n_heads/TP) * S * (S//ratio) * index_head_dim * 2."""
    TP = cfg.rt.tp
    nh = cfg.model.index_n_heads // TP
    hd = cfg.model.index_head_dim
    S_compressed = S // ratio
    flops = B * nh * S * S_compressed * hd * 2
    # Act memory: iq[B, nh, S, hd] + ik[B, 1, S_compressed, hd] + scores[B, S, S_compressed]
    act_in = bytes2(B * nh * S * hd) + bytes2(B * S_compressed * hd)
    act_out = B * S * S_compressed * 4  # FP32 scores
    return roofline_time("index_score", flops, 0, act_in + act_out, cfg.hw)


def op_index_score_allreduce(B: int, S: int, ratio: int, cfg: Config) -> OpProfile:
    """AllReduce partial scores across TP: B*S*(S//ratio)*4 bytes (FP32)."""
    TP = cfg.rt.tp
    S_compressed = S // ratio
    vol = B * S * S_compressed * 4  # FP32
    t = allreduce_time(vol, TP, cfg.net.tp_bandwidth_gbps,
                       cfg.net.latency_us, cfg.net.bandwidth_utilization)
    return OpProfile(name="index_score_ar", comm_bytes=vol, comm_time_s=t,
                     time_s=t, bottleneck="COMM" if t > 0 else "")


def op_index_score_decode(B: int, S_total: int, ratio: int, cfg: Config) -> OpProfile:
    """Decode: score 1 query against S_total//ratio index keys.
    FLOPs = B * (index_n_heads/TP) * 1 * (S_total//ratio) * index_head_dim * 2."""
    TP = cfg.rt.tp
    nh = cfg.model.index_n_heads // TP
    hd = cfg.model.index_head_dim
    S_compressed = S_total // ratio
    flops = B * nh * S_compressed * hd * 2
    # Memory: read all index K cache + query
    act_in = bytes2(B * S_compressed * hd) + bytes2(B * nh * hd)
    act_out = B * S_compressed * 4  # FP32 scores
    return roofline_time("index_score", flops, 0, act_in + act_out, cfg.hw)


def op_index_score_allreduce_decode(B: int, S_total: int, ratio: int, cfg: Config) -> OpProfile:
    """Decode: AllReduce scores, B*1*(S_total//ratio)*4 bytes."""
    TP = cfg.rt.tp
    S_compressed = S_total // ratio
    vol = B * S_compressed * 4
    t = allreduce_time(vol, TP, cfg.net.tp_bandwidth_gbps,
                       cfg.net.latency_us, cfg.net.bandwidth_utilization)
    return OpProfile(name="index_score_ar", comm_bytes=vol, comm_time_s=t,
                     time_s=t, bottleneck="COMM" if t > 0 else "")


# --- KV Compression Placeholder ---

def op_kv_compression_prefill(B: int, S: int, ratio: int, cfg: Config) -> OpProfile:
    """KV compression for prefill: compress both K and V caches.
    Per K/V: 4 projections [d, d] + group compression (g tokens -> 1).
    Cube: 2*[8*B*S*d^2 + (8g-2)*B*d*(S/g)]  (x2 for K and V)
    Vec:  2*(4g+1)*d*B*(S/g)                  (x2 for K and V)
    Mem:  2*5*B*S*d elements                   (x2 for K and V)
    """
    d = cfg.model.head_dim   # d_c = 512
    g = ratio                 # group size = compression ratio
    S_comp = S // g

    cube_flops = 2 * (8 * B * S * d**2 + (8 * g - 2) * B * d * S_comp)
    vec_ops = 2 * (4 * g + 1) * d * B * S_comp
    mem_bytes = bytes2(2 * 5 * B * S * d)

    return roofline_time("kv_compression", cube_flops, vec_ops, mem_bytes, cfg.hw)


def op_kv_compression_decode(B: int, S_total: int, ratio: int, cfg: Config) -> OpProfile:
    """KV compression for decode: exact per-step cost for both K and V.
    When S_total % ratio == 0: projections + group compression.
    Otherwise: projections only (no group compression).
    """
    d = cfg.model.head_dim   # d_c = 512
    g = ratio

    if S_total % g == 0:
        cube_flops = 2 * (8 * B * d**2 + (8 * g - 2) * B * d)
        vec_ops    = 2 * (4 * g + 1) * B * d
        mem_elems  = 2 * (10 * B * d + 4 * d**2)
    else:
        cube_flops = 2 * 8 * B * d**2
        vec_ops    = 0
        mem_elems  = 2 * (5 * B * d + 4 * d**2)

    return roofline_time("kv_compression_decode", cube_flops, vec_ops, bytes2(mem_elems), cfg.hw)


# --- Attention Score Computation ---

def op_attention_prefill_full(B: int, S: int, cfg: Config) -> OpProfile:
    """Prefill attention for ratio=1 layers (full MQA, flash attention memory model).
    QK^T + softmax + Score*V."""
    TP = cfg.rt.tp
    Nq = cfg.model.num_attention_heads // TP
    kd = cfg.model.k_dim
    vd = cfg.model.v_dim
    Dqc = cfg.model.head_dim
    Dr = cfg.model.rope_head_dim

    # QK^T: B * Nq * S * S * k_dim * 2
    flops_qk = B * Nq * S * S * kd * 2
    # Score*V: B * Nq * S * S * v_dim * 2
    flops_sv = B * Nq * S * S * vd * 2
    total_flops = flops_qk + flops_sv

    # Softmax: vec ops = B * Nq * S * S * 4 (exp, sum, div, sub)
    vec_ops = B * Nq * S * S * 4

    # Flash attention memory: Q + K + V read, O write (no intermediate to HBM)
    q_bytes = bytes2(B * Nq * S * (Dqc + Dr))
    k_bytes = bytes2(B * 1 * S * kd)
    v_bytes = bytes2(B * 1 * S * vd)
    o_bytes = bytes2(B * Nq * S * Dqc)
    mem = q_bytes + k_bytes + v_bytes + o_bytes

    return roofline_time("attention_full", total_flops, vec_ops, mem, cfg.hw)


def op_attention_prefill_compressed(B: int, S: int, ratio: int, cfg: Config,
                                    use_index: bool = True) -> OpProfile:
    """Prefill attention for ratio>1 layers: compressed + SWA.
    If use_index=True (C4A), attends to topK selected entries.
    If use_index=False (C128A), attends to all S//ratio compressed entries."""
    TP = cfg.rt.tp
    Nq = cfg.model.num_attention_heads // TP
    c_k = cfg.model.compress_c_k
    c_v = cfg.model.compress_c_v
    kd = cfg.model.k_dim
    vd = cfg.model.v_dim
    Dqc = cfg.model.head_dim
    Dr = cfg.model.rope_head_dim
    W = cfg.model.window_size
    S_comp = S // ratio

    # Number of compressed KV entries attended to per query
    n_attend = cfg.model.index_topk if use_index else S_comp

    # Compressed attention
    flops_comp_qk = B * Nq * S * n_attend * c_k * 2
    flops_comp_sv = B * Nq * S * n_attend * c_v * 2
    vec_comp = B * Nq * S * n_attend * 4

    # SWA attention
    flops_swa_qk = B * Nq * S * W * kd * 2
    flops_swa_sv = B * Nq * S * W * vd * 2
    vec_swa = B * Nq * S * W * 4

    total_flops = flops_comp_qk + flops_comp_sv + flops_swa_qk + flops_swa_sv
    total_vec = vec_comp + vec_swa

    # Flash attention memory model
    q_bytes = bytes2(B * Nq * S * (Dqc + Dr))
    o_bytes = bytes2(B * Nq * S * Dqc)
    # Read full compressed KV cache (S//ratio entries) + SWA window (prefill read full, decode read n_attend entries)
    comp_kv_read = bytes2(B * S_comp * c_k) + bytes2(B * S_comp * c_v)
    swa_kv_read = bytes2(B * W * kd) + bytes2(B * W * vd)
    mem = q_bytes + comp_kv_read + swa_kv_read + o_bytes

    return roofline_time("attention_comp", total_flops, total_vec, mem, cfg.hw)


def op_attention_decode_full(B: int, S_total: int, cfg: Config) -> OpProfile:
    """Decode attention for ratio=1 layers. S_query=1, S_kv=S_total."""
    TP = cfg.rt.tp
    Nq = cfg.model.num_attention_heads // TP
    kd = cfg.model.k_dim
    vd = cfg.model.v_dim
    Dqc = cfg.model.head_dim
    Dr = cfg.model.rope_head_dim

    flops_qk = B * Nq * 1 * S_total * kd * 2
    flops_sv = B * Nq * 1 * S_total * vd * 2
    total_flops = flops_qk + flops_sv

    vec_ops = B * Nq * S_total * 4  # softmax

    # Memory: KV cache read dominates
    q_bytes = bytes2(B * Nq * 1 * (Dqc + Dr))
    kv_cache_bytes = bytes2(B * S_total * kd) + bytes2(B * S_total * vd)
    o_bytes = bytes2(B * Nq * 1 * Dqc)
    mem = q_bytes + kv_cache_bytes + o_bytes

    return roofline_time("attention_full", total_flops, vec_ops, mem, cfg.hw)


def op_attention_decode_compressed(B: int, S_total: int, ratio: int, cfg: Config,
                                   use_index: bool = True) -> OpProfile:
    """Decode attention for ratio>1 layers: compressed + SWA.
    If use_index=True (C4A), attends to topK selected entries.
    If use_index=False (C128A), attends to all S_total//ratio compressed entries."""
    TP = cfg.rt.tp
    Nq = cfg.model.num_attention_heads // TP
    c_k = cfg.model.compress_c_k
    c_v = cfg.model.compress_c_v
    kd = cfg.model.k_dim
    vd = cfg.model.v_dim
    Dqc = cfg.model.head_dim
    Dr = cfg.model.rope_head_dim
    W = cfg.model.window_size
    S_comp = S_total // ratio

    # Number of compressed KV entries attended to
    n_attend = cfg.model.index_topk if use_index else S_comp

    # Compressed attention: 1 query against n_attend entries
    flops_comp_qk = B * Nq * 1 * n_attend * c_k * 2
    flops_comp_sv = B * Nq * 1 * n_attend * c_v * 2
    vec_comp = B * Nq * n_attend * 4

    # SWA: 1 query against window
    flops_swa_qk = B * Nq * 1 * W * kd * 2
    flops_swa_sv = B * Nq * 1 * W * vd * 2
    vec_swa = B * Nq * W * 4

    total_flops = flops_comp_qk + flops_comp_sv + flops_swa_qk + flops_swa_sv
    total_vec = vec_comp + vec_swa

    # Memory reads:
    q_bytes = bytes2(B * Nq * 1 * (Dqc + Dr))
    # Compressed KV (selected topK or all S_comp entries)
    comp_kv = bytes2(B * n_attend * c_k) + bytes2(B * n_attend * c_v)
    # SWA window
    swa_kv = bytes2(B * W * kd) + bytes2(B * W * vd)
    o_bytes = bytes2(B * Nq * 1 * Dqc)
    mem = q_bytes + comp_kv + swa_kv + o_bytes

    return roofline_time("attention_comp", total_flops, total_vec, mem, cfg.hw)


# --- mHC (Hyper Connection) ---

def op_mhc_pre(T: int, cfg: Config, label: str = "mhc_pre") -> OpProfile:
    """mHC pre (UNFUSED baseline): linear projections before sub-layer.
    T is effective token count (B*S/TP if SP, else B*S).
    n = hc_mult, D = hidden_size.
    FP32 throughout (×4 bytes per element).

    Without kernel fusion, each intermediate step materializes its result
    to HBM. The dominant memory term T*(n²+2n)*n*D captures these
    unfused intermediate tensors — the main bottleneck."""
    n = cfg.model.hc_mult
    D = cfg.model.hidden_size
    cube_flops = 2 * T * (n**2 + 2*n) * n * D + 5 * T * n + 2 * T * n**2
    vec_ops = 2 * T * n
    # FP32 memory (×4): input/output activations + intermediate
    mem_bytes = 4 * (3 * T * n * D + T * (n**2 + 2*n) * n * D + T * (n**2 + 2*n))
    return roofline_time(label, cube_flops, vec_ops, mem_bytes, cfg.hw)


def op_mhc_sinkhorn(T: int, cfg: Config, label: str = "sinkhorn") -> OpProfile:
    """Sinkhorn normalization step (UNFUSED baseline).
    T is effective token count, n = hc_mult.
    FP32 throughout (×4 bytes per element).

    Without fusion, each of the 20 Sinkhorn iterations reads/writes the
    n×n matrix from/to HBM per token. With fusion, the n×n matrix lives
    in registers (only 16 FP32 values for n=4)."""
    n = cfg.model.hc_mult
    cube_flops = 0
    vec_ops = T * n**2 + 40 * T * n * (2*n - 1)
    # FP32 memory:
    mem_bytes = 4 * (2 * T * n**2 + 20 * (2 * T * n**2 + 2 * T * n**2))
    return roofline_time(label, cube_flops, vec_ops, mem_bytes, cfg.hw)


def op_mhc_post(T: int, cfg: Config, label: str = "mhc_post") -> OpProfile:
    """mHC post (UNFUSED baseline): linear projections after sub-layer.
    T is effective token count, n = hc_mult, D = hidden_size.
    FP32 throughout (×4 bytes per element)."""
    n = cfg.model.hc_mult
    D = cfg.model.hidden_size
    cube_flops = 2 * T * n**2 * D + 3 * T * n * D
    vec_ops = 0
    # FP32 memory
    mem_bytes = 4 * (T * n + T * D + 6 * T * n * D + T * n**2)
    return roofline_time(label, cube_flops, vec_ops, mem_bytes, cfg.hw)


# ── Fused mHC ops (kernel fusion) ──────────────────────────────────────────
#
# Background (mHC paper, arXiv:2512.24880):
#
#   The mHC layer update is:
#     x_{l+1} = H_res_l · x_l  +  H_post_l ⊗ F(H_pre_l · x_l, W_l)
#
#   where  x_l ∈ [T, n, D]   — expanded hidden state (n parallel streams)
#          H_res ∈ [n, n]     — doubly stochastic residual mixing matrix
#          H_pre ∈ [n]        — non-negative aggregation weights (softmax)
#          H_post ∈ [n]       — non-negative distribution weights (softmax)
#          F()                — sub-layer function (attention or MoE)
#
#   H_res is constrained to the Birkhoff polytope (doubly stochastic) via
#   Sinkhorn-Knopp: 20 iterations of alternating row/column normalization.
#   This ensures spectral norm ≤ 1, preventing signal amplification.
#
# Why fusion matters:
#
#   The unfused mHC ops have arithmetic intensity (AI) of only ~0.4 FLOP/byte,
#   vs ~745 FLOP/byte for standard matmuls like Q/K/V projections. This means
#   mHC is entirely MEM-bound: the bottleneck is HBM traffic, not compute.
#
#   Root cause: without fusion, intermediate tensors of shape [T, (n²+2n), n*D]
#   are written to HBM between every elementwise step, then immediately re-read.
#   These intermediates are ~21× larger than the actual input+output.
#
#   With kernel fusion, ALL intermediates stay in on-chip SRAM/registers:
#   - H_res [n,n] = 16 FP32 values (for n=4) → fits in registers
#   - H_pre [n] = 4 values → fits in registers
#   - Sinkhorn iterations operate on 4×4 matrix in registers
#   - Per-token computation: small [n,n]×[n,D] matmul in SRAM
#
#   Only the true inputs and outputs traverse HBM.
#
# Memory reduction (n=4, D=4096, T=8192, comparing one mhc_pre call):
#
#   Unfused FP32:  14.5 GB  (AI = 0.4 FLOP/byte)  → 10.07 ms on 910C
#   Fused   FP32:   1.2 GB  (AI = 5.3 FLOP/byte)  →  0.84 ms (12× faster)
#   Fused   BF16:   0.6 GB  (AI = 10.6 FLOP/byte) →  0.42 ms (24× faster)
#
# ────────────────────────────────────────────────────────────────────────────

def op_mhc_pre_fused(T: int, cfg: Config, label: str = "mhc_pre") -> OpProfile:
    """Fused mHC pre + Sinkhorn: single kernel for residual mixing + aggregation.

    Fuses three previously separate ops into one kernel:
      1. Sinkhorn normalization of H_res  (was: op_mhc_sinkhorn)
      2. Residual mixing: H_res @ x_l     (was: part of op_mhc_pre)
      3. Input aggregation: H_pre · x_l   (was: part of op_mhc_pre)

    Kernel pseudocode (one GPU kernel launch):
    ──────────────────────────────────────────
      // Phase 1: Sinkhorn in registers (shared across all token tiles)
      H = exp(H_res_logits)                         // [n, n] in registers
      for iter in range(20):                         // 20 Sinkhorn iterations
          H[i,j] /= sum_j(H[i,:])   for all i      // row normalize
          H[i,j] /= sum_i(H[:,j])   for all j      // col normalize
      h_pre = softmax(H_pre_logits)                  // [n] in registers

      // Phase 2: stream through x_l in tiles
      for tile in tiles(T):
          x_tile = load(x_l[tile, :, :])             // [tile_sz, n, D] from HBM
          // Residual mixing: [n,n] @ [tile_sz, n, D] → [tile_sz, n, D]
          residual_tile = einsum('ij, tjd -> tid', H, x_tile)   // in SRAM
          // Input aggregation: [n] · [tile_sz, n, D] → [tile_sz, D]
          sub_input_tile = einsum('i, tid -> td', h_pre, x_tile) // in SRAM
          store(residual[tile, :, :], residual_tile)  // to HBM
          store(sub_input[tile, :], sub_input_tile)   // to HBM

    HBM traffic:
      Read:  x_l [T, n, D]              → bpe × T × n × D
      Read:  H_res_logits [n, n]         → 4 × n² (negligible)
      Read:  H_pre_logits [n]            → 4 × n  (negligible)
      Write: residual [T, n, D]          → bpe × T × n × D
      Write: sub_input [T, D]            → bpe × T × D
      ─────────────────────────────────────────────────
      Total ≈ bpe × T × D × (2n + 1)

    where bpe = 2 (BF16, inference-safe) or 4 (FP32).

    Compared to unfused: 4 × T × (n²+2n) × n × D  (≈ 21× more for n=4, FP32)

    FLOPs unchanged — same computation, just better memory access pattern.
    """
    n = cfg.model.hc_mult
    D = cfg.model.hidden_size

    # FLOPs: identical to unfused pre + sinkhorn combined
    # pre:      2*T*(n²+2n)*n*D + 5*T*n + 2*T*n²  (H_res matmul + H_pre dot)
    # sinkhorn: T*n² + 40*T*n*(2n-1)               (20 iters × row/col norm)
    cube_flops = 2 * T * (n**2 + 2*n) * n * D + 5 * T * n + 2 * T * n**2
    vec_ops = T * n**2 + 40 * T * n * (2*n - 1) + 2 * T * n

    # Fused HBM traffic: ONLY input read + output writes
    # BF16 is safe for inference because H_res is doubly stochastic
    # (spectral norm ≤ 1, no signal amplification without gradients)
    bpe = 2 if cfg.rt.mhc_fused_bf16 else 4
    mem_bytes = (
        bpe * T * n * D           # read x_l [T, n, D]
        + bpe * T * n * D         # write residual [T, n, D]
        + bpe * T * D             # write sub_input [T, D]
        + 4 * (n * n + n)         # read weights (FP32, negligible)
    )
    return roofline_time(label, cube_flops, vec_ops, mem_bytes, cfg.hw)


def op_mhc_post_fused(T: int, cfg: Config, label: str = "mhc_post") -> OpProfile:
    """Fused mHC post: distribute sub-layer output back to n streams + residual add.

    Mathematical operation:
      x_{l+1}[t, i, :] = residual[t, i, :] + H_post[i] × sub_output[t, :]

    where H_post = softmax(H_post_logits) ∈ [n], non-negative, sums to 1.
    This scatters the sub-layer's [T, D] output into the [T, n, D] stream,
    weighted by H_post, then adds the stored residual from mhc_pre.

    Kernel pseudocode:
    ──────────────────────────────────────────
      h_post = softmax(H_post_logits)                // [n] in registers
      for tile in tiles(T):
          res_tile = load(residual[tile, :, :])       // [tile_sz, n, D] from HBM
          y_tile = load(sub_output[tile, :])           // [tile_sz, D] from HBM
          // Broadcast multiply + add:
          // x_next[t, i, d] = res[t, i, d] + h_post[i] * y[t, d]
          x_next_tile = res_tile + outer(h_post, y_tile)  // in SRAM
          store(x_next[tile, :, :], x_next_tile)      // to HBM

    HBM traffic:
      Read:  residual [T, n, D]          → bpe × T × n × D
      Read:  sub_output [T, D]           → bpe × T × D
      Read:  H_post_logits [n]           → 4 × n  (negligible)
      Write: x_{l+1} [T, n, D]          → bpe × T × n × D
      ─────────────────────────────────────────────────
      Total ≈ bpe × T × D × (2n + 1)

    Compared to unfused: 4 × (T*n + T*D + 6*T*n*D + T*n²)  (≈ 2.8× more)
    """
    n = cfg.model.hc_mult
    D = cfg.model.hidden_size

    # FLOPs: same as unfused
    cube_flops = 2 * T * n**2 * D + 3 * T * n * D
    vec_ops = 0

    # Fused HBM traffic
    bpe = 2 if cfg.rt.mhc_fused_bf16 else 4
    mem_bytes = (
        bpe * T * n * D           # read residual [T, n, D]
        + bpe * T * D             # read sub_output [T, D]
        + bpe * T * n * D         # write x_{l+1} [T, n, D]
        + 4 * n                   # read H_post logits (FP32, negligible)
    )
    return roofline_time(label, cube_flops, vec_ops, mem_bytes, cfg.hw)


def op_mhc_post_pre_fused(T: int, cfg: Config, label: str = "mhc_post_pre") -> OpProfile:
    """Deep-fused mHC: post(sublayer_k) + sinkhorn(k+1) + pre(sublayer_k+1).

    This fuses across the sublayer boundary (e.g. post_attn → pre_moe),
    eliminating the intermediate x_{l+0.5} from HBM entirely.

    In the standard (unfused) flow within one layer:
      ... → F_attn() → mhc_post_attn → mhc_pre_moe → RMSNorm → F_moe() → ...
                           ↓   writes x_{l+0.5}   ↑ reads x_{l+0.5}

    With deep fusion, x_{l+0.5} stays in SRAM:
      ... → F_attn() → [mhc_post_attn + mhc_pre_moe fused] → RMSNorm → F_moe() → ...

    Kernel pseudocode:
    ──────────────────────────────────────────
      // Load all weights into registers
      h_post_k   = softmax(H_post_logits_k)               // [n]
      H_res_k1   = sinkhorn(exp(H_res_logits_{k+1}), 20)  // [n, n]
      h_pre_k1   = softmax(H_pre_logits_{k+1})             // [n]

      for tile in tiles(T):
          res_k     = load(residual_k[tile])                // [tile_sz, n, D]
          y_k       = load(sub_output_k[tile])              // [tile_sz, D]

          // ── Post step (sublayer k) ──
          // x_mid = res_k + h_post_k ⊗ y_k                // [tile_sz, n, D]
          x_mid = res_k + outer(h_post_k, y_k)             // stays in SRAM!

          // ── Pre step (sublayer k+1) ──
          // residual_{k+1} = H_res_{k+1} @ x_mid
          res_k1 = einsum('ij, tjd -> tid', H_res_k1, x_mid)   // in SRAM
          // sub_input_{k+1} = h_pre_{k+1} · x_mid
          sub_k1 = einsum('i, tid -> td', h_pre_k1, x_mid)     // in SRAM

          store(residual_{k+1}[tile], res_k1)               // to HBM
          store(sub_input_{k+1}[tile], sub_k1)               // to HBM

    HBM traffic:
      Read:  residual_k [T, n, D]            → bpe × T × n × D
      Read:  sub_output_k [T, D]             → bpe × T × D
      Read:  weights (3 sets)                → 4 × (n² + 2n)  (negligible)
      Write: residual_{k+1} [T, n, D]       → bpe × T × n × D
      Write: sub_input_{k+1} [T, D]         → bpe × T × D
      ─────────────────────────────────────────────────
      Total ≈ bpe × T × D × (2n + 2)

    vs. separate fused post + fused pre = 2 × bpe × T × D × (2n + 1)
    Savings: eliminates bpe × T × n × D × 2 bytes (read+write of x_{l+0.5})
    """
    n = cfg.model.hc_mult
    D = cfg.model.hidden_size

    # FLOPs: sum of post + sinkhorn + pre
    cube_flops = (
        (2 * T * n**2 * D + 3 * T * n * D)                   # post
        + (2 * T * (n**2 + 2*n) * n * D + 5 * T * n + 2 * T * n**2)  # pre
    )
    vec_ops = T * n**2 + 40 * T * n * (2*n - 1) + 2 * T * n  # sinkhorn + pre

    # Deep-fused HBM traffic: x_{l+0.5} never written to HBM
    bpe = 2 if cfg.rt.mhc_fused_bf16 else 4
    mem_bytes = (
        bpe * T * n * D           # read residual_k [T, n, D]
        + bpe * T * D             # read sub_output_k [T, D]
        + bpe * T * n * D         # write residual_{k+1} [T, n, D]
        + bpe * T * D             # write sub_input_{k+1} [T, D]
        + 4 * (n * n + 2 * n)    # read all weights (FP32, negligible)
    )
    return roofline_time(label, cube_flops, vec_ops, mem_bytes, cfg.hw)


# --- MoE ---

def op_moe_gate(T: int, cfg: Config) -> OpProfile:
    """Gate projection: [H, n_routed_experts]."""
    H = cfg.model.hidden_size
    n_exp = cfg.model.n_routed_experts
    flops = T * H * n_exp * 2
    weight_bytes = bytes2(H * n_exp)
    act_in = bytes2(T * H)
    act_out = bytes2(T * n_exp)
    return roofline_time("moe_gate", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


def op_moe_ep_dispatch(T: int, layer_idx: int, cfg: Config) -> OpProfile:
    """EP dispatch AllToAll: (EP-1)/EP * T * top_k * H * load_factor * 2 bytes.
    Volume scaled by load_factor to model bottleneck rank receiving more tokens."""
    EP = cfg.rt.ep
    H = cfg.model.hidden_size
    top_k = cfg.model.num_experts_per_tok
    n_hash = cfg.model.n_hash_layers
    load_factor = 1.0 if layer_idx < n_hash else cfg.rt.moe_load_balance_factor
    vol = bytes2(T * top_k * H * load_factor)
    t = alltoall_time(vol, EP, cfg.net.ep_bandwidth_gbps,
                      cfg.net.latency_us, cfg.net.bandwidth_utilization)
    return OpProfile(name="moe_ep_dispatch", comm_bytes=vol, comm_time_s=t,
                     time_s=t, bottleneck="COMM" if t > 0 else "")


def op_moe_ep_combine(T: int, layer_idx: int, cfg: Config) -> OpProfile:
    """EP combine AllToAll: same volume as dispatch (scaled by load_factor)."""
    EP = cfg.rt.ep
    H = cfg.model.hidden_size
    top_k = cfg.model.num_experts_per_tok
    n_hash = cfg.model.n_hash_layers
    load_factor = 1.0 if layer_idx < n_hash else cfg.rt.moe_load_balance_factor
    vol = bytes2(T * top_k * H * load_factor)
    t = alltoall_time(vol, EP, cfg.net.ep_bandwidth_gbps,
                      cfg.net.latency_us, cfg.net.bandwidth_utilization)
    return OpProfile(name="moe_ep_combine", comm_bytes=vol, comm_time_s=t,
                     time_s=t, bottleneck="COMM" if t > 0 else "")


def op_moe_routed_experts(T: int, layer_idx: int, cfg: Config) -> List[OpProfile]:
    """Routed experts: gate_proj, up_proj, SiLU+mul, down_proj.
    Per-rank tokens = T * top_k * load_factor / EP.
    Returns list of sub-ops."""
    EP = cfg.rt.ep
    H = cfg.model.hidden_size
    inter = cfg.model.moe_inter_dim
    top_k = cfg.model.num_experts_per_tok
    n_hash = cfg.model.n_hash_layers

    if layer_idx < n_hash:
        load_factor = 1.0
    else:
        load_factor = cfg.rt.moe_load_balance_factor

    # Tokens hitting this rank's experts (bottleneck rank)
    T_rank = T * top_k * load_factor / EP

    # gate_proj: [H, inter]
    flops_gate = T_rank * H * inter * 2
    w_gate = bytes2(H * inter * (cfg.model.n_routed_experts // EP))
    act_in_gate = bytes2(T_rank * H)
    act_out_gate = bytes2(T_rank * inter)
    gate_op = roofline_time("routed_gate_proj", flops_gate, 0,
                            w_gate + act_in_gate + act_out_gate, cfg.hw)

    # up_proj: [H, inter]
    flops_up = T_rank * H * inter * 2
    w_up = bytes2(H * inter * (cfg.model.n_routed_experts // EP))
    act_in_up = bytes2(T_rank * H)
    act_out_up = bytes2(T_rank * inter)
    up_op = roofline_time("routed_up_proj", flops_up, 0,
                          w_up + act_in_up + act_out_up, cfg.hw)

    # SiLU + element-wise multiply (vector ops)
    vec_silu = T_rank * inter * 2  # SiLU: x * sigmoid(x) ~ 2 ops
    vec_mul = T_rank * inter       # element-wise multiply
    total_vec = vec_silu + vec_mul
    act_silu_mem = bytes2(T_rank * inter) * 2 + bytes2(T_rank * inter)  # read gate+up, write
    silu_op = roofline_time("routed_silu_mul", 0, total_vec, act_silu_mem, cfg.hw)

    # down_proj: [inter, H]
    flops_down = T_rank * inter * H * 2
    w_down = bytes2(inter * H * (cfg.model.n_routed_experts // EP))
    act_in_down = bytes2(T_rank * inter)
    act_out_down = bytes2(T_rank * H)
    down_op = roofline_time("routed_down_proj", flops_down, 0,
                            w_down + act_in_down + act_out_down, cfg.hw)

    return [gate_op, up_op, silu_op, down_op]


def op_moe_shared_expert(T: int, cfg: Config) -> List[OpProfile]:
    """Shared expert (replicated): gate_proj, up_proj, SiLU+mul, down_proj."""
    H = cfg.model.hidden_size
    inter = cfg.model.moe_inter_dim

    # gate_proj: [H, inter]
    flops_gate = T * H * inter * 2
    w_gate = bytes2(H * inter)
    act_in_gate = bytes2(T * H)
    act_out_gate = bytes2(T * inter)
    gate_op = roofline_time("shared_gate_proj", flops_gate, 0,
                            w_gate + act_in_gate + act_out_gate, cfg.hw)

    # up_proj: [H, inter]
    flops_up = T * H * inter * 2
    w_up = bytes2(H * inter)
    act_in_up = bytes2(T * H)
    act_out_up = bytes2(T * inter)
    up_op = roofline_time("shared_up_proj", flops_up, 0,
                          w_up + act_in_up + act_out_up, cfg.hw)

    # SiLU + element-wise multiply (vector ops)
    vec_silu = T * inter * 2  # SiLU: x * sigmoid(x) ~ 2 ops
    vec_mul = T * inter       # element-wise multiply
    total_vec = vec_silu + vec_mul
    act_silu_mem = bytes2(T * inter) * 2 + bytes2(T * inter)  # read gate+up, write
    silu_op = roofline_time("shared_silu_mul", 0, total_vec, act_silu_mem, cfg.hw)

    # down_proj: [inter, H]
    flops_down = T * inter * H * 2
    w_down = bytes2(inter * H)
    act_in_down = bytes2(T * inter)
    act_out_down = bytes2(T * H)
    down_op = roofline_time("shared_down_proj", flops_down, 0,
                            w_down + act_in_down + act_out_down, cfg.hw)

    return [gate_op, up_op, silu_op, down_op]


# --- RMSNorm, Embedding, LM Head ---

def op_rmsnorm(T: int, cfg: Config, label: str = "rmsnorm") -> OpProfile:
    """RMSNorm: ~3H vec ops per token, read/write [T, H]."""
    H = cfg.model.hidden_size
    vec_ops = T * H * 3  # square, mean, divide
    mem = bytes2(T * H) * 2  # read + write
    return roofline_time(label, 0, vec_ops, mem, cfg.hw)


def op_embedding(B: int, S: int, cfg: Config) -> OpProfile:
    """Embedding lookup: no FLOPs, output B*S*H*2 bytes."""
    H = cfg.model.hidden_size
    mem = bytes2(B * S * H)
    return roofline_time("embedding", 0, 0, mem, cfg.hw)


def op_lm_head(T: int, cfg: Config) -> OpProfile:
    """LM Head: [H, vocab/TP] ColumnParallel."""
    H = cfg.model.hidden_size
    V = cfg.model.vocab_size
    TP = cfg.rt.tp
    v_per_rank = V // TP
    flops = T * H * v_per_rank * 2
    weight_bytes = bytes2(H * v_per_rank)
    act_in = bytes2(T * H)
    act_out = bytes2(T * v_per_rank)
    return roofline_time("lm_head", flops, 0, weight_bytes + act_in + act_out, cfg.hw)


# --- SP AllGather ---

def op_sp_allgather(T_sp: int, cfg: Config, label: str = "sp_allgather") -> OpProfile:
    """SP AllGather: collect T_sp -> T_full along sequence dim.
    Volume = T_full * H * 2 bytes (full output each rank receives).
    The (n-1)/n factor in allgather_time handles actual transfer fraction."""
    H = cfg.model.hidden_size
    TP = cfg.rt.tp
    if not cfg.rt.sp or TP <= 1:
        return OpProfile(name=label)
    T_full = T_sp * TP
    vol = bytes2(T_full * H)
    t = allgather_time(vol, TP, cfg.net.tp_bandwidth_gbps,
                       cfg.net.latency_us, cfg.net.bandwidth_utilization)
    return OpProfile(name=label, comm_bytes=vol, comm_time_s=t,
                     time_s=t, bottleneck="COMM" if t > 0 else "")
