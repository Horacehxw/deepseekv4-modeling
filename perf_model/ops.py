"""All per-operation cost functions (~30 ops)."""

from typing import List

from .config import Config
from .roofline import OpProfile, roofline_time, allreduce_time, alltoall_time, bytes2


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
    """W_uq: [q_lora_rank, Nq/TP * (Dqc + Dr)]."""
    qlr = cfg.model.q_lora_rank
    Nq = cfg.model.num_attention_heads
    Dqc = cfg.model.head_dim
    Dr = cfg.model.rope_head_dim
    TP = cfg.rt.tp
    out_dim = (Nq // TP) * (Dqc + Dr)
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
    H = cfg.model.hidden_size
    TP = cfg.rt.tp
    nh = cfg.model.index_n_heads
    hd = cfg.model.index_head_dim
    out_dim = (nh // TP) * hd
    flops = T * H * out_dim * 2
    weight_bytes = bytes2(H * out_dim)
    act_in = bytes2(T * H)
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


def op_index_kv_compression(B: int, S: int, ratio: int, cfg: Config) -> OpProfile:
    """PLACEHOLDER: KV compression compute for index keys. Returns zero."""
    return OpProfile(name="index_kv_compress")


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
    """PLACEHOLDER: returns zero. User fills in compression algorithm costs."""
    return OpProfile(name="kv_compression")


def op_kv_compression_decode(B: int, ratio: int, cfg: Config) -> OpProfile:
    """PLACEHOLDER: amortized per-step cost. User fills in."""
    return OpProfile(name="kv_compression_decode")


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
    # Read full compressed KV cache (S//ratio entries) + SWA window
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
    """mHC pre: linear projections before sub-layer.
    T is effective token count (B*S/TP if SP, else B*S).
    n = hc_mult, D = hidden_size.
    FP32 throughout (×4 bytes per element)."""
    n = cfg.model.hc_mult
    D = cfg.model.hidden_size
    cube_flops = 2 * T * (n**2 + 2*n) * n * D + 5 * T * n + 2 * T * n**2
    vec_ops = 2 * T * n
    # FP32 memory (×4): input/output activations + intermediate
    mem_bytes = 4 * (3 * T * n * D + T * (n**2 + 2*n) * n * D + T * (n**2 + 2*n))
    return roofline_time(label, cube_flops, vec_ops, mem_bytes, cfg.hw)


def op_mhc_sinkhorn(T: int, cfg: Config, label: str = "sinkhorn") -> OpProfile:
    """Sinkhorn normalization step.
    T is effective token count, n = hc_mult.
    FP32 throughout (×4 bytes per element)."""
    n = cfg.model.hc_mult
    cube_flops = 0
    vec_ops = T * n**2 + 40 * T * n * (2*n - 1)
    # FP32 memory: 82 * T * n^2 * 4
    mem_bytes = 4 * 82 * T * n**2
    return roofline_time(label, cube_flops, vec_ops, mem_bytes, cfg.hw)


def op_mhc_post(T: int, cfg: Config, label: str = "mhc_post") -> OpProfile:
    """mHC post: linear projections after sub-layer.
    T is effective token count, n = hc_mult, D = hidden_size.
    FP32 throughout (×4 bytes per element)."""
    n = cfg.model.hc_mult
    D = cfg.model.hidden_size
    cube_flops = 2 * T * n**2 * D + 3 * T * n * D
    vec_ops = 0
    # FP32 memory
    mem_bytes = 4 * (T * n + T * D + 6 * T * n * D + T * n**2)
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

    flops_gate = T * H * inter * 2
    w_gate = bytes2(H * inter)
    gate_op = roofline_time("shared_gate_proj", flops_gate, 0,
                            w_gate + bytes2(T * H) + bytes2(T * inter), cfg.hw)

    flops_up = T * H * inter * 2
    w_up = bytes2(H * inter)
    up_op = roofline_time("shared_up_proj", flops_up, 0,
                          w_up + bytes2(T * H) + bytes2(T * inter), cfg.hw)

    vec_silu = T * inter * 2
    vec_mul = T * inter
    silu_op = roofline_time("shared_silu_mul", 0, vec_silu + vec_mul,
                            bytes2(T * inter) * 2 + bytes2(T * inter), cfg.hw)

    flops_down = T * inter * H * 2
    w_down = bytes2(inter * H)
    down_op = roofline_time("shared_down_proj", flops_down, 0,
                            w_down + bytes2(T * inter) + bytes2(T * H), cfg.hw)

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
