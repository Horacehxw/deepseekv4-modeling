"""LayerProfile, PhaseProfile, and layer/phase aggregation."""

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config
from .roofline import OpProfile, sum_ops
from .ops import (
    op_q_proj_dq, op_q_proj_uq, op_kv_proj, op_wo_a, op_wo_b,
    op_attn_tp_allreduce,
    op_index_iq_proj, op_index_kv_compression_prefill, op_index_kv_compression_decode,
    op_index_score, op_index_score_allreduce,
    op_index_score_decode, op_index_score_allreduce_decode,
    op_kv_compression_prefill, op_kv_compression_decode,
    op_attention_prefill_swa, op_attention_prefill_compressed,
    op_attention_decode_swa, op_attention_decode_compressed,
    op_mhc_pre, op_mhc_sinkhorn, op_mhc_post,
    op_mhc_pre_fused, op_mhc_post_fused, op_mhc_post_pre_fused,
    op_moe_gate, op_moe_ep_dispatch, op_moe_ep_combine,
    op_moe_routed_experts, op_moe_shared_expert,
    op_rmsnorm, op_embedding, op_lm_head,
    op_sp_allgather,
)


@dataclass
class LayerProfile:
    layer_idx: int
    ratio: int
    ops: List[OpProfile] = field(default_factory=list)
    total: Optional[OpProfile] = None

    def compute_total(self):
        self.total = sum_ops(self.ops, f"layer_{self.layer_idx}")


def prefill_layer(layer_idx: int, cfg: Config) -> LayerProfile:
    """Compute prefill cost for a single layer."""
    B = cfg.rt.batch_size // cfg.rt.dp
    S = cfg.rt.seq_len
    TP = cfg.rt.tp
    ratio = cfg.model.compress_ratios[layer_idx]
    T_full = B * S
    T_sp = B * S // TP if cfg.rt.sp else T_full
    T_mhc = T_sp if (cfg.rt.sp and cfg.rt.mhc_sp) else T_full
    fused = cfg.rt.mhc_kernel_fused

    ops = []

    # ── mHC pre-attention ──
    # Unfused: mhc_pre + (later) sinkhorn_attn + mhc_post_attn
    # Fused:   mhc_pre_fused absorbs sinkhorn; post handled later
    if fused:
        ops.append(op_mhc_pre_fused(T_mhc, cfg, "mhc_pre_attn"))
    else:
        ops.append(op_mhc_pre(T_mhc, cfg, "mhc_pre_attn"))

    # RMSNorm (in SP region)
    ops.append(op_rmsnorm(T_sp, cfg, "rmsnorm_attn"))

    # SP AllGather: T_sp -> T_full before attention projections
    ops.append(op_sp_allgather(T_sp, cfg, "sp_ag_before_attn"))

    # Q/KV projections (T_full tokens, weights split by TP for Q; K=V shared)
    ops.append(op_q_proj_dq(T_full, cfg))
    ops.append(op_q_proj_uq(T_full, cfg))
    ops.append(op_kv_proj(T_full, cfg))

    # Determine if this layer uses Lightning Index
    # Index is needed when compressed seq len > topK (e.g. C4A: S//4=1024 > 512)
    # Not needed when compressed seq is already small (e.g. C128A: S//128=32 < 512)
    use_index = (ratio == 4)
    # S_comp = S // ratio if ratio > 1 else S
    # use_index = ratio > 1 and S_comp > cfg.model.index_topk

    # Index + Compression (only for layers that use Lightning Index)
    if use_index:
        ops.append(op_index_iq_proj(T_full, cfg))
        ops.append(op_index_kv_compression_prefill(B, S, ratio, cfg))
        ops.append(op_index_score(B, S, ratio, cfg))
        ops.append(op_index_score_allreduce(B, S, ratio, cfg))

    # KV Compression (for all compressed layers)
    if ratio > 1:
        ops.append(op_kv_compression_prefill(B, S, ratio, cfg))

    # Attention
    ops.append(op_attention_prefill_swa(B, S, cfg))
    if ratio > 1:
        ops.append(op_attention_prefill_compressed(B, S, ratio, cfg, use_index=use_index))

    # Output projection
    ops.append(op_wo_a(T_full, cfg))
    ops.append(op_wo_b(T_full, cfg))

    # TP AllReduce (or AG+RS for SP, same volume)
    ops.append(op_attn_tp_allreduce(T_full, cfg))

    # ── mHC post-attention → pre-MoE ──
    # Unfused: sinkhorn_attn + mhc_post_attn + mhc_pre_moe (3 separate ops)
    # Fused:   deep-fused post_pre replaces all 3 — x_{l+0.5} stays on-chip
    if fused:
        ops.append(op_mhc_post_pre_fused(T_mhc, cfg, "mhc_post_attn_pre_moe"))
    else:
        ops.append(op_mhc_sinkhorn(T_mhc, cfg, "sinkhorn_attn"))
        ops.append(op_mhc_post(T_mhc, cfg, "mhc_post_attn"))
        ops.append(op_mhc_pre(T_mhc, cfg, "mhc_pre_moe"))

    # RMSNorm (in SP region)
    ops.append(op_rmsnorm(T_sp, cfg, "rmsnorm_moe"))

    # SP AllGather: T_sp -> T_full before MoE gate
    ops.append(op_sp_allgather(T_sp, cfg, "sp_ag_before_moe"))

    # MoE
    ops.append(op_moe_gate(T_full, cfg))
    ops.append(op_moe_ep_dispatch(T_sp, layer_idx, cfg))

    routed_ops = op_moe_routed_experts(T_sp, layer_idx, cfg)
    shared_ops = op_moe_shared_expert(T_sp, cfg)

    if cfg.rt.shared_expert_overlapped:
        # Shared expert overlaps with dispatch + combine (comm stays fixed; compute scales).
        # shared_time and routed_comm are already at the active quant speed, so in W8A8
        # mode the shared-expert GEMMs complete faster and may finish before the comm,
        # yielding zero excess — which is physically correct.
        routed_time = sum(op.time_s for op in routed_ops)
        shared_time = sum(op.time_s for op in shared_ops)
        dispatch_op = ops[-1]  # ep_dispatch
        ops.extend(routed_ops)
        ops.append(op_moe_ep_combine(T_sp, layer_idx, cfg))
        combine_time = ops[-1].time_s
        routed_comm = dispatch_op.time_s + combine_time
        if shared_time > routed_comm:
            excess = shared_time - routed_comm
            ops.append(OpProfile(name="shared_expert_excess", time_s=excess))
    else:
        ops.extend(routed_ops)
        ops.append(op_moe_ep_combine(T_sp, layer_idx, cfg))
        ops.extend(shared_ops)

    # SP AllGather: T_sp -> T_full after MoE combine
    ops.append(op_sp_allgather(T_sp, cfg, "sp_ag_after_moe"))

    # ── mHC post-MoE ──
    # Unfused: sinkhorn_moe + mhc_post_moe
    # Fused:   mhc_post_fused (sinkhorn was already in the pre_fused of next layer)
    if fused:
        ops.append(op_mhc_post_fused(T_mhc, cfg, "mhc_post_moe"))
    else:
        ops.append(op_mhc_sinkhorn(T_mhc, cfg, "sinkhorn_moe"))
        ops.append(op_mhc_post(T_mhc, cfg, "mhc_post_moe"))

    lp = LayerProfile(layer_idx=layer_idx, ratio=ratio, ops=ops)
    lp.compute_total()
    return lp


def decode_layer(layer_idx: int, S_total: int, cfg: Config) -> LayerProfile:
    """Compute decode cost for a single layer (1 token generation step)."""
    B = cfg.rt.batch_size // cfg.rt.dp
    TP = cfg.rt.tp
    ratio = cfg.model.compress_ratios[layer_idx]
    T_full = B * 1  # 1 token per sequence
    fused = cfg.rt.mhc_kernel_fused

    ops = []

    # mHC pre-attention
    if fused:
        ops.append(op_mhc_pre_fused(T_full, cfg, "mhc_pre_attn"))
    else:
        ops.append(op_mhc_pre(T_full, cfg, "mhc_pre_attn"))

    # RMSNorm
    ops.append(op_rmsnorm(T_full, cfg, "rmsnorm_attn"))

    # Q/KV projections (K=V shared)
    ops.append(op_q_proj_dq(T_full, cfg))
    ops.append(op_q_proj_uq(T_full, cfg))
    ops.append(op_kv_proj(T_full, cfg))

    # Determine if this layer uses Lightning Index
    S_comp = S_total // ratio if ratio > 1 else S_total
    use_index = ratio > 1 and S_comp > cfg.model.index_topk

    # Index (only for layers that use Lightning Index)
    if use_index:
        ops.append(op_index_iq_proj(T_full, cfg))
        ops.append(op_index_kv_compression_decode(B, S_total, ratio, cfg))
        ops.append(op_index_score_decode(B, S_total, ratio, cfg))
        ops.append(op_index_score_allreduce_decode(B, S_total, ratio, cfg))

    # KV Compression (for all compressed layers)
    if ratio > 1:
        ops.append(op_kv_compression_decode(B, S_total, ratio, cfg))

    # Attention
    ops.append(op_attention_decode_swa(B, S_total, cfg))
    if ratio > 1:
        ops.append(op_attention_decode_compressed(B, S_total, ratio, cfg, use_index=use_index))

    # Output projection
    ops.append(op_wo_a(T_full, cfg))
    ops.append(op_wo_b(T_full, cfg))

    # TP AllReduce
    ops.append(op_attn_tp_allreduce(T_full, cfg))

    # mHC post-attention → pre-MoE
    if fused:
        ops.append(op_mhc_post_pre_fused(T_full, cfg, "mhc_post_attn_pre_moe"))
    else:
        ops.append(op_mhc_sinkhorn(T_full, cfg, "sinkhorn_attn"))
        ops.append(op_mhc_post(T_full, cfg, "mhc_post_attn"))
        ops.append(op_mhc_pre(T_full, cfg, "mhc_pre_moe"))

    # RMSNorm
    ops.append(op_rmsnorm(T_full, cfg, "rmsnorm_moe"))

    # MoE
    ops.append(op_moe_gate(T_full, cfg))
    ops.append(op_moe_ep_dispatch(T_full, layer_idx, cfg))

    routed_ops = op_moe_routed_experts(T_full, layer_idx, cfg)
    shared_ops = op_moe_shared_expert(T_full, cfg)

    if cfg.rt.shared_expert_overlapped:
        routed_time = sum(op.time_s for op in routed_ops)
        shared_time = sum(op.time_s for op in shared_ops)
        dispatch_op = ops[-1]
        ops.extend(routed_ops)
        ops.append(op_moe_ep_combine(T_full, layer_idx, cfg))
        combine_time = ops[-1].time_s
        routed_total = dispatch_op.time_s + routed_time + combine_time
        if shared_time > routed_total:
            excess = shared_time - routed_total
            ops.append(OpProfile(name="shared_expert_excess", time_s=excess))
    else:
        ops.extend(routed_ops)
        ops.append(op_moe_ep_combine(T_full, layer_idx, cfg))
        ops.extend(shared_ops)

    # mHC post-MoE
    if fused:
        ops.append(op_mhc_post_fused(T_full, cfg, "mhc_post_moe"))
    else:
        ops.append(op_mhc_sinkhorn(T_full, cfg, "sinkhorn_moe"))
        ops.append(op_mhc_post(T_full, cfg, "mhc_post_moe"))

    lp = LayerProfile(layer_idx=layer_idx, ratio=ratio, ops=ops)
    lp.compute_total()
    return lp


# =============================================================================
# End-to-End
# =============================================================================

@dataclass
class PhaseProfile:
    phase: str
    layer_profiles: List[LayerProfile] = field(default_factory=list)
    extra_ops: List[OpProfile] = field(default_factory=list)
    total_time_s: float = 0.0
    total_tokens: int = 0


def prefill_model(cfg: Config) -> PhaseProfile:
    """Full prefill: Embedding + all layers + LM Head."""
    cfg = cfg.for_phase("prefill")
    B = cfg.rt.batch_size // cfg.rt.dp
    S = cfg.rt.seq_len
    T_full = B * S

    phase = PhaseProfile(phase="prefill", total_tokens=T_full)

    # Embedding
    emb = op_embedding(B, S, cfg)
    phase.extra_ops.append(emb)
    phase.total_time_s += emb.time_s

    # All layers
    for i in range(cfg.model.num_hidden_layers):
        lp = prefill_layer(i, cfg)
        phase.layer_profiles.append(lp)
        phase.total_time_s += lp.total.time_s

    # Final RMSNorm
    T_sp = B * S // cfg.rt.tp if cfg.rt.sp else T_full
    final_norm = op_rmsnorm(T_sp, cfg, "final_rmsnorm")
    phase.extra_ops.append(final_norm)
    phase.total_time_s += final_norm.time_s

    # SP AllGather: T_sp -> T_full before LM Head
    sp_ag = op_sp_allgather(T_sp, cfg, "sp_ag_before_lm_head")
    phase.extra_ops.append(sp_ag)
    phase.total_time_s += sp_ag.time_s

    # LM Head
    lm = op_lm_head(T_full, cfg)
    phase.extra_ops.append(lm)
    phase.total_time_s += lm.time_s

    return phase


def decode_step(S_total: int, cfg: Config) -> PhaseProfile:
    """Single decode step at context length S_total."""
    cfg = cfg.for_phase("decode")
    B = cfg.rt.batch_size // cfg.rt.dp
    T_full = B * 1

    phase = PhaseProfile(phase=f"decode_step@{S_total}", total_tokens=B)

    # Embedding (single token)
    emb = op_embedding(B, 1, cfg)
    phase.extra_ops.append(emb)
    phase.total_time_s += emb.time_s

    # All layers
    for i in range(cfg.model.num_hidden_layers):
        lp = decode_layer(i, S_total, cfg)
        phase.layer_profiles.append(lp)
        phase.total_time_s += lp.total.time_s

    # Final RMSNorm
    final_norm = op_rmsnorm(T_full, cfg, "final_rmsnorm")
    phase.extra_ops.append(final_norm)
    phase.total_time_s += final_norm.time_s

    # LM Head
    lm = op_lm_head(T_full, cfg)
    phase.extra_ops.append(lm)
    phase.total_time_s += lm.time_s

    return phase


def _compression_period(cfg: Config) -> int:
    """LCM of all unique compression ratios (decode step periodicity)."""
    from math import gcd
    ratios = set(cfg.model.compress_ratios)
    result = 1
    for r in ratios:
        result = result * r // gcd(result, r)
    return result


def decode_model(cfg: Config) -> PhaseProfile:
    """Total decode phase using periodic sampling + trapezoidal interpolation.

    Per-step cost decomposes as: constant + linear(S) + periodic(S, period=P).
    Sampling the first P and last P steps captures the full periodic pattern
    at both ends; the trapezoidal rule is exact for the linear component.
    Falls back to exact iteration when output_len <= 2*P.
    """
    S_base = cfg.rt.seq_len
    N = cfg.rt.output_len
    P = _compression_period(cfg)

    # Detailed profile at first step (for reporting)
    first_step = decode_step(S_base, cfg)

    if N <= 2 * P:
        # Small output: iterate all steps exactly
        total_time = sum(decode_step(S_base + i, cfg).total_time_s for i in range(N))
    else:
        # Sample first and last periods, interpolate
        T_first = sum(decode_step(S_base + i, cfg).total_time_s for i in range(P))
        T_last = sum(decode_step(S_base + N - P + i, cfg).total_time_s for i in range(P))
        total_time = N * (T_first + T_last) / (2 * P)

    B = cfg.rt.batch_size // cfg.rt.dp
    first_step.total_time_s = total_time
    first_step.phase = "decode_total"
    first_step.total_tokens = B * N
    return first_step
