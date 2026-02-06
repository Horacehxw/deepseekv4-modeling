"""KV cache and weight memory analysis."""

from .config import Config
from .roofline import bytes2


def kv_cache_memory(cfg: Config) -> dict:
    """Per-layer and total KV cache memory (per batch)."""
    B = cfg.rt.batch_size // cfg.rt.dp
    S = cfg.rt.seq_len
    layers = {}
    total_bytes = 0

    for i in range(cfg.model.num_hidden_layers):
        ratio = cfg.model.compress_ratios[i]
        if ratio == 1:
            # Full KV: S * (k_dim + v_dim) * 2 per batch
            layer_bytes = B * S * (cfg.model.k_dim + cfg.model.v_dim) * 2
            layers[i] = {"type": "full", "bytes": layer_bytes}
        else:
            # Compressed KV
            S_comp = S // ratio
            comp_bytes = B * S_comp * (cfg.model.compress_c_k + cfg.model.compress_c_v) * 2
            # SWA window (fixed size)
            swa_bytes = B * cfg.model.window_size * (cfg.model.k_dim + cfg.model.v_dim) * 2
            # Index K cache (only for layers that use Lightning Index)
            use_index = S_comp > cfg.model.index_topk
            idx_bytes = B * S_comp * cfg.model.index_head_dim * 2 if use_index else 0
            layer_bytes = comp_bytes + swa_bytes + idx_bytes
            layer_info = {
                "type": f"C{ratio}A",
                "comp_bytes": comp_bytes,
                "swa_bytes": swa_bytes,
                "bytes": layer_bytes,
            }
            if use_index:
                layer_info["idx_bytes"] = idx_bytes
            layers[i] = layer_info
        total_bytes += layer_bytes

    return {"layers": layers, "total_bytes": total_bytes}


def weight_memory_per_rank(cfg: Config) -> dict:
    """Weight memory per rank in bytes."""
    H = cfg.model.hidden_size
    TP = cfg.rt.tp
    EP = cfg.rt.ep
    m = cfg.model

    # --- Attention weights (per layer, per rank) ---
    # W_dq: [H, q_lora_rank] — replicated
    w_dq = H * m.q_lora_rank
    # W_uq: [q_lora_rank, Nq/TP * (Dqc + Dr)]
    w_uq = m.q_lora_rank * (m.num_attention_heads // TP) * (m.head_dim + m.rope_head_dim)
    # W_k: [H, k_dim] — replicated (MQA)
    w_k = H * m.k_dim
    # W_v: [H, v_dim] — replicated (MQA)
    w_v = H * m.v_dim
    # wo_a: Ng/TP blocks of [Nq/Ng * Dqc, o_lora_rank]
    Ng = m.o_groups
    w_wo_a = (Ng // TP) * (m.num_attention_heads // Ng) * m.head_dim * m.o_lora_rank
    # wo_b: [o_mid_dim/TP, H]
    w_wo_b = (m.o_mid_dim // TP) * H

    attn_per_layer = bytes2(w_dq + w_uq + w_k + w_v + w_wo_a + w_wo_b)

    # Index weights (only for ratio > 1 layers)
    # W_iq: [H, index_n_heads/TP * index_head_dim]
    w_iq = H * (m.index_n_heads // TP) * m.index_head_dim
    # W_ik: [H, index_head_dim] — replicated
    w_ik = H * m.index_head_dim
    index_per_layer = bytes2(w_iq + w_ik)

    # --- MoE weights (per layer, per rank) ---
    # Gate: [H, n_routed_experts] — replicated
    w_gate = H * m.n_routed_experts
    # Routed experts: 3 matrices [H, inter] per expert, split by EP
    experts_per_rank = m.n_routed_experts // EP
    w_routed = experts_per_rank * 3 * H * m.moe_inter_dim
    # Shared expert: replicated, 3 matrices
    w_shared = m.n_shared_experts * 3 * H * m.moe_inter_dim

    moe_per_layer = bytes2(w_gate + w_routed + w_shared)

    # mHC weights: 3 small [hc_mult, hc_mult] matrices * 2 sub-layers * 2 (attn+moe)
    mhc_per_layer = bytes2(4 * 3 * m.hc_mult * m.hc_mult)

    # RMSNorm: 2 per layer, each H params
    norm_per_layer = bytes2(2 * H)

    # Count layers by type
    S = cfg.rt.seq_len
    n_full = sum(1 for r in m.compress_ratios if r == 1)
    n_comp = sum(1 for r in m.compress_ratios if r > 1)
    # Index weights only for layers that use Lightning Index (S//ratio > topK)
    n_index = sum(1 for r in m.compress_ratios if r > 1 and S // r > m.index_topk)

    total_attn = attn_per_layer * m.num_hidden_layers + index_per_layer * n_index
    total_moe = moe_per_layer * m.num_hidden_layers
    total_mhc = mhc_per_layer * m.num_hidden_layers
    total_norm = norm_per_layer * m.num_hidden_layers

    # Embedding + LM Head
    # Embedding: [vocab, H] — replicated (lookup table)
    emb_bytes = bytes2(m.vocab_size * H)
    # LM Head: [H, vocab/TP]
    lm_head_bytes = bytes2(H * (m.vocab_size // TP))
    # Final RMSNorm
    final_norm = bytes2(H)

    total = total_attn + total_moe + total_mhc + total_norm + emb_bytes + lm_head_bytes + final_norm

    return {
        "attn_per_layer": attn_per_layer,
        "index_per_layer": index_per_layer,
        "moe_per_layer": moe_per_layer,
        "mhc_per_layer": mhc_per_layer,
        "norm_per_layer": norm_per_layer,
        "n_full_layers": n_full,
        "n_comp_layers": n_comp,
        "total_attn": total_attn,
        "total_moe": total_moe,
        "total_other": total_mhc + total_norm + emb_bytes + lm_head_bytes + final_norm,
        "embedding": emb_bytes,
        "lm_head": lm_head_bytes,
        "total": total,
    }
