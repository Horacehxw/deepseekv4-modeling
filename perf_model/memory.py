"""KV cache and weight memory analysis."""

from .config import Config
from .roofline import bytes2

WEIGHT_BYTE_RATIOS = {"bf16": 1.0, "w8a8": 0.5}
KV_BYTE_RATIOS     = {"bf16": 1.0, "kv8": 0.5, "kv4": 0.25}


def kv_cache_memory(cfg: Config) -> dict:
    """Per-layer and total KV cache memory (per batch), after KV quantization policy."""
    B = cfg.rt.batch_size // cfg.rt.dp
    S = cfg.rt.seq_len
    kv_ratio = KV_BYTE_RATIOS[cfg.rt.kv_cache_quant_mode]
    layers = {}
    total_bytes = 0

    for i in range(cfg.model.num_hidden_layers):
        ratio = cfg.model.compress_ratios[i]
        if ratio == 1:
            W = cfg.model.window_size
            layer_bytes = B * W * cfg.model.kv_dim * 2 * kv_ratio
            layers[i] = {"type": "SWA", "bytes": layer_bytes}
        else:
            S_comp = S // ratio
            comp_bytes = B * S_comp * cfg.model.compress_c_kv * 2 * kv_ratio
            swa_bytes  = B * cfg.model.window_size * cfg.model.kv_dim * 2 * kv_ratio
            use_index  = (ratio == 4)
            idx_bytes  = B * S_comp * cfg.model.index_head_dim * 2 * kv_ratio if use_index else 0
            layer_bytes = comp_bytes + swa_bytes + idx_bytes
            layer_info = {
                "type": f"C{ratio}A",
                "comp_bytes": comp_bytes,
                "swa_bytes":  swa_bytes,
                "bytes":      layer_bytes,
            }
            if use_index:
                layer_info["idx_bytes"] = idx_bytes
            layers[i] = layer_info
        total_bytes += layer_bytes

    if cfg.rt.kv_cache_quant_mode == "bf16":
        return {"layers": layers, "total_bytes": total_bytes}
    result = {
        "layers": layers,
        "total_bytes": total_bytes + cfg.rt.kv_scale_overhead_bytes,
        "kv_cache_quant_mode": cfg.rt.kv_cache_quant_mode,
    }
    return result


def weight_memory_per_rank(cfg: Config) -> dict:
    """Weight memory per rank in bytes, after weight quantization policy."""
    H = cfg.model.hidden_size
    TP = cfg.rt.tp
    EP = cfg.rt.ep
    m = cfg.model
    w_ratio = WEIGHT_BYTE_RATIOS[cfg.rt.quant_mode]

    # --- Attention weights (per layer, per rank) ---
    w_dq   = H * m.q_lora_rank
    w_uq   = m.q_lora_rank * (m.num_attention_heads // TP) * (m.head_dim + m.rope_head_dim)
    w_kv   = H * m.kv_dim
    Ng     = m.o_groups
    w_wo_a = (Ng // TP) * (m.num_attention_heads // Ng) * m.head_dim * m.o_lora_rank
    w_wo_b = (m.o_mid_dim // TP) * H

    attn_per_layer = bytes2(w_dq + w_uq + w_kv + w_wo_a + w_wo_b) * w_ratio

    # Index weights (only for ratio > 1 layers)
    w_iq = H * (m.index_n_heads // TP) * m.index_head_dim
    index_per_layer = bytes2(w_iq) * w_ratio

    # --- MoE weights (per layer, per rank) ---
    w_gate   = H * m.n_routed_experts
    experts_per_rank = m.n_routed_experts // EP
    w_routed = experts_per_rank * 3 * H * m.moe_inter_dim
    w_shared = m.n_shared_experts * 3 * H * m.moe_inter_dim

    moe_per_layer = bytes2(w_gate + w_routed + w_shared) * w_ratio

    mhc_per_layer  = bytes2(4 * 3 * m.hc_mult * m.hc_mult) * w_ratio
    norm_per_layer = bytes2(2 * H) * w_ratio

    S       = cfg.rt.seq_len
    n_swa   = sum(1 for r in m.compress_ratios if r == 1)
    n_comp  = sum(1 for r in m.compress_ratios if r > 1)
    n_index = sum(1 for r in m.compress_ratios if r > 1 and S // r > m.index_topk)

    total_attn = attn_per_layer * m.num_hidden_layers + index_per_layer * n_index
    total_moe  = moe_per_layer * m.num_hidden_layers
    total_mhc  = mhc_per_layer * m.num_hidden_layers
    total_norm = norm_per_layer * m.num_hidden_layers

    emb_bytes      = bytes2(m.vocab_size * H) * w_ratio
    lm_head_bytes  = bytes2(H * (m.vocab_size // TP)) * w_ratio
    final_norm     = bytes2(H) * w_ratio

    total_other = total_mhc + total_norm + emb_bytes + lm_head_bytes + final_norm
    raw_total   = total_attn + total_moe + total_other

    base = {
        "attn_per_layer":  attn_per_layer,
        "index_per_layer": index_per_layer,
        "moe_per_layer":   moe_per_layer,
        "mhc_per_layer":   mhc_per_layer,
        "norm_per_layer":  norm_per_layer,
        "n_swa_layers":    n_swa,
        "n_comp_layers":   n_comp,
        "total_attn":      total_attn,
        "total_moe":       total_moe,
        "total_other":     total_other,
        "embedding":       emb_bytes,
        "lm_head":         lm_head_bytes,
        "total":           raw_total,
    }
    if cfg.rt.quant_mode == "bf16":
        return base
    base["total"] = raw_total + cfg.rt.weight_scale_overhead_bytes
    base["quant_mode"] = cfg.rt.quant_mode
    return base
