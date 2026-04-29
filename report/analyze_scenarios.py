#!/usr/bin/env python3
"""Comprehensive analysis: parameter search, P/D ratio, op analysis for DeepSeek V4.

Runs search scenarios across context lengths, prefix-cache hit rates, phases, and
hardware platforms, computes P/D disaggregation ratios, and exports JSON data for
report generation.
"""

import json
import math
import os
import sys
import time
from dataclasses import replace
from itertools import product

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perf_model.config import Config, HardwareConfig, NetworkConfig, RuntimeConfig
from perf_model.layers import prefill_model, decode_step, prefill_layer, decode_layer
from perf_model.memory import kv_cache_memory, weight_memory_per_rank
from perf_model.roofline import sum_ops

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

COMBOS = [
    {"name": "8K_4K",   "seq_len": 8192,   "output_len": 4096},
    {"name": "32K_4K",  "seq_len": 32768,  "output_len": 4096},
    {"name": "128K_4K", "seq_len": 131072, "output_len": 4096},
    {"name": "256K_4K", "seq_len": 262144, "output_len": 4096},
    {"name": "1M_4K",   "seq_len": 1_000_000, "output_len": 4096},
]
PREFIX_CACHE_HIT_RATE_VALUES = [0.0, 0.9, 0.99]

PREFILL_GPU_VALUES = [8, 16, 32, 64]
DECODE_GPU_VALUES = [8, 16, 32, 64]

TP_VALUES = [1, 2, 4, 8, 16, 32, 64]
EP_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DP_VALUES = [1, 2, 4, 8]
BATCH_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

PD_RATIO_TOLERANCE = 0.1
PD_RATIO_MAX_INSTANCES = 10_000

HARDWARE_CONFIGS = {
    "910C": {
        "device": os.path.join(BASE_DIR, "configs", "device_910C.json"),
        "network": os.path.join(BASE_DIR, "configs", "network_910C.json"),
    },
    "H20": {
        "device": os.path.join(BASE_DIR, "configs", "device_h20.json"),
        "network": os.path.join(BASE_DIR, "configs", "network_h20.json"),
    },
}


def combo_result_name(combo_name, prefix_cache_hit_rate):
    """Return a stable result key for a base combo and prefix-cache hit rate."""
    if prefix_cache_hit_rate == 0.0:
        return combo_name
    return f"{combo_name}_hit{int(round(prefix_cache_hit_rate * 100))}"


def iter_result_combos():
    """Yield every report combo expanded by prefix-cache hit rate."""
    for combo in COMBOS:
        for prefix_cache_hit_rate in PREFIX_CACHE_HIT_RATE_VALUES:
            result = dict(combo)
            result["prefix_cache_hit_rate"] = prefix_cache_hit_rate
            result["result_name"] = combo_result_name(combo["name"], prefix_cache_hit_rate)
            yield result


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_base_config(hw_name):
    hw_cfg = HARDWARE_CONFIGS[hw_name]
    return Config.from_json(
        hw_cfg["device"],
        hw_cfg["network"],
        os.path.join(BASE_DIR, "configs", "model_deepseekv4.json"),
        os.path.join(BASE_DIR, "configs", "runtime_deepseekv4.json"),
    )


def make_config(base_cfg, tp, ep, dp, batch_size, seq_len, output_len,
                sp=True, shared_expert_overlapped=True, mhc_sp=False,
                mhc_kernel_fused=None, mhc_fused_bf16=None,
                input_len=None, decode_context_len=None, prefix_cache_hit_rate=0.0):
    kwargs = dict(
        tp=tp, ep=ep, dp=dp,
        batch_size=batch_size, seq_len=seq_len,
        output_len=output_len,
        sp=sp, shared_expert_overlapped=shared_expert_overlapped,
        mhc_sp=mhc_sp,
        input_len=input_len,
        decode_context_len=decode_context_len,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
    )
    if mhc_kernel_fused is not None:
        kwargs["mhc_kernel_fused"] = mhc_kernel_fused
    if mhc_fused_bf16 is not None:
        kwargs["mhc_fused_bf16"] = mhc_fused_bf16
    new_rt = replace(base_cfg.rt, **kwargs)
    return replace(base_cfg, rt=new_rt)


def make_phase_configs(base_cfg, phase, tp, ep, dp, batch_size, seq_len, output_len,
                       prefix_cache_hit_rate):
    """Build full-context memory cfg and phase-specific compute cfg."""
    full_cfg = make_config(
        base_cfg, tp=tp, ep=ep, dp=dp, batch_size=batch_size,
        seq_len=seq_len, output_len=output_len,
        input_len=seq_len, decode_context_len=seq_len,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
    )
    eval_seq_len = (
        full_cfg.rt.effective_prefill_len
        if phase == "prefill"
        else full_cfg.rt.decode_context_len_effective
    )
    eval_cfg = replace(full_cfg, rt=replace(full_cfg.rt, seq_len=eval_seq_len))
    return full_cfg, eval_cfg


def validate_parallelism(tp, ep, model_cfg):
    if model_cfg.num_attention_heads % tp != 0:
        return False
    if model_cfg.n_routed_experts % ep != 0:
        return False
    if model_cfg.o_groups % tp != 0:
        return False
    return True


def check_memory(cfg, hbm_limit_gb=None):
    wm = weight_memory_per_rank(cfg)
    kv = kv_cache_memory(cfg)
    weight_gb = (wm["total"] + wm.get("scale_overhead_bytes", 0)) / 1e9
    kv_gb = (kv["total_bytes"] + kv.get("scale_overhead_bytes", 0)) / 1e9
    total_gb = weight_gb + kv_gb
    hbm_limit_gb = cfg.hw.usable_hbm_capacity_gb if hbm_limit_gb is None else hbm_limit_gb
    return weight_gb, kv_gb, total_gb, total_gb <= hbm_limit_gb


def approx_decode(cfg):
    S = cfg.rt.seq_len
    output_len = cfg.rt.output_len
    first = decode_step(S, cfg)
    last = decode_step(S + output_len - 1, cfg)
    approx_total_s = (first.total_time_s + last.total_time_s) / 2 * output_len
    return first.total_time_s, approx_total_s


def evaluate_prefill(cfg, logical_input_len=None):
    prefill = prefill_model(cfg)
    prefill_ms = prefill.total_time_s * 1000
    logical_len = logical_input_len if logical_input_len is not None else cfg.rt.request_input_len
    physical_gpus = cfg.rt.tp * cfg.rt.dp
    prefill_tps = (
        cfg.rt.batch_size * logical_len / prefill.total_time_s
        if prefill.total_time_s > 0 else 0
    )
    prefill_tps_per_gpu = prefill_tps / physical_gpus if physical_gpus > 0 else 0
    prefill_qps_instance = (
        cfg.rt.batch_size / prefill.total_time_s
        if prefill.total_time_s > 0 else 0
    )
    return {
        "prefill_time_ms": prefill_ms,
        "prefill_tps": prefill_tps,
        "prefill_tps_instance": prefill_tps,
        "prefill_tps_per_gpu": prefill_tps_per_gpu,
        "prefill_qps_instance": prefill_qps_instance,
    }


def evaluate_decode(cfg):
    output_len = cfg.rt.output_len
    first_step_s, approx_total_s = approx_decode(cfg)
    first_step_ms = first_step_s * 1000
    physical_gpus = cfg.rt.tp * cfg.rt.dp
    decode_tps = (
        cfg.rt.batch_size * output_len / approx_total_s
        if approx_total_s > 0 else 0
    )
    decode_tps_per_gpu = decode_tps / physical_gpus if physical_gpus > 0 else 0
    decode_qps_instance = (
        cfg.rt.batch_size / approx_total_s
        if approx_total_s > 0 else 0
    )
    return {
        "decode_first_step_ms": first_step_ms,
        "decode_total_ms_approx": approx_total_s * 1000,
        "decode_tps": decode_tps,
        "decode_tps_instance": decode_tps,
        "decode_tps_per_gpu": decode_tps_per_gpu,
        "decode_qps_instance": decode_qps_instance,
    }


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def run_search(base_cfg, phase, seq_len, output_len, hbm_limit_gb, gpu_values,
               prefix_cache_hit_rate=0.0):
    """Run grid search for a single scenario."""
    results = []
    evaluated = 0
    memory_filtered = 0

    for tp, ep, dp, batch_size in product(TP_VALUES, EP_VALUES, DP_VALUES, BATCH_VALUES):
        physical_gpus = tp * dp
        if physical_gpus not in gpu_values:
            continue
        if physical_gpus % ep != 0:
            continue
        if batch_size % dp != 0:
            continue
        if not validate_parallelism(tp, ep, base_cfg.model):
            continue

        full_cfg, eval_cfg = make_phase_configs(
            base_cfg, phase=phase, tp=tp, ep=ep, dp=dp,
            batch_size=batch_size, seq_len=seq_len, output_len=output_len,
            prefix_cache_hit_rate=prefix_cache_hit_rate,
        )

        weight_gb, kv_gb, total_gb, fits = check_memory(full_cfg, hbm_limit_gb)
        if not fits:
            memory_filtered += 1
            continue

        if phase == "prefill":
            metrics = evaluate_prefill(eval_cfg, logical_input_len=full_cfg.rt.request_input_len)
        else:
            metrics = evaluate_decode(eval_cfg)
        evaluated += 1

        per_rank_batch = batch_size // dp
        edp = physical_gpus // ep
        row = {
            "tp": tp, "ep": ep, "dp": dp, "edp": edp,
            "batch_size": batch_size, "seq_len": seq_len, "output_len": output_len,
            "logical_input_len": full_cfg.rt.request_input_len,
            "effective_prefill_len": full_cfg.rt.effective_prefill_len,
            "decode_context_len": full_cfg.rt.decode_context_len_effective,
            "prefix_cache_hit_rate": prefix_cache_hit_rate,
            "physical_gpus": physical_gpus, "per_rank_batch": per_rank_batch,
            "weight_gb": round(weight_gb, 3),
            "kv_cache_gb": round(kv_gb, 3),
            "hbm_total_gb": round(total_gb, 3),
        }
        row.update(metrics)
        results.append(row)

        if evaluated % 500 == 0:
            print(f"    Evaluated {evaluated} configs (mem-filtered {memory_filtered})...")

    print(f"    Evaluated: {evaluated}, Memory-filtered: {memory_filtered}")
    return results


def sort_results(results, phase, scenario):
    if phase == "prefill" and scenario == "latency":
        results.sort(key=lambda r: r["prefill_time_ms"])
    elif phase == "decode" and scenario == "latency":
        results.sort(key=lambda r: r["decode_first_step_ms"])
    elif phase == "prefill" and scenario == "throughput":
        results.sort(key=lambda r: r["prefill_tps_per_gpu"], reverse=True)
    elif phase == "decode" and scenario == "throughput":
        results.sort(key=lambda r: r["decode_tps_per_gpu"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# P/D Ratio Calculator
# ---------------------------------------------------------------------------

def compute_pd_ratio(p_qps_instance, d_qps_instance,
                     tolerance=PD_RATIO_TOLERANCE,
                     max_instances=PD_RATIO_MAX_INSTANCES):
    """Find the smallest integer P:D ratio that balances aggregate QPS.

    Solves N * prefill_qps ~= M * decode_qps with integer N and M. The
    returned pair is the smallest deployment unit within the requested
    relative imbalance tolerance.
    """
    if p_qps_instance <= 0:
        raise ValueError("p_qps_instance must be positive")
    if d_qps_instance <= 0:
        raise ValueError("d_qps_instance must be positive")
    if tolerance < 0 or tolerance >= 1:
        raise ValueError("tolerance must be in [0, 1)")
    if max_instances < 1:
        raise ValueError("max_instances must be positive")

    ratio_float = d_qps_instance / p_qps_instance
    best = None
    best_score = None
    nearest = None
    nearest_score = None

    for decode_instances in range(1, max_instances + 1):
        desired_prefill = ratio_float * decode_instances
        min_prefill = desired_prefill * (1 - tolerance)
        max_prefill = desired_prefill / (1 - tolerance)
        candidates = {
            math.ceil(min_prefill),
            math.floor(max_prefill),
            math.floor(desired_prefill),
            round(desired_prefill),
            math.ceil(desired_prefill),
        }

        for prefill_instances in candidates:
            if prefill_instances < 1 or prefill_instances > max_instances:
                continue

            prefill_aggregate_qps = prefill_instances * p_qps_instance
            decode_aggregate_qps = decode_instances * d_qps_instance
            qps_imbalance = prefill_aggregate_qps - decode_aggregate_qps
            qps_imbalance_pct = (
                abs(qps_imbalance)
                / max(prefill_aggregate_qps, decode_aggregate_qps)
            )
            ratio_actual = prefill_instances / decode_instances

            result = {
                "pd_ratio_float": ratio_float,
                "pd_ratio_actual": ratio_actual,
                "prefill_instances": prefill_instances,
                "decode_instances": decode_instances,
                "prefill_aggregate_qps": prefill_aggregate_qps,
                "decode_aggregate_qps": decode_aggregate_qps,
                "qps_imbalance": qps_imbalance,
                "qps_imbalance_pct": qps_imbalance_pct,
                "balance_tolerance": tolerance,
                "label": f"{prefill_instances}P:{decode_instances}D",
            }
            score = (
                prefill_instances + decode_instances,
                qps_imbalance_pct,
                0 if qps_imbalance >= 0 else 1,
                abs(ratio_actual - ratio_float),
                prefill_instances,
                decode_instances,
            )
            nearest_candidate_score = (
                qps_imbalance_pct,
                prefill_instances + decode_instances,
                prefill_instances,
                decode_instances,
            )

            if nearest_score is None or nearest_candidate_score < nearest_score:
                nearest = result
                nearest_score = nearest_candidate_score
            if qps_imbalance_pct <= tolerance + 1e-12:
                if best_score is None or score < best_score:
                    best = result
                    best_score = score

    if best is None:
        detail = ""
        if nearest is not None:
            detail = (
                f"; nearest={nearest['label']} "
                f"imbalance={nearest['qps_imbalance_pct']:.6f}"
            )
        raise ValueError(
            f"No P/D ratio found within tolerance={tolerance} "
            f"and max_instances={max_instances}{detail}"
        )
    return best


# ---------------------------------------------------------------------------
# Op Analysis
# ---------------------------------------------------------------------------

OP_CATEGORIES = {
    "Attention Proj": ["q_proj_dq", "q_proj_uq", "kv_proj", "wo_a", "wo_b"],
    "Attention Compute": ["attention_swa", "attention_comp"],
    "KV Compression": ["kv_compression", "kv_compression_decode"],
    "Lightning Index": ["index_iq_proj", "index_kv_compress",
                        "index_kv_compress_decode", "index_score", "index_score_ar"],
    "mHC": ["mhc_pre_attn", "sinkhorn_attn", "mhc_post_attn",
            "mhc_pre_moe", "sinkhorn_moe", "mhc_post_moe",
            "mhc_post_attn_pre_moe"],
    "MoE Gate": ["moe_gate"],
    "MoE Routed": ["routed_gate_proj", "routed_up_proj", "routed_silu_mul", "routed_down_proj"],
    "MoE Shared": ["shared_gate_proj", "shared_up_proj", "shared_silu_mul", "shared_down_proj",
                    "shared_expert_excess"],
    "Communication": ["attn_tp_allreduce", "moe_ep_dispatch", "moe_ep_combine",
                      "sp_ag_before_attn", "sp_ag_before_moe", "sp_ag_after_moe",
                      "sp_ag_before_lm_head"],
    "Norm": ["rmsnorm_attn", "rmsnorm_moe", "final_rmsnorm"],
    "Embedding/LMHead": ["embedding", "lm_head"],
}


def categorize_ops(ops_list):
    """Categorize a flat list of ops by OP_CATEGORIES, return time breakdown."""
    cat_times = {cat: 0.0 for cat in OP_CATEGORIES}
    cat_bottlenecks = {cat: {"CUBE": 0.0, "VEC": 0.0, "MEM": 0.0, "COMM": 0.0}
                       for cat in OP_CATEGORIES}
    uncategorized = 0.0

    for op in ops_list:
        found = False
        for cat, names in OP_CATEGORIES.items():
            if op.name in names:
                cat_times[cat] += op.time_s
                if op.bottleneck in cat_bottlenecks[cat]:
                    cat_bottlenecks[cat][op.bottleneck] += op.time_s
                found = True
                break
        if not found:
            uncategorized += op.time_s

    total = sum(cat_times.values()) + uncategorized
    result = {}
    for cat in OP_CATEGORIES:
        t = cat_times[cat]
        pct = (t / total * 100) if total > 0 else 0
        dominant = max(cat_bottlenecks[cat], key=cat_bottlenecks[cat].get)
        result[cat] = {
            "time_s": t,
            "time_ms": t * 1000,
            "pct": round(pct, 2),
            "dominant_bottleneck": dominant if t > 0 else "N/A",
        }
    if uncategorized > 0:
        result["Other"] = {
            "time_s": uncategorized,
            "time_ms": uncategorized * 1000,
            "pct": round(uncategorized / total * 100, 2),
            "dominant_bottleneck": "N/A",
        }
    return result


def collect_all_ops_prefill(cfg):
    """Collect all ops from full prefill model run."""
    phase = prefill_model(cfg)
    all_ops = list(phase.extra_ops)
    for lp in phase.layer_profiles:
        all_ops.extend(lp.ops)
    return all_ops, phase.total_time_s


def collect_all_ops_decode(cfg):
    """Collect all ops from a single decode step at seq_len."""
    S = cfg.rt.seq_len
    phase = decode_step(S, cfg)
    all_ops = list(phase.extra_ops)
    for lp in phase.layer_profiles:
        all_ops.extend(lp.ops)
    return all_ops, phase.total_time_s


# ---------------------------------------------------------------------------
# SP / mHC-SP Comparison
# ---------------------------------------------------------------------------

def run_sp_comparison(base_cfg, seq_len, output_len, tp, ep, dp, batch_size,
                      prefix_cache_hit_rate=0.0):
    """Run 3 SP configs and compare prefill times."""
    results = {}

    configs = [
        ("no_SP", False, False),
        ("SP_only", True, False),
        ("SP_mHC_SP", True, True),
    ]

    for label, sp, mhc_sp in configs:
        full_cfg = make_config(
            base_cfg, tp=tp, ep=ep, dp=dp,
            batch_size=batch_size, seq_len=seq_len,
            output_len=output_len, sp=sp, mhc_sp=mhc_sp,
            input_len=seq_len, decode_context_len=seq_len,
            prefix_cache_hit_rate=prefix_cache_hit_rate,
        )
        cfg = replace(full_cfg, rt=replace(full_cfg.rt, seq_len=full_cfg.rt.effective_prefill_len))
        all_ops, total_time = collect_all_ops_prefill(cfg)
        cat_breakdown = categorize_ops(all_ops)

        results[label] = {
            "prefill_time_ms": total_time * 1000,
            "sp": sp,
            "mhc_sp": mhc_sp,
            "logical_input_len": full_cfg.rt.request_input_len,
            "effective_prefill_len": full_cfg.rt.effective_prefill_len,
            "prefix_cache_hit_rate": prefix_cache_hit_rate,
            "category_breakdown": cat_breakdown,
        }

    return results


# ---------------------------------------------------------------------------
# mHC Optimization Comparison
# ---------------------------------------------------------------------------

MHC_OPTIMIZATION_LEVELS = [
    {"label": "unfused_fp32",   "mhc_kernel_fused": False, "mhc_sp": False, "mhc_fused_bf16": False,
     "description": "Original baseline (unfused FP32)"},
    {"label": "fused_fp32",     "mhc_kernel_fused": True,  "mhc_sp": False, "mhc_fused_bf16": False,
     "description": "Kernel-fused FP32 (new default)"},
    {"label": "fused_fp32_sp",  "mhc_kernel_fused": True,  "mhc_sp": True,  "mhc_fused_bf16": False,
     "description": "Kernel-fused FP32 + Sequence Parallelism"},
    {"label": "fused_bf16_sp",  "mhc_kernel_fused": True,  "mhc_sp": True,  "mhc_fused_bf16": True,
     "description": "Kernel-fused BF16 + Sequence Parallelism"},
]


def run_mhc_optimization_comparison(base_cfg, seq_len, output_len, tp, ep, dp, batch_size,
                                    prefix_cache_hit_rate=0.0):
    """Benchmark 4 mHC optimization levels for prefill and decode."""
    results = {}

    for level in MHC_OPTIMIZATION_LEVELS:
        label = level["label"]
        full_cfg = make_config(
            base_cfg, tp=tp, ep=ep, dp=dp,
            batch_size=batch_size, seq_len=seq_len,
            output_len=output_len, sp=True,
            mhc_sp=level["mhc_sp"],
            mhc_kernel_fused=level["mhc_kernel_fused"],
            mhc_fused_bf16=level["mhc_fused_bf16"],
            input_len=seq_len, decode_context_len=seq_len,
            prefix_cache_hit_rate=prefix_cache_hit_rate,
        )
        prefill_cfg = replace(full_cfg, rt=replace(full_cfg.rt, seq_len=full_cfg.rt.effective_prefill_len))
        decode_cfg = replace(full_cfg, rt=replace(full_cfg.rt, seq_len=full_cfg.rt.decode_context_len_effective))

        # Prefill
        ops_p, time_p = collect_all_ops_prefill(prefill_cfg)
        cat_p = categorize_ops(ops_p)

        # Decode (at seq_len)
        ops_d, time_d = collect_all_ops_decode(decode_cfg)
        cat_d = categorize_ops(ops_d)

        # Per-op detail for representative layer (layer 3, C4A)
        lp = prefill_layer(3, prefill_cfg)
        per_op_detail = []
        for op in lp.ops:
            per_op_detail.append({
                "name": op.name,
                "time_ms": op.time_s * 1000,
                "bottleneck": op.bottleneck,
            })

        results[label] = {
            "description": level["description"],
            "mhc_kernel_fused": level["mhc_kernel_fused"],
            "mhc_sp": level["mhc_sp"],
            "mhc_fused_bf16": level["mhc_fused_bf16"],
            "logical_input_len": full_cfg.rt.request_input_len,
            "effective_prefill_len": full_cfg.rt.effective_prefill_len,
            "decode_context_len": full_cfg.rt.decode_context_len_effective,
            "prefix_cache_hit_rate": prefix_cache_hit_rate,
            "prefill_time_ms": time_p * 1000,
            "decode_step_ms": time_d * 1000,
            "prefill_category_breakdown": cat_p,
            "decode_category_breakdown": cat_d,
            "representative_layer_ops": per_op_detail,
        }

    return results


# ---------------------------------------------------------------------------
# V3 Comparison
# ---------------------------------------------------------------------------

# DeepSeek V3 reference parameters (from paper)
V3_PARAMS = {
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_kv_heads": 1,  # MLA with kv_lora_rank
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "q_lora_rank": 1536,
    "n_routed_experts": 256,
    "n_activated_experts": 8,  # top-8
    "n_shared_experts": 1,
    "moe_inter_dim": 2048,
    "vocab_size": 129280,
    "has_kv_compression": False,
    "has_mhc": False,
}


def compute_v3_comparison():
    """Compute side-by-side V4 vs V3 architecture comparison."""
    v4 = {
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_kv_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "n_routed_experts": 256,
        "n_activated_experts": 6,
        "n_shared_experts": 1,
        "moe_inter_dim": 2048,
        "vocab_size": 129280,
        "has_kv_compression": True,
        "compress_ratios": "2 full + 21 C4A + 20 C128A",
        "has_mhc": True,
        "hc_mult": 4,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "window_size": 128,
    }
    v3 = dict(V3_PARAMS)

    # --- Parameter count estimates ---
    # V4 attention per layer (approx)
    v4_attn_params = (
        v4["hidden_size"] * v4["q_lora_rank"]  # W_dq
        + v4["q_lora_rank"] * v4["num_attention_heads"] * (v4["head_dim"] + 64)  # W_uq (head_dim + rope)
        + v4["hidden_size"] * v4["head_dim"]  # W_k
        + v4["hidden_size"] * v4["head_dim"]  # W_v
        + v4["o_groups"] * (v4["num_attention_heads"] // v4["o_groups"]) * v4["head_dim"] * v4["o_lora_rank"]  # wo_a
        + v4["o_groups"] * v4["o_lora_rank"] * v4["hidden_size"]  # wo_b
    )

    # V3 attention per layer (MLA)
    v3_attn_params = (
        v3["hidden_size"] * v3["q_lora_rank"]  # W_dq
        + v3["q_lora_rank"] * v3["num_attention_heads"] * (v3["qk_nope_head_dim"] + v3["qk_rope_head_dim"])  # W_uq
        + v3["hidden_size"] * (v3["kv_lora_rank"] + v3["qk_rope_head_dim"])  # W_kv_down (joint K+V)
        + v3["kv_lora_rank"] * v3["num_attention_heads"] * (v3["qk_nope_head_dim"] + v3["v_head_dim"])  # W_kv_up
        + v3["num_attention_heads"] * v3["v_head_dim"] * v3["hidden_size"]  # W_o
    )

    # MoE per layer
    v4_moe_params = (
        v4["hidden_size"] * v4["n_routed_experts"]  # gate
        + v4["n_routed_experts"] * 3 * v4["hidden_size"] * v4["moe_inter_dim"]  # routed
        + v4["n_shared_experts"] * 3 * v4["hidden_size"] * v4["moe_inter_dim"]  # shared
    )
    v3_moe_params = (
        v3["hidden_size"] * v3["n_routed_experts"]  # gate
        + v3["n_routed_experts"] * 3 * v3["hidden_size"] * v3["moe_inter_dim"]  # routed
        + v3["n_shared_experts"] * 3 * v3["hidden_size"] * v3["moe_inter_dim"]  # shared
    )

    # V4 mHC per layer: 4 sub-layers × 3 matrices × hc_mult × hc_mult
    v4_mhc_params = 4 * 3 * v4["hc_mult"] * v4["hc_mult"]

    # V4 index per layer (for compressed layers)
    v4_index_params = (
        v4["hidden_size"] * v4["index_n_heads"] * v4["index_head_dim"]  # W_iq
        + v4["hidden_size"] * v4["index_head_dim"]  # W_ik
    )

    # Total params estimate
    v4_total = (
        v4_attn_params * v4["num_hidden_layers"]
        + v4_moe_params * v4["num_hidden_layers"]
        + v4_mhc_params * v4["num_hidden_layers"]
        + v4_index_params * 41  # 21 C4A + 20 C128A layers
        + v4["vocab_size"] * v4["hidden_size"] * 2  # embedding + lm_head
    )
    v3_total = (
        v3_attn_params * v3["num_hidden_layers"]
        + v3_moe_params * v3["num_hidden_layers"]
        + v3["vocab_size"] * v3["hidden_size"] * 2  # embedding + lm_head
    )

    # --- KV cache per token (BF16 = 2 bytes) ---
    # V4: compressed layers have much less KV cache
    # Full layers: (k_dim + v_dim) × 2 = (512 + 512) × 2 = 2048 bytes
    v4_kv_full = (v4["head_dim"] + v4["head_dim"]) * 2  # per token per layer
    # C4A: compressed KV at 1/4 + SWA window is fixed
    v4_kv_c4a_per_token = (v4["head_dim"] + v4["head_dim"]) * 2 / 4  # compressed contribution per input token
    # C128A: compressed KV at 1/128
    v4_kv_c128a_per_token = (v4["head_dim"] + v4["head_dim"]) * 2 / 128

    # V3: MLA stores kv_lora_rank + rope dim per token per layer
    v3_kv_per_token_per_layer = (v3["kv_lora_rank"] + v3["qk_rope_head_dim"]) * 2  # (512+64)*2 = 1152 bytes

    v4_kv_per_token_total = (
        v4_kv_full * 2  # 2 full layers
        + v4_kv_c4a_per_token * 21  # 21 C4A layers
        + v4_kv_c128a_per_token * 20  # 20 C128A layers
    )
    v3_kv_per_token_total = v3_kv_per_token_per_layer * v3["num_hidden_layers"]

    # --- Per-layer FLOPs estimate (prefill, per token) ---
    # Attention FLOPs per token (approx, just matmuls):
    v4_attn_flops = v4_attn_params * 2  # each param contributes ~2 FLOPs for matmul
    v3_attn_flops = v3_attn_params * 2

    # MoE FLOPs per token (only activated experts):
    v4_moe_flops = (
        v4["n_activated_experts"] * 3 * v4["hidden_size"] * v4["moe_inter_dim"] * 2
        + v4["n_shared_experts"] * 3 * v4["hidden_size"] * v4["moe_inter_dim"] * 2
    )
    v3_moe_flops = (
        v3["n_activated_experts"] * 3 * v3["hidden_size"] * v3["moe_inter_dim"] * 2
        + v3["n_shared_experts"] * 3 * v3["hidden_size"] * v3["moe_inter_dim"] * 2
    )

    # Weight memory per rank (for reference, at TP=8, EP=16)
    # Use V4 model to get actual weight memory
    base_cfg = load_base_config("910C")
    cfg_v4 = make_config(base_cfg, tp=8, ep=16, dp=2, batch_size=16,
                         seq_len=8192, output_len=4096)
    v4_weights = weight_memory_per_rank(cfg_v4)

    return {
        "v4": v4,
        "v3": v3,
        "comparison": {
            "v4_attn_params_per_layer": v4_attn_params,
            "v3_attn_params_per_layer": v3_attn_params,
            "v4_moe_params_per_layer": v4_moe_params,
            "v3_moe_params_per_layer": v3_moe_params,
            "v4_mhc_params_per_layer": v4_mhc_params,
            "v4_index_params_per_layer": v4_index_params,
            "v4_total_params_approx": v4_total,
            "v3_total_params_approx": v3_total,
            "v4_kv_per_token_bytes": round(v4_kv_per_token_total, 1),
            "v3_kv_per_token_bytes": round(v3_kv_per_token_total, 1),
            "kv_compression_ratio": round(v3_kv_per_token_total / v4_kv_per_token_total, 2)
                if v4_kv_per_token_total > 0 else 0,
            "v4_attn_flops_per_token": v4_attn_flops,
            "v3_attn_flops_per_token": v3_attn_flops,
            "v4_moe_flops_per_token": v4_moe_flops,
            "v3_moe_flops_per_token": v3_moe_flops,
            "v4_weight_memory_per_rank_gb": round(v4_weights["total"] / 1e9, 2),
        },
    }


# ---------------------------------------------------------------------------
# KV Cache Scaling
# ---------------------------------------------------------------------------

KV_SCALING_SEQ_LENS = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1_000_000,
]


def compute_kv_cache_scaling(base_cfg, hw_name):
    """Sweep seq_len and compute KV cache, decode time, and op breakdown."""
    hbm_limit = base_cfg.hw.usable_hbm_capacity_gb
    results = []

    for seq_len in KV_SCALING_SEQ_LENS:
        # Use a standard config: TP=4, EP=32, DP=8, BS=8 for decode
        tp, ep, dp, bs = 4, 32, 8, 8
        cfg = make_config(base_cfg, tp=tp, ep=ep, dp=dp,
                          batch_size=bs, seq_len=seq_len, output_len=4096)

        # KV cache
        kv = kv_cache_memory(cfg)
        kv_gb = (kv["total_bytes"] + kv.get("scale_overhead_bytes", 0)) / 1e9

        # Weight memory
        wm = weight_memory_per_rank(cfg)
        weight_gb = (wm["total"] + wm.get("scale_overhead_bytes", 0)) / 1e9
        total_gb = weight_gb + kv_gb
        fits = total_gb <= hbm_limit

        # Decode step time
        if fits:
            ops_d, time_d = collect_all_ops_decode(cfg)
            cat_d = categorize_ops(ops_d)
            decode_ms = time_d * 1000
        else:
            cat_d = {}
            decode_ms = None

        # Compute hypothetical "no compression" KV cache
        # If all 43 layers used full attention: S * (k_dim + v_dim) * 2 * B * 43
        B = bs // dp
        no_comp_kv_bytes = B * seq_len * (cfg.model.kv_dim * 2) * 2 * cfg.model.num_hidden_layers
        no_comp_kv_gb = no_comp_kv_bytes / 1e9

        # V3-style KV cache: (kv_lora_rank + rope_dim) * 2 * B * 61 layers
        v3_kv_bytes = B * seq_len * (V3_PARAMS["kv_lora_rank"] + V3_PARAMS["qk_rope_head_dim"]) * 2 * V3_PARAMS["num_hidden_layers"]
        v3_kv_gb = v3_kv_bytes / 1e9

        results.append({
            "seq_len": seq_len,
            "kv_cache_gb": round(kv_gb, 3),
            "weight_gb": round(weight_gb, 3),
            "total_hbm_gb": round(total_gb, 3),
            "fits_in_hbm": fits,
            "decode_step_ms": round(decode_ms, 3) if decode_ms else None,
            "no_compression_kv_gb": round(no_comp_kv_gb, 3),
            "v3_style_kv_gb": round(v3_kv_gb, 3),
            "compression_ratio": round(no_comp_kv_gb / kv_gb, 2) if kv_gb > 0 else 0,
            "category_breakdown": cat_d,
        })

    return results


# ---------------------------------------------------------------------------
# Attention Analysis
# ---------------------------------------------------------------------------

def compute_attention_analysis(base_cfg, hw_name):
    """Per-layer-type KV cache breakdown and attention scaling analysis."""
    hbm_limit = base_cfg.hw.usable_hbm_capacity_gb

    # --- Per-layer-type KV cache breakdown ---
    layer_type_kv = {}
    for seq_len in [8192, 32768, 131072, 262144, 1_000_000]:
        cfg = make_config(base_cfg, tp=4, ep=32, dp=8, batch_size=8,
                          seq_len=seq_len, output_len=4096)
        kv = kv_cache_memory(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp

        full_bytes = sum(l["bytes"] for l in kv["layers"].values() if l["type"] == "full")
        c4a_bytes = sum(l["bytes"] for l in kv["layers"].values() if l["type"] == "C4A")
        c128a_bytes = sum(l["bytes"] for l in kv["layers"].values() if l["type"] == "C128A")

        layer_type_kv[str(seq_len)] = {
            "full_attn_gb": round(full_bytes / 1e9, 4),
            "c4a_gb": round(c4a_bytes / 1e9, 4),
            "c128a_gb": round(c128a_bytes / 1e9, 4),
            "total_gb": round(kv["total_bytes"] / 1e9, 4),
            "full_attn_pct": round(full_bytes / kv["total_bytes"] * 100, 1) if kv["total_bytes"] > 0 else 0,
            "c4a_pct": round(c4a_bytes / kv["total_bytes"] * 100, 1) if kv["total_bytes"] > 0 else 0,
            "c128a_pct": round(c128a_bytes / kv["total_bytes"] * 100, 1) if kv["total_bytes"] > 0 else 0,
        }

    # --- Compressed vs uncompressed comparison ---
    compressed_vs_uncompressed = {}
    for seq_len in [8192, 32768, 65536, 131072, 262144, 1_000_000]:
        cfg = make_config(base_cfg, tp=4, ep=32, dp=8, batch_size=8,
                          seq_len=seq_len, output_len=4096)
        kv = kv_cache_memory(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp

        v4_kv_gb = kv["total_bytes"] / 1e9
        # Hypothetical uncompressed: all 43 layers as full attention
        no_comp = B * seq_len * (cfg.model.kv_dim * 2) * 2 * cfg.model.num_hidden_layers / 1e9
        # V3 MLA style
        v3_kv = B * seq_len * (V3_PARAMS["kv_lora_rank"] + V3_PARAMS["qk_rope_head_dim"]) * 2 * V3_PARAMS["num_hidden_layers"] / 1e9

        compressed_vs_uncompressed[str(seq_len)] = {
            "v4_compressed_gb": round(v4_kv_gb, 3),
            "v4_uncompressed_gb": round(no_comp, 3),
            "v3_mla_gb": round(v3_kv, 3),
            "v4_savings_vs_uncompressed": round(no_comp / v4_kv_gb, 2) if v4_kv_gb > 0 else 0,
            "v4_vs_v3": round(v3_kv / v4_kv_gb, 2) if v4_kv_gb > 0 else 0,
        }

    # --- Attention compute scaling across seq_len per layer type ---
    attn_scaling = {}
    for seq_len in [8192, 32768, 131072, 262144, 1_000_000]:
        cfg = make_config(base_cfg, tp=8, ep=16, dp=2, batch_size=16,
                          seq_len=seq_len, output_len=4096)

        # Check memory
        _, _, total_gb, fits = check_memory(cfg, hbm_limit)
        if not fits:
            # Try smaller batch
            for smaller_bs in [8, 4, 2]:
                cfg = make_config(base_cfg, tp=8, ep=16, dp=2,
                                  batch_size=smaller_bs, seq_len=seq_len, output_len=4096)
                _, _, total_gb, fits = check_memory(cfg, hbm_limit)
                if fits:
                    break
            else:
                attn_scaling[str(seq_len)] = {"error": "OOM"}
                continue

        # Get per-layer breakdown for different layer types
        # Layer 0 = full attention, Layer 2 = C4A, Layer 3 = C128A (check ratios)
        layer_data = {}
        for layer_idx in [0, 2, 3]:  # full, first C4A, first C128A
            ratio = cfg.model.compress_ratios[layer_idx]
            if ratio == 1:
                ltype = "full"
            elif ratio == 4:
                ltype = "C4A"
            else:
                ltype = f"C{ratio}A"

            lp = prefill_layer(layer_idx, cfg)
            # Extract attention-related ops
            attn_ops_time = 0
            total_time = 0
            for op in lp.ops:
                total_time += op.time_s
                if op.name in ["attention_full", "attention_comp", "q_proj_dq", "q_proj_uq",
                               "k_proj", "v_proj", "wo_a", "wo_b"]:
                    attn_ops_time += op.time_s

            layer_data[ltype] = {
                "layer_idx": layer_idx,
                "ratio": ratio,
                "attn_time_ms": round(attn_ops_time * 1000, 3),
                "layer_total_ms": round(total_time * 1000, 3),
                "attn_pct": round(attn_ops_time / total_time * 100, 1) if total_time > 0 else 0,
            }

        attn_scaling[str(seq_len)] = {
            "batch_size": cfg.rt.batch_size,
            "layers": layer_data,
        }

    return {
        "layer_type_kv_breakdown": layer_type_kv,
        "compressed_vs_uncompressed": compressed_vs_uncompressed,
        "attention_compute_scaling": attn_scaling,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    start_time = time.time()

    all_search_results = {}
    all_pd_results = {}
    all_op_analysis = {}
    all_sp_comparison = {}
    all_mhc_comparison = {}

    for hw_name in ["910C", "H20"]:
        print(f"\n{'='*70}")
        print(f"HARDWARE: {hw_name}")
        print(f"{'='*70}")

        base_cfg = load_base_config(hw_name)
        hbm_limit = base_cfg.hw.usable_hbm_capacity_gb

        hw_search = {}
        hw_pd = {}
        hw_ops = {}
        hw_sp = {}
        hw_mhc = {}

        for combo in iter_result_combos():
            combo_name = combo["result_name"]
            base_combo_name = combo["name"]
            seq_len = combo["seq_len"]
            output_len = combo["output_len"]
            prefix_cache_hit_rate = combo["prefix_cache_hit_rate"]

            print(f"\n  --- Combo: {combo_name} "
                  f"(base={base_combo_name}, seq={seq_len}, out={output_len}, "
                  f"hit={prefix_cache_hit_rate:.0%}) ---")

            combo_search = {}

            # 4 scenarios per combo
            for phase, scenario in [("prefill", "latency"), ("prefill", "throughput"),
                                     ("decode", "latency"), ("decode", "throughput")]:
                key = f"{phase}_{scenario}"
                gpu_values = PREFILL_GPU_VALUES if phase == "prefill" else DECODE_GPU_VALUES
                print(f"  [{hw_name}] {combo_name} {key}...")

                results = run_search(
                    base_cfg, phase, seq_len, output_len, hbm_limit, gpu_values,
                    prefix_cache_hit_rate=prefix_cache_hit_rate,
                )
                results = sort_results(results, phase, scenario)
                combo_search[key] = results[:20]  # Keep top 20

                # Print top 3
                if results:
                    print(f"    Top 3:")
                    for i, r in enumerate(results[:3]):
                        if phase == "prefill":
                            if scenario == "latency":
                                print(f"      #{i+1}: TP={r['tp']} EP={r['ep']} DP={r['dp']} "
                                      f"BS={r['batch_size']} GPUs={r['physical_gpus']} "
                                      f"-> {r['prefill_time_ms']:.1f}ms")
                            else:
                                print(f"      #{i+1}: TP={r['tp']} EP={r['ep']} DP={r['dp']} "
                                      f"BS={r['batch_size']} GPUs={r['physical_gpus']} "
                                      f"-> {r['prefill_tps_per_gpu']:.2f} tps/gpu")
                        else:
                            if scenario == "latency":
                                print(f"      #{i+1}: TP={r['tp']} EP={r['ep']} DP={r['dp']} "
                                      f"BS={r['batch_size']} GPUs={r['physical_gpus']} "
                                      f"-> {r['decode_first_step_ms']:.3f}ms")
                            else:
                                print(f"      #{i+1}: TP={r['tp']} EP={r['ep']} DP={r['dp']} "
                                      f"BS={r['batch_size']} GPUs={r['physical_gpus']} "
                                      f"-> {r['decode_tps_per_gpu']:.3f} tps/gpu")
                else:
                    print(f"    No valid configs found!")

            hw_search[combo_name] = combo_search

            # --- P/D Ratio ---
            p_throughput = combo_search.get("prefill_throughput", [])
            d_throughput = combo_search.get("decode_throughput", [])

            if p_throughput and d_throughput:
                p_best = p_throughput[0]
                d_best = d_throughput[0]
                g_p = p_best["physical_gpus"]
                g_d = d_best["physical_gpus"]
                p_tps_gpu = p_best["prefill_tps_per_gpu"]
                d_tps_gpu = d_best["decode_tps_per_gpu"]
                p_qps = p_best["prefill_qps_instance"]
                d_qps = d_best["decode_qps_instance"]

                pd_ratio = compute_pd_ratio(p_qps, d_qps)
                n_p = pd_ratio["prefill_instances"]
                n_d = pd_ratio["decode_instances"]
                total_gpus_min = n_p * g_p + n_d * g_d

                pd_info = {
                    "schema_version": 2,
                    "prefill_config": {
                        "tp": p_best["tp"], "ep": p_best["ep"], "dp": p_best["dp"],
                        "batch_size": p_best["batch_size"],
                        "gpus": g_p, "tps_per_gpu": round(p_tps_gpu, 2),
                        "tps_instance": round(p_best["prefill_tps_instance"], 2),
                        "qps_instance": round(p_qps, 4),
                    },
                    "decode_config": {
                        "tp": d_best["tp"], "ep": d_best["ep"], "dp": d_best["dp"],
                        "batch_size": d_best["batch_size"],
                        "gpus": g_d, "tps_per_gpu": round(d_tps_gpu, 2),
                        "tps_instance": round(d_best["decode_tps_instance"], 2),
                        "qps_instance": round(d_qps, 4),
                    },
                    "input_len": seq_len,
                    "output_len": output_len,
                    "prefix_cache_hit_rate": p_best.get("prefix_cache_hit_rate", 0.0),
                    "pd_ratio_float": round(pd_ratio["pd_ratio_float"], 3),
                    "pd_ratio_actual": round(pd_ratio["pd_ratio_actual"], 3),
                    "prefill_instances": n_p,
                    "decode_instances": n_d,
                    "prefill_aggregate_qps": round(pd_ratio["prefill_aggregate_qps"], 4),
                    "decode_aggregate_qps": round(pd_ratio["decode_aggregate_qps"], 4),
                    "qps_imbalance": round(pd_ratio["qps_imbalance"], 6),
                    "qps_imbalance_pct": round(pd_ratio["qps_imbalance_pct"], 6),
                    "balance_tolerance": pd_ratio["balance_tolerance"],
                    "total_gpus_min": total_gpus_min,
                    "label": pd_ratio["label"],
                }
                hw_pd[combo_name] = pd_info
                print(f"\n  P/D Ratio ({combo_name}): {pd_info['label']} "
                      f"({n_p}×{g_p}GPU P + {n_d}×{g_d}GPU D = "
                      f"{pd_info['total_gpus_min']} GPUs, "
                      f"imbalance={pd_info['qps_imbalance_pct']*100:.2f}%)")
            else:
                hw_pd[combo_name] = {"error": "No valid configs for P/D calculation"}
                print(f"\n  P/D Ratio ({combo_name}): SKIPPED (no valid configs)")

            # --- Op Analysis (use best throughput config for detailed analysis) ---
            if p_throughput:
                best_p = p_throughput[0]
                _, cfg_p = make_phase_configs(
                    base_cfg, phase="prefill",
                    tp=best_p["tp"], ep=best_p["ep"], dp=best_p["dp"],
                    batch_size=best_p["batch_size"], seq_len=seq_len,
                    output_len=output_len, prefix_cache_hit_rate=prefix_cache_hit_rate,
                )
                ops_p, time_p = collect_all_ops_prefill(cfg_p)
                cat_p = categorize_ops(ops_p)
            else:
                cat_p = {}

            if d_throughput:
                best_d = d_throughput[0]
                _, cfg_d = make_phase_configs(
                    base_cfg, phase="decode",
                    tp=best_d["tp"], ep=best_d["ep"], dp=best_d["dp"],
                    batch_size=best_d["batch_size"], seq_len=seq_len,
                    output_len=output_len, prefix_cache_hit_rate=prefix_cache_hit_rate,
                )
                ops_d, time_d = collect_all_ops_decode(cfg_d)
                cat_d = categorize_ops(ops_d)
            else:
                cat_d = {}

            hw_ops[combo_name] = {
                "prefill": cat_p,
                "decode": cat_d,
            }

        # --- SP Comparison (use 8K combo with representative config) ---
        print(f"\n  --- SP / mHC-SP Comparison ({hw_name}) ---")
        # Use a moderate config for SP comparison
        sp_tp, sp_ep, sp_dp, sp_bs = 8, 16, 2, 16
        for combo in iter_result_combos():
            combo_name = combo["result_name"]
            seq_len = combo["seq_len"]
            output_len = combo["output_len"]
            prefix_cache_hit_rate = combo["prefix_cache_hit_rate"]

            # Check memory first
            cfg_test = make_config(base_cfg, tp=sp_tp, ep=sp_ep, dp=sp_dp,
                                   batch_size=sp_bs, seq_len=seq_len,
                                   output_len=output_len, input_len=seq_len,
                                   decode_context_len=seq_len,
                                   prefix_cache_hit_rate=prefix_cache_hit_rate)
            _, _, total_gb, fits = check_memory(cfg_test, hbm_limit)

            if not fits:
                # Try smaller batch
                for smaller_bs in [8, 4, 2, 1]:
                    cfg_test = make_config(base_cfg, tp=sp_tp, ep=sp_ep, dp=sp_dp,
                                           batch_size=smaller_bs * sp_dp, seq_len=seq_len,
                                           output_len=output_len, input_len=seq_len,
                                           decode_context_len=seq_len,
                                           prefix_cache_hit_rate=prefix_cache_hit_rate)
                    _, _, total_gb, fits = check_memory(cfg_test, hbm_limit)
                    if fits:
                        sp_bs_actual = smaller_bs * sp_dp
                        break
                else:
                    print(f"    {combo_name}: Cannot fit in memory for SP comparison, skipping")
                    hw_sp[combo_name] = {
                        "error": "OOM",
                        "prefix_cache_hit_rate": prefix_cache_hit_rate,
                    }
                    continue
            else:
                sp_bs_actual = sp_bs

            sp_result = run_sp_comparison(base_cfg, seq_len, output_len,
                                          sp_tp, sp_ep, sp_dp, sp_bs_actual,
                                          prefix_cache_hit_rate=prefix_cache_hit_rate)
            hw_sp[combo_name] = sp_result
            print(f"    {combo_name} hit={prefix_cache_hit_rate:.0%}: "
                  f"no_SP={sp_result['no_SP']['prefill_time_ms']:.1f}ms, "
                  f"SP={sp_result['SP_only']['prefill_time_ms']:.1f}ms, "
                  f"SP+mHC_SP={sp_result['SP_mHC_SP']['prefill_time_ms']:.1f}ms")

        # --- mHC Optimization Comparison ---
        print(f"\n  --- mHC Optimization Comparison ({hw_name}) ---")
        mhc_tp, mhc_ep, mhc_dp, mhc_bs = 8, 16, 2, 16
        for combo in iter_result_combos():
            combo_name = combo["result_name"]
            seq_len = combo["seq_len"]
            output_len = combo["output_len"]
            prefix_cache_hit_rate = combo["prefix_cache_hit_rate"]

            # Check memory first (use unfused FP32 — largest weight footprint is same)
            cfg_test = make_config(base_cfg, tp=mhc_tp, ep=mhc_ep, dp=mhc_dp,
                                   batch_size=mhc_bs, seq_len=seq_len,
                                   output_len=output_len, input_len=seq_len,
                                   decode_context_len=seq_len,
                                   prefix_cache_hit_rate=prefix_cache_hit_rate)
            _, _, total_gb, fits = check_memory(cfg_test, hbm_limit)

            if not fits:
                for smaller_bs in [8, 4, 2, 1]:
                    cfg_test = make_config(base_cfg, tp=mhc_tp, ep=mhc_ep, dp=mhc_dp,
                                           batch_size=smaller_bs * mhc_dp, seq_len=seq_len,
                                           output_len=output_len, input_len=seq_len,
                                           decode_context_len=seq_len,
                                           prefix_cache_hit_rate=prefix_cache_hit_rate)
                    _, _, total_gb, fits = check_memory(cfg_test, hbm_limit)
                    if fits:
                        mhc_bs_actual = smaller_bs * mhc_dp
                        break
                else:
                    print(f"    {combo_name}: Cannot fit in memory for mHC comparison, skipping")
                    hw_mhc[combo_name] = {
                        "error": "OOM",
                        "prefix_cache_hit_rate": prefix_cache_hit_rate,
                    }
                    continue
            else:
                mhc_bs_actual = mhc_bs

            mhc_result = run_mhc_optimization_comparison(
                base_cfg, seq_len, output_len,
                mhc_tp, mhc_ep, mhc_dp, mhc_bs_actual,
                prefix_cache_hit_rate=prefix_cache_hit_rate)
            hw_mhc[combo_name] = mhc_result

            uf = mhc_result["unfused_fp32"]["prefill_time_ms"]
            ff = mhc_result["fused_fp32"]["prefill_time_ms"]
            fs = mhc_result["fused_fp32_sp"]["prefill_time_ms"]
            fb = mhc_result["fused_bf16_sp"]["prefill_time_ms"]
            print(f"    {combo_name} hit={prefix_cache_hit_rate:.0%}: "
                  f"unfused={uf:.1f}ms, fused={ff:.1f}ms, "
                  f"fused+SP={fs:.1f}ms, fused_bf16+SP={fb:.1f}ms "
                  f"(speedup: {uf/fb:.2f}x)")

        all_search_results[hw_name] = hw_search
        all_pd_results[hw_name] = hw_pd
        all_op_analysis[hw_name] = hw_ops
        all_sp_comparison[hw_name] = hw_sp
        all_mhc_comparison[hw_name] = hw_mhc

    # --- V3 Comparison ---
    print(f"\n{'='*70}")
    print("V3 COMPARISON")
    print(f"{'='*70}")
    v3_comparison = compute_v3_comparison()
    comp = v3_comparison["comparison"]
    print(f"  V4 total params (approx): {comp['v4_total_params_approx']/1e9:.1f}B")
    print(f"  V3 total params (approx): {comp['v3_total_params_approx']/1e9:.1f}B")
    print(f"  V4 KV per token: {comp['v4_kv_per_token_bytes']:.0f} bytes")
    print(f"  V3 KV per token: {comp['v3_kv_per_token_bytes']:.0f} bytes")
    print(f"  KV compression ratio (V3/V4): {comp['kv_compression_ratio']:.1f}x")

    # --- KV Cache Scaling ---
    print(f"\n{'='*70}")
    print("KV CACHE SCALING")
    print(f"{'='*70}")
    all_kv_scaling = {}
    for hw_name in ["910C", "H20"]:
        base_cfg = load_base_config(hw_name)
        kv_results = compute_kv_cache_scaling(base_cfg, hw_name)
        all_kv_scaling[hw_name] = kv_results
        print(f"\n  {hw_name}:")
        for r in kv_results:
            status = "OK" if r["fits_in_hbm"] else "OOM"
            dec_str = f"{r['decode_step_ms']:.2f}ms" if r["decode_step_ms"] else "N/A"
            print(f"    S={r['seq_len']:>6}: KV={r['kv_cache_gb']:.2f}GB "
                  f"noComp={r['no_compression_kv_gb']:.2f}GB "
                  f"ratio={r['compression_ratio']:.1f}x "
                  f"decode={dec_str} [{status}]")

    # --- Attention Analysis ---
    print(f"\n{'='*70}")
    print("ATTENTION ANALYSIS")
    print(f"{'='*70}")
    all_attention_analysis = {}
    for hw_name in ["910C", "H20"]:
        base_cfg = load_base_config(hw_name)
        attn_results = compute_attention_analysis(base_cfg, hw_name)
        all_attention_analysis[hw_name] = attn_results
        print(f"\n  {hw_name} - Layer-type KV breakdown:")
        for sl, data in attn_results["layer_type_kv_breakdown"].items():
            print(f"    S={sl}: full={data['full_attn_pct']:.1f}% "
                  f"C4A={data['c4a_pct']:.1f}% C128A={data['c128a_pct']:.1f}%")

    # --- Hardware Comparison ---
    print(f"\n{'='*70}")
    print("HARDWARE COMPARISON")
    print(f"{'='*70}")

    hw_comparison = {}
    for combo in iter_result_combos():
        cn = combo["result_name"]
        comp = {}
        for phase_sc in ["prefill_latency", "prefill_throughput",
                         "decode_latency", "decode_throughput"]:
            r910 = all_search_results.get("910C", {}).get(cn, {}).get(phase_sc, [])
            rh20 = all_search_results.get("H20", {}).get(cn, {}).get(phase_sc, [])
            if r910 and rh20:
                b910 = r910[0]
                bh20 = rh20[0]
                comp[phase_sc] = {
                    "910C": _extract_metric(b910, phase_sc),
                    "H20": _extract_metric(bh20, phase_sc),
                }
        hw_comparison[cn] = comp

    # --- Save all results ---
    def save_json(name, data):
        path = os.path.join(DATA_DIR, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {path}")

    print(f"\nSaving results to {DATA_DIR}/...")
    save_json("search_results_910C.json", all_search_results.get("910C", {}))
    save_json("search_results_H20.json", all_search_results.get("H20", {}))
    save_json("pd_ratio_analysis.json", all_pd_results)
    save_json("op_analysis.json", all_op_analysis)
    save_json("sp_comparison.json", all_sp_comparison)
    save_json("mhc_optimization_comparison.json", all_mhc_comparison)
    save_json("hardware_comparison.json", hw_comparison)
    save_json("v3_comparison.json", v3_comparison)
    save_json("kv_cache_scaling.json", all_kv_scaling)
    save_json("attention_analysis.json", all_attention_analysis)

    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed:.1f}s")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for hw_name in ["910C", "H20"]:
        print(f"\n  {hw_name}:")
        for combo in iter_result_combos():
            cn = combo["result_name"]
            search = all_search_results.get(hw_name, {}).get(cn, {})
            pd = all_pd_results.get(hw_name, {}).get(cn, {})

            p_lat = search.get("prefill_latency", [{}])
            p_tput = search.get("prefill_throughput", [{}])
            d_lat = search.get("decode_latency", [{}])
            d_tput = search.get("decode_throughput", [{}])

            print(f"    {cn}:")
            if p_lat and p_lat[0]:
                r = p_lat[0]
                print(f"      Prefill Lat:  TP={r['tp']} EP={r['ep']} DP={r['dp']} "
                      f"BS={r['batch_size']} GPUs={r['physical_gpus']} "
                      f"-> {r['prefill_time_ms']:.1f}ms")
            if p_tput and p_tput[0]:
                r = p_tput[0]
                print(f"      Prefill Tput: TP={r['tp']} EP={r['ep']} DP={r['dp']} "
                      f"BS={r['batch_size']} GPUs={r['physical_gpus']} "
                      f"-> {r['prefill_tps_per_gpu']:.2f} tps/gpu")
            if d_lat and d_lat[0]:
                r = d_lat[0]
                print(f"      Decode Lat:   TP={r['tp']} EP={r['ep']} DP={r['dp']} "
                      f"BS={r['batch_size']} GPUs={r['physical_gpus']} "
                      f"-> {r['decode_first_step_ms']:.3f}ms")
            if d_tput and d_tput[0]:
                r = d_tput[0]
                print(f"      Decode Tput:  TP={r['tp']} EP={r['ep']} DP={r['dp']} "
                      f"BS={r['batch_size']} GPUs={r['physical_gpus']} "
                      f"-> {r['decode_tps_per_gpu']:.3f} tps/gpu")
            if isinstance(pd, dict) and "label" in pd:
                print(f"      P/D Ratio:    {pd['label']} -> {pd['total_gpus_min']} GPUs min")

    print(f"\nDone. Results in {DATA_DIR}/")


def _extract_metric(row, phase_sc):
    """Extract the key metric from a result row based on scenario type."""
    result = {
        "tp": row["tp"], "ep": row["ep"], "dp": row["dp"],
        "batch_size": row["batch_size"], "gpus": row["physical_gpus"],
    }
    if "prefill_latency" in phase_sc:
        result["metric"] = round(row["prefill_time_ms"], 2)
        result["unit"] = "ms"
    elif "prefill_throughput" in phase_sc:
        result["metric"] = round(row["prefill_tps_per_gpu"], 2)
        result["unit"] = "tps/gpu"
    elif "decode_latency" in phase_sc:
        result["metric"] = round(row["decode_first_step_ms"], 3)
        result["unit"] = "ms"
    elif "decode_throughput" in phase_sc:
        result["metric"] = round(row["decode_tps_per_gpu"], 3)
        result["unit"] = "tps/gpu"
    return result


if __name__ == "__main__":
    main()
