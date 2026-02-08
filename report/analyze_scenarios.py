#!/usr/bin/env python3
"""Comprehensive analysis: parameter search, P/D ratio, op analysis for DeepSeek V4.

Runs 24 search scenarios (3 combos × 4 phases × 2 hardware platforms),
computes P/D disaggregation ratios, and exports JSON data for report generation.
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
    {"name": "128K_4K", "seq_len": 131072, "output_len": 4096},
    {"name": "256K_4K", "seq_len": 262144, "output_len": 4096},
]

PREFILL_GPU_VALUES = [8, 16, 32, 64, 128, 256]
DECODE_GPU_VALUES = [16, 32]

TP_VALUES = [1, 2, 4, 8, 16, 32, 64]
EP_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DP_VALUES = [1, 2, 4, 8]
BATCH_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

HARDWARE_CONFIGS = {
    "910C": {
        "device": os.path.join(BASE_DIR, "configs", "device_910C.json"),
        "network": os.path.join(BASE_DIR, "configs", "network_910C.json"),
        "hbm_limit_gb": 64,
    },
    "H20": {
        "device": os.path.join(BASE_DIR, "configs", "device_h20.json"),
        "network": os.path.join(BASE_DIR, "configs", "network_h20.json"),
        "hbm_limit_gb": 96,
    },
}


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
                mhc_kernel_fused=None, mhc_fused_bf16=None):
    kwargs = dict(
        tp=tp, ep=ep, dp=dp,
        batch_size=batch_size, seq_len=seq_len,
        output_len=output_len,
        sp=sp, shared_expert_overlapped=shared_expert_overlapped,
        mhc_sp=mhc_sp,
    )
    if mhc_kernel_fused is not None:
        kwargs["mhc_kernel_fused"] = mhc_kernel_fused
    if mhc_fused_bf16 is not None:
        kwargs["mhc_fused_bf16"] = mhc_fused_bf16
    new_rt = replace(base_cfg.rt, **kwargs)
    return replace(base_cfg, rt=new_rt)


def validate_parallelism(tp, ep, model_cfg):
    if model_cfg.num_attention_heads % tp != 0:
        return False
    if model_cfg.n_routed_experts % ep != 0:
        return False
    if model_cfg.o_groups % tp != 0:
        return False
    return True


def check_memory(cfg, hbm_limit_gb):
    wm = weight_memory_per_rank(cfg)
    kv = kv_cache_memory(cfg)
    weight_gb = wm["total"] / 1e9
    kv_gb = kv["total_bytes"] / 1e9
    total_gb = weight_gb + kv_gb
    return weight_gb, kv_gb, total_gb, total_gb <= hbm_limit_gb


def approx_decode(cfg):
    S = cfg.rt.seq_len
    output_len = cfg.rt.output_len
    first = decode_step(S, cfg)
    last = decode_step(S + output_len - 1, cfg)
    approx_total_s = (first.total_time_s + last.total_time_s) / 2 * output_len
    return first.total_time_s, approx_total_s


def evaluate_prefill(cfg):
    prefill = prefill_model(cfg)
    prefill_ms = prefill.total_time_s * 1000
    B = cfg.rt.batch_size // cfg.rt.dp
    S = cfg.rt.seq_len
    physical_gpus = cfg.rt.tp * cfg.rt.dp
    prefill_tps = B * S / prefill.total_time_s if prefill.total_time_s > 0 else 0
    prefill_tps_per_gpu = prefill_tps / physical_gpus if physical_gpus > 0 else 0
    return {
        "prefill_time_ms": prefill_ms,
        "prefill_tps": prefill_tps,
        "prefill_tps_per_gpu": prefill_tps_per_gpu,
    }


def evaluate_decode(cfg):
    output_len = cfg.rt.output_len
    first_step_s, approx_total_s = approx_decode(cfg)
    first_step_ms = first_step_s * 1000
    B = cfg.rt.batch_size // cfg.rt.dp
    physical_gpus = cfg.rt.tp * cfg.rt.dp
    decode_tps = B * output_len / approx_total_s if approx_total_s > 0 else 0
    decode_tps_per_gpu = decode_tps / physical_gpus if physical_gpus > 0 else 0
    return {
        "decode_first_step_ms": first_step_ms,
        "decode_total_ms_approx": approx_total_s * 1000,
        "decode_tps": decode_tps,
        "decode_tps_per_gpu": decode_tps_per_gpu,
    }


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def run_search(base_cfg, phase, seq_len, output_len, hbm_limit_gb, gpu_values):
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

        cfg = make_config(base_cfg, tp=tp, ep=ep, dp=dp,
                          batch_size=batch_size, seq_len=seq_len,
                          output_len=output_len)

        weight_gb, kv_gb, total_gb, fits = check_memory(cfg, hbm_limit_gb)
        if not fits:
            memory_filtered += 1
            continue

        if phase == "prefill":
            metrics = evaluate_prefill(cfg)
        else:
            metrics = evaluate_decode(cfg)
        evaluated += 1

        per_rank_batch = batch_size // dp
        edp = physical_gpus // ep
        row = {
            "tp": tp, "ep": ep, "dp": dp, "edp": edp,
            "batch_size": batch_size, "seq_len": seq_len, "output_len": output_len,
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

def compute_pd_ratio(p_tps_per_gpu, g_p, d_tps_per_gpu, g_d, input_len, output_len):
    """Compute minimum P:D instance ratio for QPS balance.

    Returns (ratio_float, n_p_per_d_ceil).
    """
    p_tps_instance = p_tps_per_gpu * g_p
    d_tps_instance = d_tps_per_gpu * g_d
    if p_tps_instance == 0:
        return float('inf'), 999
    # N_p / N_d >= (D_tps_instance * input_len) / (P_tps_instance * output_len)
    ratio = (d_tps_instance * input_len) / (p_tps_instance * output_len)
    return ratio, math.ceil(ratio)


# ---------------------------------------------------------------------------
# Op Analysis
# ---------------------------------------------------------------------------

OP_CATEGORIES = {
    "Attention Proj": ["q_proj_dq", "q_proj_uq", "k_proj", "v_proj", "wo_a", "wo_b"],
    "Attention Compute": ["attention_full", "attention_comp"],
    "KV Compression": ["kv_compression", "kv_compression_decode"],
    "Lightning Index": ["index_iq_proj", "index_ik_proj", "index_kv_compress",
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

def run_sp_comparison(base_cfg, seq_len, output_len, tp, ep, dp, batch_size):
    """Run 3 SP configs and compare prefill times."""
    results = {}

    configs = [
        ("no_SP", False, False),
        ("SP_only", True, False),
        ("SP_mHC_SP", True, True),
    ]

    for label, sp, mhc_sp in configs:
        cfg = make_config(base_cfg, tp=tp, ep=ep, dp=dp,
                          batch_size=batch_size, seq_len=seq_len,
                          output_len=output_len, sp=sp, mhc_sp=mhc_sp)
        prefill = prefill_model(cfg)
        all_ops, total_time = collect_all_ops_prefill(cfg)
        cat_breakdown = categorize_ops(all_ops)

        results[label] = {
            "prefill_time_ms": total_time * 1000,
            "sp": sp,
            "mhc_sp": mhc_sp,
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


def run_mhc_optimization_comparison(base_cfg, seq_len, output_len, tp, ep, dp, batch_size):
    """Benchmark 4 mHC optimization levels for prefill and decode."""
    results = {}

    for level in MHC_OPTIMIZATION_LEVELS:
        label = level["label"]
        cfg = make_config(base_cfg, tp=tp, ep=ep, dp=dp,
                          batch_size=batch_size, seq_len=seq_len,
                          output_len=output_len, sp=True,
                          mhc_sp=level["mhc_sp"],
                          mhc_kernel_fused=level["mhc_kernel_fused"],
                          mhc_fused_bf16=level["mhc_fused_bf16"])

        # Prefill
        ops_p, time_p = collect_all_ops_prefill(cfg)
        cat_p = categorize_ops(ops_p)

        # Decode (at seq_len)
        ops_d, time_d = collect_all_ops_decode(cfg)
        cat_d = categorize_ops(ops_d)

        # Per-op detail for representative layer (layer 3, C4A)
        lp = prefill_layer(3, cfg)
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
            "prefill_time_ms": time_p * 1000,
            "decode_step_ms": time_d * 1000,
            "prefill_category_breakdown": cat_p,
            "decode_category_breakdown": cat_d,
            "representative_layer_ops": per_op_detail,
        }

    return results


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
        hbm_limit = HARDWARE_CONFIGS[hw_name]["hbm_limit_gb"]

        hw_search = {}
        hw_pd = {}
        hw_ops = {}
        hw_sp = {}
        hw_mhc = {}

        for combo in COMBOS:
            combo_name = combo["name"]
            seq_len = combo["seq_len"]
            output_len = combo["output_len"]

            print(f"\n  --- Combo: {combo_name} (seq={seq_len}, out={output_len}) ---")

            combo_search = {}

            # 4 scenarios per combo
            for phase, scenario in [("prefill", "latency"), ("prefill", "throughput"),
                                     ("decode", "latency"), ("decode", "throughput")]:
                key = f"{phase}_{scenario}"
                gpu_values = PREFILL_GPU_VALUES if phase == "prefill" else DECODE_GPU_VALUES
                print(f"  [{hw_name}] {combo_name} {key}...")

                results = run_search(base_cfg, phase, seq_len, output_len, hbm_limit, gpu_values)
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

                ratio_float, n_p_ceil = compute_pd_ratio(
                    p_tps_gpu, g_p, d_tps_gpu, g_d, seq_len, output_len)

                pd_info = {
                    "prefill_config": {
                        "tp": p_best["tp"], "ep": p_best["ep"], "dp": p_best["dp"],
                        "batch_size": p_best["batch_size"],
                        "gpus": g_p, "tps_per_gpu": round(p_tps_gpu, 2),
                        "tps_instance": round(p_tps_gpu * g_p, 2),
                    },
                    "decode_config": {
                        "tp": d_best["tp"], "ep": d_best["ep"], "dp": d_best["dp"],
                        "batch_size": d_best["batch_size"],
                        "gpus": g_d, "tps_per_gpu": round(d_tps_gpu, 2),
                        "tps_instance": round(d_tps_gpu * g_d, 2),
                    },
                    "input_len": seq_len,
                    "output_len": output_len,
                    "pd_ratio_float": round(ratio_float, 3),
                    "pd_ratio_ceil": n_p_ceil,
                    "total_gpus_min": n_p_ceil * g_p + 1 * g_d,
                    "label": f"{n_p_ceil}P:1D",
                }
                hw_pd[combo_name] = pd_info
                print(f"\n  P/D Ratio ({combo_name}): {pd_info['label']} "
                      f"({n_p_ceil}×{g_p}GPU P + 1×{g_d}GPU D = "
                      f"{pd_info['total_gpus_min']} GPUs)")
            else:
                hw_pd[combo_name] = {"error": "No valid configs for P/D calculation"}
                print(f"\n  P/D Ratio ({combo_name}): SKIPPED (no valid configs)")

            # --- Op Analysis (use best throughput config for detailed analysis) ---
            if p_throughput:
                best_p = p_throughput[0]
                cfg_p = make_config(base_cfg, tp=best_p["tp"], ep=best_p["ep"],
                                    dp=best_p["dp"], batch_size=best_p["batch_size"],
                                    seq_len=seq_len, output_len=output_len)
                ops_p, time_p = collect_all_ops_prefill(cfg_p)
                cat_p = categorize_ops(ops_p)
            else:
                cat_p = {}

            if d_throughput:
                best_d = d_throughput[0]
                cfg_d = make_config(base_cfg, tp=best_d["tp"], ep=best_d["ep"],
                                    dp=best_d["dp"], batch_size=best_d["batch_size"],
                                    seq_len=seq_len, output_len=output_len)
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
        for combo in COMBOS:
            combo_name = combo["name"]
            seq_len = combo["seq_len"]
            output_len = combo["output_len"]

            # Check memory first
            cfg_test = make_config(base_cfg, tp=sp_tp, ep=sp_ep, dp=sp_dp,
                                   batch_size=sp_bs, seq_len=seq_len,
                                   output_len=output_len)
            _, _, total_gb, fits = check_memory(cfg_test, hbm_limit)

            if not fits:
                # Try smaller batch
                for smaller_bs in [8, 4, 2, 1]:
                    cfg_test = make_config(base_cfg, tp=sp_tp, ep=sp_ep, dp=sp_dp,
                                           batch_size=smaller_bs * sp_dp, seq_len=seq_len,
                                           output_len=output_len)
                    _, _, total_gb, fits = check_memory(cfg_test, hbm_limit)
                    if fits:
                        sp_bs_actual = smaller_bs * sp_dp
                        break
                else:
                    print(f"    {combo_name}: Cannot fit in memory for SP comparison, skipping")
                    hw_sp[combo_name] = {"error": "OOM"}
                    continue
            else:
                sp_bs_actual = sp_bs

            sp_result = run_sp_comparison(base_cfg, seq_len, output_len,
                                          sp_tp, sp_ep, sp_dp, sp_bs_actual)
            hw_sp[combo_name] = sp_result
            print(f"    {combo_name}: no_SP={sp_result['no_SP']['prefill_time_ms']:.1f}ms, "
                  f"SP={sp_result['SP_only']['prefill_time_ms']:.1f}ms, "
                  f"SP+mHC_SP={sp_result['SP_mHC_SP']['prefill_time_ms']:.1f}ms")

        # --- mHC Optimization Comparison ---
        print(f"\n  --- mHC Optimization Comparison ({hw_name}) ---")
        mhc_tp, mhc_ep, mhc_dp, mhc_bs = 8, 16, 2, 16
        for combo in COMBOS:
            combo_name = combo["name"]
            seq_len = combo["seq_len"]
            output_len = combo["output_len"]

            # Check memory first (use unfused FP32 — largest weight footprint is same)
            cfg_test = make_config(base_cfg, tp=mhc_tp, ep=mhc_ep, dp=mhc_dp,
                                   batch_size=mhc_bs, seq_len=seq_len,
                                   output_len=output_len)
            _, _, total_gb, fits = check_memory(cfg_test, hbm_limit)

            if not fits:
                for smaller_bs in [8, 4, 2, 1]:
                    cfg_test = make_config(base_cfg, tp=mhc_tp, ep=mhc_ep, dp=mhc_dp,
                                           batch_size=smaller_bs * mhc_dp, seq_len=seq_len,
                                           output_len=output_len)
                    _, _, total_gb, fits = check_memory(cfg_test, hbm_limit)
                    if fits:
                        mhc_bs_actual = smaller_bs * mhc_dp
                        break
                else:
                    print(f"    {combo_name}: Cannot fit in memory for mHC comparison, skipping")
                    hw_mhc[combo_name] = {"error": "OOM"}
                    continue
            else:
                mhc_bs_actual = mhc_bs

            mhc_result = run_mhc_optimization_comparison(
                base_cfg, seq_len, output_len,
                mhc_tp, mhc_ep, mhc_dp, mhc_bs_actual)
            hw_mhc[combo_name] = mhc_result

            uf = mhc_result["unfused_fp32"]["prefill_time_ms"]
            ff = mhc_result["fused_fp32"]["prefill_time_ms"]
            fs = mhc_result["fused_fp32_sp"]["prefill_time_ms"]
            fb = mhc_result["fused_bf16_sp"]["prefill_time_ms"]
            print(f"    {combo_name}: unfused={uf:.1f}ms, fused={ff:.1f}ms, "
                  f"fused+SP={fs:.1f}ms, fused_bf16+SP={fb:.1f}ms "
                  f"(speedup: {uf/fb:.2f}x)")

        all_search_results[hw_name] = hw_search
        all_pd_results[hw_name] = hw_pd
        all_op_analysis[hw_name] = hw_ops
        all_sp_comparison[hw_name] = hw_sp
        all_mhc_comparison[hw_name] = hw_mhc

    # --- Hardware Comparison ---
    print(f"\n{'='*70}")
    print("HARDWARE COMPARISON")
    print(f"{'='*70}")

    hw_comparison = {}
    for combo in COMBOS:
        cn = combo["name"]
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

    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed:.1f}s")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for hw_name in ["910C", "H20"]:
        print(f"\n  {hw_name}:")
        for combo in COMBOS:
            cn = combo["name"]
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
