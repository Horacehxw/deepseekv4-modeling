#!/usr/bin/env python3
"""Grid search for optimal DeepSeek V4 deployment configurations.

Searches 4 scenarios independently:
  - Prefill Latency:    minimize prefill_time_ms
  - Decode Latency:     minimize decode_first_step_ms
  - Prefill Throughput:  maximize prefill_tps_per_gpu
  - Decode Throughput:   maximize decode_tps_per_gpu

GPU formula: physical_gpus = TP * DP, with constraint (TP*DP) % EP == 0 (EDP integer).
"""

import csv
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from itertools import product

# Add parent to path so we can import perf_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perf_model.config import Config
from perf_model.layers import prefill_model, decode_step, decode_model
from perf_model.memory import kv_cache_memory, weight_memory_per_rank


# ---------------------------------------------------------------------------
# Search grids
# ---------------------------------------------------------------------------

TP_VALUES = [1, 2, 4, 8, 16, 32, 64]
EP_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEQ_VALUES = [1024, 2048, 4096, 8192, 16384, 32768, 1_000_000]
DP_VALUES = [1, 2, 4, 8]
BATCH_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
PREFIX_CACHE_HIT_RATE_VALUES = [0.0, 0.9, 0.99]

MIN_GPUS = 8
MAX_GPUS = 64
CSV_COLUMNS = [
    "tp", "ep", "dp", "edp", "batch_size", "seq_len",
    "logical_input_len", "effective_prefill_len", "decode_context_len", "prefix_cache_hit_rate",
    "sp", "shared_expert_overlapped",
    "physical_gpus", "per_rank_batch", "weight_gb", "kv_cache_gb", "hbm_total_gb",
    # prefill phase columns
    "prefill_time_ms", "prefill_tps", "prefill_tps_instance",
    "prefill_tps_per_gpu", "prefill_qps_instance",
    # decode phase columns
    "decode_first_step_ms", "decode_tps", "decode_tps_instance",
    "decode_tps_per_gpu", "decode_qps_instance",
    # decode verification columns
    "decode_total_ms_approx", "decode_total_ms_exact", "approx_error_pct",
]


def load_base_config():
    """Load base config from JSON files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return Config.from_json(
        os.path.join(base_dir, "configs", "device_910C.json"),
        os.path.join(base_dir, "configs", "network_910C.json"),
        os.path.join(base_dir, "configs", "model_deepseekv4.json"),
        os.path.join(base_dir, "configs", "runtime_deepseekv4.json"),
    )


def make_config(base_cfg, tp, ep, dp, batch_size, seq_len, sp, shared_expert_overlapped,
                input_len=None, decode_context_len=None, prefix_cache_hit_rate=0.0):
    """Create a new Config with modified runtime parameters."""
    new_rt = replace(
        base_cfg.rt,
        tp=tp, ep=ep, dp=dp,
        batch_size=batch_size, seq_len=seq_len,
        sp=sp, shared_expert_overlapped=shared_expert_overlapped,
        input_len=input_len, decode_context_len=decode_context_len,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
    )
    return replace(base_cfg, rt=new_rt)


def make_phase_configs(base_cfg, phase, tp, ep, dp, batch_size, seq_len, sp,
                       shared_expert_overlapped, prefix_cache_hit_rate):
    """Build full-context memory cfg and phase-specific compute cfg."""
    full_cfg = make_config(
        base_cfg,
        tp=tp, ep=ep, dp=dp,
        batch_size=batch_size, seq_len=seq_len,
        sp=sp, shared_expert_overlapped=shared_expert_overlapped,
        input_len=seq_len,
        decode_context_len=seq_len,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
    )
    eval_seq_len = (
        full_cfg.rt.effective_prefill_len
        if phase == "prefill"
        else full_cfg.rt.decode_context_len_effective
    )
    eval_cfg = replace(full_cfg, rt=replace(full_cfg.rt, seq_len=eval_seq_len))
    return full_cfg, eval_cfg


def check_memory(cfg):
    """Check if config fits in HBM. Returns (weight_gb, kv_gb, total_gb, fits)."""
    wm = weight_memory_per_rank(cfg)
    kv = kv_cache_memory(cfg)
    weight_gb = (wm["total"] + wm.get("scale_overhead_bytes", 0)) / 1e9
    kv_gb = (kv["total_bytes"] + kv.get("scale_overhead_bytes", 0)) / 1e9
    total_gb = weight_gb + kv_gb
    return weight_gb, kv_gb, total_gb, total_gb <= cfg.hw.usable_hbm_capacity_gb


def validate_parallelism(tp, ep, model_cfg):
    """Check that TP divides Q heads and EP divides experts."""
    if model_cfg.num_attention_heads % tp != 0:
        return False
    if model_cfg.n_routed_experts % ep != 0:
        return False
    if model_cfg.o_groups % tp != 0:
        return False
    return True


def approx_decode(cfg):
    """Approximate total decode time by sampling first and last steps.

    Returns (first_step_s, approx_total_s).
    """
    S = cfg.rt.seq_len
    output_len = cfg.rt.output_len
    first = decode_step(S, cfg)
    last = decode_step(S + output_len - 1, cfg)
    approx_total_s = (first.total_time_s + last.total_time_s) / 2 * output_len
    return first.total_time_s, approx_total_s


def evaluate_prefill(cfg, logical_input_len=None):
    """Run prefill model and return metrics dict."""
    prefill = prefill_model(cfg)
    prefill_ms = prefill.total_time_s * 1000
    logical_len = logical_input_len if logical_input_len is not None else cfg.rt.request_input_len
    physical_gpus = cfg.rt.tp * cfg.rt.dp
    prefill_tps_instance = (
        cfg.rt.batch_size * logical_len / prefill.total_time_s
        if prefill.total_time_s > 0 else 0
    )
    prefill_tps_per_gpu = prefill_tps_instance / physical_gpus if physical_gpus > 0 else 0
    prefill_qps_instance = (
        cfg.rt.batch_size / prefill.total_time_s
        if prefill.total_time_s > 0 else 0
    )
    return {
        "prefill_time_ms": prefill_ms,
        "prefill_tps": prefill_tps_instance,
        "prefill_tps_instance": prefill_tps_instance,
        "prefill_tps_per_gpu": prefill_tps_per_gpu,
        "prefill_qps_instance": prefill_qps_instance,
    }


def evaluate_decode(cfg):
    """Run decode step (first step) and approximate total decode. Return metrics dict."""
    output_len = cfg.rt.output_len
    first_step_s, approx_total_s = approx_decode(cfg)
    first_step_ms = first_step_s * 1000
    physical_gpus = cfg.rt.tp * cfg.rt.dp
    decode_tps_instance = (
        cfg.rt.batch_size * output_len / approx_total_s
        if approx_total_s > 0 else 0
    )
    decode_tps_per_gpu = decode_tps_instance / physical_gpus if physical_gpus > 0 else 0
    decode_qps_instance = (
        cfg.rt.batch_size / approx_total_s
        if approx_total_s > 0 else 0
    )
    return {
        "decode_first_step_ms": first_step_ms,
        "decode_total_ms_approx": approx_total_s * 1000,
        "decode_tps": decode_tps_instance,
        "decode_tps_instance": decode_tps_instance,
        "decode_tps_per_gpu": decode_tps_per_gpu,
        "decode_qps_instance": decode_qps_instance,
    }


def verify_decode_top_n(results, base_cfg, top_n=10):
    """Run full decode_model on top-N decode candidates for exact total decode time."""
    verified_count = 0
    for row in results[:top_n]:
        cfg = make_config(
            base_cfg,
            tp=int(row["tp"]), ep=int(row["ep"]),
            dp=int(row["dp"]), batch_size=int(row["batch_size"]),
            seq_len=int(row["seq_len"]),
            sp=row["sp"] if isinstance(row["sp"], bool) else row["sp"] == "True",
            shared_expert_overlapped=(
                row["shared_expert_overlapped"]
                if isinstance(row["shared_expert_overlapped"], bool)
                else row["shared_expert_overlapped"] == "True"
            ),
            input_len=int(row.get("logical_input_len", row["seq_len"])),
            decode_context_len=int(row.get("decode_context_len", row["seq_len"])),
            prefix_cache_hit_rate=float(row.get("prefix_cache_hit_rate", 0.0)),
        )
        decode_total = decode_model(cfg)
        decode_exact_ms = decode_total.total_time_s * 1000
        row["decode_total_ms_exact"] = f"{decode_exact_ms:.3f}"

        approx_ms = float(row["decode_total_ms_approx"])
        if decode_exact_ms > 0:
            error_pct = abs(approx_ms - decode_exact_ms) / decode_exact_ms * 100
            row["approx_error_pct"] = f"{error_pct:.2f}"
        else:
            row["approx_error_pct"] = "0.00"

        # Recompute exact instance throughput.
        output_len = cfg.rt.output_len
        physical_gpus = int(row["physical_gpus"])
        exact_tps_instance = (
            int(row["batch_size"]) * output_len / decode_total.total_time_s
            if decode_total.total_time_s > 0 else 0
        )
        exact_tps_per_gpu = exact_tps_instance / physical_gpus if physical_gpus > 0 else 0
        exact_qps_instance = (
            int(row["batch_size"]) / decode_total.total_time_s
            if decode_total.total_time_s > 0 else 0
        )
        row["decode_tps"] = f"{exact_tps_instance:.3f}"
        row["decode_tps_instance"] = f"{exact_tps_instance:.3f}"
        row["decode_tps_per_gpu"] = f"{exact_tps_per_gpu:.3f}"
        row["decode_qps_instance"] = f"{exact_qps_instance:.3f}"

        verified_count += 1
        print(f"  Verified {verified_count}/{top_n}: "
              f"tp={row['tp']} ep={row['ep']} dp={row['dp']} bs={row['batch_size']} "
              f"seq={row['seq_len']} | approx={approx_ms:.1f}ms exact={decode_exact_ms:.1f}ms "
              f"err={row['approx_error_pct']}%")
    return results


def write_csv(filepath, rows):
    """Write results to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_search(base_cfg, phase, scenario):
    """Unified grid search.

    phase:    "prefill" or "decode"
    scenario: "latency" or "throughput"

    Latency grid: varies tp, ep, dp, batch_size, seq_len, sp, shared_expert_overlapped
    Throughput grid: same but fixes sp=True, shared_expert_overlapped=True

    Returns list of result dicts sorted by the appropriate metric.
    """
    label = f"{phase.upper()} {scenario.upper()}"
    print("=" * 70)
    print(f"SCENARIO: {label}")
    print("=" * 70)

    if scenario == "latency":
        sp_values = [True, False]
        ovl_values = [True, False]
    else:
        sp_values = [True]
        ovl_values = [True]

    results = []
    total_combos = 0
    memory_filtered = 0
    evaluated = 0

    for tp, ep, dp, batch_size, seq_len, prefix_cache_hit_rate, sp, shared_ovl in product(
        TP_VALUES, EP_VALUES, DP_VALUES, BATCH_VALUES, SEQ_VALUES,
        PREFIX_CACHE_HIT_RATE_VALUES,
        sp_values, ovl_values,
    ):
        total_combos += 1
        physical_gpus = tp * dp
        if physical_gpus < MIN_GPUS or physical_gpus > MAX_GPUS:
            continue
        # EDP must be a positive integer
        if physical_gpus % ep != 0:
            continue
        edp = physical_gpus // ep
        if batch_size % dp != 0:
            continue
        if not validate_parallelism(tp, ep, base_cfg.model):
            continue

        full_cfg, eval_cfg = make_phase_configs(
            base_cfg, phase=phase, tp=tp, ep=ep, dp=dp, batch_size=batch_size,
            seq_len=seq_len, sp=sp, shared_expert_overlapped=shared_ovl,
            prefix_cache_hit_rate=prefix_cache_hit_rate,
        )

        weight_gb, kv_gb, total_gb, fits = check_memory(full_cfg)
        if not fits:
            memory_filtered += 1
            continue

        # Evaluate only the relevant phase
        if phase == "prefill":
            metrics = evaluate_prefill(eval_cfg, logical_input_len=full_cfg.rt.request_input_len)
        else:
            metrics = evaluate_decode(eval_cfg)
        evaluated += 1

        per_rank_batch = batch_size // dp
        row = {
            "tp": tp, "ep": ep, "dp": dp, "edp": edp,
            "batch_size": batch_size, "seq_len": seq_len,
            "logical_input_len": full_cfg.rt.request_input_len,
            "effective_prefill_len": full_cfg.rt.effective_prefill_len,
            "decode_context_len": full_cfg.rt.decode_context_len_effective,
            "prefix_cache_hit_rate": prefix_cache_hit_rate,
            "sp": sp, "shared_expert_overlapped": shared_ovl,
            "physical_gpus": physical_gpus, "per_rank_batch": per_rank_batch,
            "weight_gb": f"{weight_gb:.3f}", "kv_cache_gb": f"{kv_gb:.3f}",
            "hbm_total_gb": f"{total_gb:.3f}",
            # Phase columns not computed are left blank
            "prefill_time_ms": "", "prefill_tps": "", "prefill_tps_instance": "",
            "prefill_tps_per_gpu": "", "prefill_qps_instance": "",
            "decode_first_step_ms": "", "decode_tps": "", "decode_tps_instance": "",
            "decode_tps_per_gpu": "", "decode_qps_instance": "",
            "decode_total_ms_approx": "", "decode_total_ms_exact": "", "approx_error_pct": "",
        }
        # Fill in computed metrics
        for k, v in metrics.items():
            row[k] = f"{v:.3f}"

        results.append(row)

        if evaluated % 200 == 0:
            print(f"  Evaluated {evaluated} configs...")

    print(f"  Total combos: {total_combos}, memory-filtered: {memory_filtered}, evaluated: {evaluated}")

    # Sort by the appropriate metric
    if phase == "prefill" and scenario == "latency":
        results.sort(key=lambda r: float(r["prefill_time_ms"]))
    elif phase == "decode" and scenario == "latency":
        results.sort(key=lambda r: float(r["decode_first_step_ms"]))
    elif phase == "prefill" and scenario == "throughput":
        results.sort(key=lambda r: float(r["prefill_tps_per_gpu"]), reverse=True)
    elif phase == "decode" and scenario == "throughput":
        results.sort(key=lambda r: float(r["decode_tps_per_gpu"]), reverse=True)

    return results


def print_top5(results, phase, scenario):
    """Print a top-5 summary table to console."""
    print(f"\n  Top-5 {phase.capitalize()} {scenario.capitalize()} Configs:")

    if phase == "prefill" and scenario == "latency":
        print(f"  {'Rank':>4s} | {'TP':>3s} | {'EP':>4s} | {'DP':>2s} | {'BS':>3s} | {'SeqLen':>6s} | "
              f"{'SP':>5s} | {'Ovl':>5s} | {'GPUs':>4s} | {'EDP':>3s} | {'Prefill(ms)':>11s}")
        print("  " + "-" * 80)
        for i, r in enumerate(results[:5]):
            print(f"  {i+1:>4d} | {int(r['tp']):>3d} | {int(r['ep']):>4d} | {int(r['dp']):>2d} | "
                  f"{int(r['batch_size']):>3d} | {int(r['seq_len']):>6d} | "
                  f"{str(r['sp']):>5s} | {str(r['shared_expert_overlapped']):>5s} | "
                  f"{int(r['physical_gpus']):>4d} | {int(r['edp']):>3d} | "
                  f"{float(r['prefill_time_ms']):>11.1f}")

    elif phase == "decode" and scenario == "latency":
        print(f"  {'Rank':>4s} | {'TP':>3s} | {'EP':>4s} | {'DP':>2s} | {'BS':>3s} | {'SeqLen':>6s} | "
              f"{'SP':>5s} | {'Ovl':>5s} | {'GPUs':>4s} | {'EDP':>3s} | {'1st Step(ms)':>12s}")
        print("  " + "-" * 85)
        for i, r in enumerate(results[:5]):
            print(f"  {i+1:>4d} | {int(r['tp']):>3d} | {int(r['ep']):>4d} | {int(r['dp']):>2d} | "
                  f"{int(r['batch_size']):>3d} | {int(r['seq_len']):>6d} | "
                  f"{str(r['sp']):>5s} | {str(r['shared_expert_overlapped']):>5s} | "
                  f"{int(r['physical_gpus']):>4d} | {int(r['edp']):>3d} | "
                  f"{float(r['decode_first_step_ms']):>12.3f}")

    elif phase == "prefill" and scenario == "throughput":
        print(f"  {'Rank':>4s} | {'TP':>3s} | {'EP':>4s} | {'DP':>2s} | {'BS':>3s} | {'SeqLen':>6s} | "
              f"{'GPUs':>4s} | {'EDP':>3s} | {'TPS/GPU':>10s}")
        print("  " + "-" * 65)
        for i, r in enumerate(results[:5]):
            print(f"  {i+1:>4d} | {int(r['tp']):>3d} | {int(r['ep']):>4d} | {int(r['dp']):>2d} | "
                  f"{int(r['batch_size']):>3d} | {int(r['seq_len']):>6d} | "
                  f"{int(r['physical_gpus']):>4d} | {int(r['edp']):>3d} | "
                  f"{float(r['prefill_tps_per_gpu']):>10.2f}")

    elif phase == "decode" and scenario == "throughput":
        print(f"  {'Rank':>4s} | {'TP':>3s} | {'EP':>4s} | {'DP':>2s} | {'BS':>3s} | {'SeqLen':>6s} | "
              f"{'GPUs':>4s} | {'EDP':>3s} | {'TPS/GPU':>10s}")
        print("  " + "-" * 65)
        for i, r in enumerate(results[:5]):
            print(f"  {i+1:>4d} | {int(r['tp']):>3d} | {int(r['ep']):>4d} | {int(r['dp']):>2d} | "
                  f"{int(r['batch_size']):>3d} | {int(r['seq_len']):>6d} | "
                  f"{int(r['physical_gpus']):>4d} | {int(r['edp']):>3d} | "
                  f"{float(r['decode_tps_per_gpu']):>10.3f}")

    print()


def main():
    start_time = time.time()
    base_cfg = load_base_config()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")
    print()

    scenarios = [
        ("prefill", "latency"),
        ("decode",  "latency"),
        ("prefill", "throughput"),
        ("decode",  "throughput"),
    ]

    all_results = {}
    search_times = {}

    for phase, scenario in scenarios:
        key = f"{phase}_{scenario}"
        t0 = time.time()
        results = run_search(base_cfg, phase, scenario)
        elapsed = time.time() - t0
        search_times[key] = elapsed
        print(f"  Search time: {elapsed:.1f}s, {len(results)} valid configs")

        # Verify decode throughput top-10 with full decode_model
        if phase == "decode" and scenario == "throughput":
            print("\n  Verifying decode throughput top-10 with full decode_model...")
            results = verify_decode_top_n(results, base_cfg, top_n=10)

        print_top5(results, phase, scenario)

        all_results[key] = results

        # Write CSVs
        write_csv(os.path.join(results_dir, f"{key}_all.csv"), results)
        write_csv(os.path.join(results_dir, f"{key}_top10.csv"), results[:10])

    total_time = time.time() - start_time

    # Save metadata
    meta = {
        "timestamp": timestamp,
        "total_time_s": round(total_time, 1),
        "search_times_s": {k: round(v, 1) for k, v in search_times.items()},
        "valid_configs": {k: len(v) for k, v in all_results.items()},
        "grids": {
            "tp": TP_VALUES,
            "ep": EP_VALUES,
            "dp": DP_VALUES,
            "batch_size": BATCH_VALUES,
            "seq_len": SEQ_VALUES,
            "prefix_cache_hit_rate": PREFIX_CACHE_HIT_RATE_VALUES,
        },
        "constraints": {
            "min_gpus": MIN_GPUS,
            "max_gpus": MAX_GPUS,
            "hbm_capacity_gb": base_cfg.hw.hbm_capacity_gb,
            "hbm_reserved_pct": base_cfg.hw.hbm_reserved_pct,
            "usable_hbm_capacity_gb": base_cfg.hw.usable_hbm_capacity_gb,
            "gpu_formula": "physical_gpus = tp * dp",
            "edp_constraint": "(tp * dp) % ep == 0",
        },
        "scenarios": [f"{p}_{s}" for p, s in scenarios],
    }
    with open(os.path.join(results_dir, "search_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Total search time: {total_time:.1f}s")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
