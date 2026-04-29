"""Formatting, printing, and CSV export functions."""

import csv
import json
import dataclasses
from typing import List, Optional

from .config import Config
from .roofline import OpProfile
from .layers import LayerProfile, PhaseProfile
from .memory import kv_cache_memory, weight_memory_per_rank


def fmt_bytes(b: float) -> str:
    """Format bytes to human-readable."""
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    return f"{b:.0f} B"


def fmt_ms(s: float) -> str:
    """Format seconds to ms."""
    return f"{s * 1000:.3f}"


def print_op_table(ops: List[OpProfile], indent: str = "  "):
    """Print per-op table with roofline breakdown."""
    header = f"{indent}{'Operation':<25s} | {'Cube(ms)':>9s} | {'Vec(ms)':>9s} | {'Mem(ms)':>9s} | {'Comm(ms)':>9s} | {'Total(ms)':>10s} | {'Bound':<5s}"
    sep = indent + "-" * (len(header) - len(indent))
    print(header)
    print(sep)
    for op in ops:
        print(f"{indent}{op.name:<25s} | {fmt_ms(op.cube_time_s):>9s} | {fmt_ms(op.vec_time_s):>9s} | {fmt_ms(op.mem_time_s):>9s} | {fmt_ms(op.comm_time_s):>9s} | {fmt_ms(op.time_s):>10s} | {op.bottleneck:<5s}")


def print_layer_summary(layer_profiles: List[LayerProfile], indent: str = "  "):
    """Print summary table of all layers."""
    header = (f"{indent}{'Layer':>5s} | {'Ratio':>5s} | {'Comp(ms)':>10s} | "
              f"{'Comm(ms)':>10s} | {'Total(ms)':>10s} | {'Comm%':>6s} | {'Bound':<5s}")
    sep = indent + "-" * (len(header) - len(indent))
    print(header)
    print(sep)
    for lp in layer_profiles:
        t = lp.total
        comp = t.time_s - t.comm_time_s
        comm_pct = t.comm_time_s / t.time_s * 100 if t.time_s > 0 else 0.0
        print(f"{indent}{lp.layer_idx:>5d} | {lp.ratio:>5d} | {fmt_ms(comp):>10s} | "
              f"{fmt_ms(t.comm_time_s):>10s} | {fmt_ms(t.time_s):>10s} | {comm_pct:>5.1f}% | {t.bottleneck:<5s}")


def print_comm_analysis(phase: PhaseProfile, indent: str = "  "):
    """Print communication vs computation breakdown for a phase."""
    # Known comm op names
    comm_op_names = {
        "attn_tp_allreduce": "TP AllReduce",
        "moe_ep_dispatch": "EP Dispatch",
        "moe_ep_combine": "EP Combine",
        "index_score_ar": "Index Score AR",
    }

    # Sum comm times by op name across all layers + extra ops
    comm_by_type = {k: 0.0 for k in comm_op_names}
    total_time = phase.total_time_s
    total_comm = 0.0

    for lp in phase.layer_profiles:
        for op in lp.ops:
            if op.name in comm_op_names:
                comm_by_type[op.name] += op.comm_time_s
                total_comm += op.comm_time_s
    for op in phase.extra_ops:
        if op.name in comm_op_names:
            comm_by_type[op.name] += op.comm_time_s
            total_comm += op.comm_time_s

    total_compute = total_time - total_comm
    comp_pct = total_compute / total_time * 100 if total_time > 0 else 0.0
    comm_pct = total_comm / total_time * 100 if total_time > 0 else 0.0

    print(f"{indent}Communication vs Computation:")
    print(f"{indent}  Compute:           {fmt_ms(total_compute):>10s} ms  ({comp_pct:.1f}%)")
    print(f"{indent}  Communication:     {fmt_ms(total_comm):>10s} ms  ({comm_pct:.1f}%)")
    for op_name, label in comm_op_names.items():
        t = comm_by_type[op_name]
        if t > 0:
            pct = t / total_time * 100
            print(f"{indent}    {label + ':':.<19s}{fmt_ms(t):>10s} ms  ({pct:.1f}%)")
    print(f"{indent}  Total:             {fmt_ms(total_time):>10s} ms")
    print()


def print_config_summary(cfg: Config):
    """Print configuration summary."""
    print("=" * 80)
    print("DeepSeek V4 Inference Performance Model")
    print("=" * 80)
    print()
    print("Hardware (per die):")
    print(f"  BF16 Cube TFLOPS:    {cfg.hw.cube_tflops}")
    print(f"  BF16 Vector TFLOPS:  {cfg.hw.vec_tflops}")
    print(f"  HBM Capacity:        {cfg.hw.hbm_capacity_gb} GB")
    print(f"  HBM Reserved:        {cfg.hw.hbm_reserved_pct:.1f}%")
    print(f"  HBM Usable:          {cfg.hw.usable_hbm_capacity_gb:.2f} GB")
    print(f"  HBM Bandwidth:       {cfg.hw.hbm_bandwidth_gbps} GB/s")
    print(f"  Cube Utilization:    {cfg.hw.effective_cube_utilization:.0%}")
    print(f"  Vector Utilization:  {cfg.hw.effective_vec_utilization:.0%}")
    print(f"  HBM BW Utilization:  {cfg.hw.hbm_bw_utilization:.0%}")
    print()
    print("Network:")
    print(f"  TP Bandwidth:        {cfg.net.tp_bandwidth_gbps} GB/s (bidirectional)")
    print(f"  EP Bandwidth:        {cfg.net.ep_bandwidth_gbps} GB/s (bidirectional)")
    print(f"  Latency:             {cfg.net.latency_us} us")
    print(f"  BW Utilization:      {cfg.net.bandwidth_utilization:.0%}")
    print()
    print("Model:")
    print(f"  Hidden size:         {cfg.model.hidden_size}")
    print(f"  Layers:              {cfg.model.num_hidden_layers}")
    print(f"  Q heads:             {cfg.model.num_attention_heads}, KV heads: {cfg.model.num_kv_heads} (MQA)")
    print(f"  Head dim:            {cfg.model.head_dim}, RoPE dim: {cfg.model.rope_head_dim}")
    print(f"  KV dim (shared):     {cfg.model.kv_dim}")
    print(f"  Q LoRA rank:         {cfg.model.q_lora_rank}, O LoRA rank: {cfg.model.o_lora_rank}")
    print(f"  O groups:            {cfg.model.o_groups}, O mid dim: {cfg.model.o_mid_dim}")
    print(f"  Index heads:         {cfg.model.index_n_heads}, dim: {cfg.model.index_head_dim}, topK: {cfg.model.index_topk}")
    print(f"  SWA window:          {cfg.model.window_size}")
    print(f"  Compress C_kv:       {cfg.model.compress_c_kv}")
    print(f"  HC mult:             {cfg.model.hc_mult}")
    print(f"  MoE: {cfg.model.n_routed_experts} routed experts, top-{cfg.model.num_experts_per_tok}, "
          f"{cfg.model.n_shared_experts} shared, inter={cfg.model.moe_inter_dim}")
    print(f"  Hash routing layers: {cfg.model.n_hash_layers}")
    print()

    # Count layer types
    ratios = cfg.model.compress_ratios
    r1 = sum(1 for r in ratios if r == 1)
    r4 = sum(1 for r in ratios if r == 4)
    r128 = sum(1 for r in ratios if r == 128)
    print(f"  Layer types:         {r1} SWA (ratio=1), {r4} C4A, {r128} C128A")
    print()

    DP = cfg.rt.dp
    TP = cfg.rt.tp
    EP = cfg.rt.ep
    total_gpus = TP * DP
    edp = total_gpus / EP
    per_rank_batch = cfg.rt.batch_size // DP

    print("Runtime:")
    print(f"  Seq len:             {cfg.rt.seq_len}")
    print(f"  Batch size (global): {cfg.rt.batch_size}")
    print(f"  Per-rank batch:      {per_rank_batch}")
    edp_str = str(int(edp)) if edp == int(edp) else f"{edp:.2f}"
    print(f"  TP: {TP}, DP: {DP}, EP: {EP}, EDP: {edp_str}, SP: {cfg.rt.sp}")
    print(f"  Total GPUs:          {total_gpus}")
    print(f"  MoE load balance:    {cfg.rt.moe_load_balance_factor}")
    print(f"  Output len:          {cfg.rt.output_len}")
    print(f"  Shared expert overlap: {cfg.rt.shared_expert_overlapped}")
    print()


def find_representative_layers(cfg: Config) -> List[int]:
    """Find representative layer indices: one ratio=1, one C4A, one C128A."""
    reps = []
    seen = set()
    for i, r in enumerate(cfg.model.compress_ratios):
        if r not in seen:
            reps.append(i)
            seen.add(r)
        if len(seen) >= 3:
            break
    return reps


def print_phase_report(phase: PhaseProfile, cfg: Config, detailed_step: Optional[PhaseProfile] = None):
    """Print full phase report."""
    is_decode = phase.phase.startswith("decode")
    phase_name = "DECODE" if is_decode else "PREFILL"

    print("=" * 80)
    print(f"  {phase_name} PHASE")
    print("=" * 80)
    print()

    # Use detailed_step for per-op breakdown if provided
    detail_phase = detailed_step if detailed_step else phase

    # Representative layers detailed
    rep_indices = find_representative_layers(cfg)
    for idx in rep_indices:
        if idx >= len(detail_phase.layer_profiles):
            continue
        lp = detail_phase.layer_profiles[idx]
        ratio = lp.ratio
        label = "SWA" if ratio == 1 else f"C{ratio}A"
        print(f"  Layer {idx} ({label}, ratio={ratio}):")
        print_op_table(lp.ops, indent="    ")
        print(f"    {'TOTAL':<25s} | {fmt_ms(lp.total.cube_time_s):>9s} | {fmt_ms(lp.total.vec_time_s):>9s} | {fmt_ms(lp.total.mem_time_s):>9s} | {fmt_ms(lp.total.comm_time_s):>9s} | {fmt_ms(lp.total.time_s):>10s} | {lp.total.bottleneck:<5s}")
        print()

    # Layer summary table
    print("  Layer Summary:")
    print_layer_summary(detail_phase.layer_profiles, indent="    ")
    print()

    # Extra ops
    if detail_phase.extra_ops:
        print("  Non-layer ops:")
        print_op_table(detail_phase.extra_ops, indent="    ")
        extra_time = sum(op.time_s for op in detail_phase.extra_ops)
        print(f"    Extra ops total: {fmt_ms(extra_time)} ms")
        print()

    # Communication vs Computation
    print_comm_analysis(detail_phase, indent="  ")

    # Totals
    if is_decode and detailed_step:
        step_time = detailed_step.total_time_s
        print(f"  Single decode step (context={cfg.rt.seq_len}): {fmt_ms(step_time)} ms")
        print(f"  Total decode ({cfg.rt.output_len} tokens): {fmt_ms(phase.total_time_s)} ms")
        print(f"  Avg tokens/s: {phase.total_tokens / phase.total_time_s:.1f}")
    else:
        print(f"  Total {phase_name.lower()} time: {fmt_ms(phase.total_time_s)} ms")
        print(f"  Tokens processed: {phase.total_tokens}")
        if phase.total_time_s > 0:
            print(f"  Throughput: {phase.total_tokens / phase.total_time_s:.1f} tokens/s")
    print()


def print_memory_report(cfg: Config):
    """Print memory analysis."""
    print("=" * 80)
    print("  MEMORY ANALYSIS")
    print("=" * 80)
    print()

    # KV Cache
    per_rank_batch = cfg.rt.batch_size // cfg.rt.dp
    kv = kv_cache_memory(cfg)
    print(f"  KV Cache (per-rank B={per_rank_batch}, S={cfg.rt.seq_len}):")
    # Show a few representative layers
    for i in find_representative_layers(cfg):
        info = kv["layers"][i]
        print(f"    Layer {i} ({info['type']}): {fmt_bytes(info['bytes'])}", end="")
        if "comp_bytes" in info:
            parts = f"comp={fmt_bytes(info['comp_bytes'])}, swa={fmt_bytes(info['swa_bytes'])}"
            if "idx_bytes" in info:
                parts += f", idx={fmt_bytes(info['idx_bytes'])}"
            print(f"  ({parts})", end="")
        print()
    print(f"    Total KV cache: {fmt_bytes(kv['total_bytes'])}")
    print()

    # Weight Memory
    wm = weight_memory_per_rank(cfg)
    print("  Weight Memory (per rank):")
    print(f"    Attention (all layers): {fmt_bytes(wm['total_attn'])}")
    print(f"      Per layer (base):    {fmt_bytes(wm['attn_per_layer'])}")
    print(f"      Per layer (index):   {fmt_bytes(wm['index_per_layer'])} (only index layers)")
    print(f"    MoE (all layers):      {fmt_bytes(wm['total_moe'])}")
    print(f"      Per layer:           {fmt_bytes(wm['moe_per_layer'])}")
    print(f"    Other (mHC+norm+emb):  {fmt_bytes(wm['total_other'])}")
    print(f"    Total weights:         {fmt_bytes(wm['total'])}")
    print()

    # Total HBM
    total_hbm = (wm["total"] + wm.get("scale_overhead_bytes", 0)
                 + kv["total_bytes"] + kv.get("scale_overhead_bytes", 0))
    capacity = cfg.hw.hbm_capacity_gb * 1e9
    usable_capacity = cfg.hw.usable_hbm_capacity_gb * 1e9
    print(f"  Total HBM Usage:         {fmt_bytes(total_hbm)}")
    print(f"  HBM Capacity:            {fmt_bytes(capacity)}")
    print(f"  HBM Reserved:            {cfg.hw.hbm_reserved_pct:.1f}%")
    print(f"  Usable HBM Capacity:     {fmt_bytes(usable_capacity)}")
    pct = total_hbm / usable_capacity * 100
    print(f"  Usable Utilization:      {pct:.1f}%")
    if total_hbm > usable_capacity:
        print(f"  WARNING: Exceeds usable HBM capacity by {fmt_bytes(total_hbm - usable_capacity)}!")
    print()


# =============================================================================
# CSV Export Functions
# =============================================================================

def export_ops_csv(filepath: str, phase_profile: PhaseProfile):
    """Export per-op breakdown for all layers + extra ops to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer_idx", "ratio", "op_name", "flops", "vec_ops", "mem_bytes",
            "comm_bytes", "cube_time_ms", "vec_time_ms", "mem_time_ms",
            "comm_time_ms", "total_time_ms", "bottleneck"
        ])
        for lp in phase_profile.layer_profiles:
            for op in lp.ops:
                writer.writerow([
                    lp.layer_idx, lp.ratio, op.name,
                    f"{op.flops:.0f}", f"{op.vec_ops:.0f}", f"{op.mem_bytes:.0f}",
                    f"{op.comm_bytes:.0f}",
                    f"{op.cube_time_s * 1000:.6f}", f"{op.vec_time_s * 1000:.6f}",
                    f"{op.mem_time_s * 1000:.6f}", f"{op.comm_time_s * 1000:.6f}",
                    f"{op.time_s * 1000:.6f}", op.bottleneck
                ])
        for op in phase_profile.extra_ops:
            writer.writerow([
                "extra", "", op.name,
                f"{op.flops:.0f}", f"{op.vec_ops:.0f}", f"{op.mem_bytes:.0f}",
                f"{op.comm_bytes:.0f}",
                f"{op.cube_time_s * 1000:.6f}", f"{op.vec_time_s * 1000:.6f}",
                f"{op.mem_time_s * 1000:.6f}", f"{op.comm_time_s * 1000:.6f}",
                f"{op.time_s * 1000:.6f}", op.bottleneck
            ])


def export_layer_summary_csv(filepath: str, prefill: PhaseProfile,
                              decode_step_profile: PhaseProfile):
    """Export layer-level summary for both phases."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "layer_idx", "ratio", "total_time_ms",
            "cube_time_ms", "vec_time_ms", "mem_time_ms", "comm_time_ms",
            "comp_time_ms", "comm_pct",
            "bottleneck"
        ])
        for lp in prefill.layer_profiles:
            t = lp.total
            comp = t.time_s - t.comm_time_s
            cpct = t.comm_time_s / t.time_s * 100 if t.time_s > 0 else 0.0
            writer.writerow([
                "prefill", lp.layer_idx, lp.ratio,
                f"{t.time_s * 1000:.6f}", f"{t.cube_time_s * 1000:.6f}",
                f"{t.vec_time_s * 1000:.6f}", f"{t.mem_time_s * 1000:.6f}",
                f"{t.comm_time_s * 1000:.6f}", f"{comp * 1000:.6f}",
                f"{cpct:.2f}", t.bottleneck
            ])
        for lp in decode_step_profile.layer_profiles:
            t = lp.total
            comp = t.time_s - t.comm_time_s
            cpct = t.comm_time_s / t.time_s * 100 if t.time_s > 0 else 0.0
            writer.writerow([
                "decode", lp.layer_idx, lp.ratio,
                f"{t.time_s * 1000:.6f}", f"{t.cube_time_s * 1000:.6f}",
                f"{t.vec_time_s * 1000:.6f}", f"{t.mem_time_s * 1000:.6f}",
                f"{t.comm_time_s * 1000:.6f}", f"{comp * 1000:.6f}",
                f"{cpct:.2f}", t.bottleneck
            ])


def export_memory_csv(filepath: str, cfg: Config):
    """Export KV cache + weight memory breakdown."""
    kv = kv_cache_memory(cfg)
    wm = weight_memory_per_rank(cfg)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "component", "bytes", "human_readable"])

        # KV cache per representative layer type
        for i in find_representative_layers(cfg):
            info = kv["layers"][i]
            writer.writerow([
                "kv_cache", f"layer_{i}_{info['type']}",
                f"{info['bytes']:.0f}", fmt_bytes(info['bytes'])
            ])
        writer.writerow([
            "kv_cache", "total",
            f"{kv['total_bytes']:.0f}", fmt_bytes(kv['total_bytes'])
        ])

        # Weight memory
        writer.writerow(["weights", "attn_per_layer", f"{wm['attn_per_layer']:.0f}", fmt_bytes(wm['attn_per_layer'])])
        writer.writerow(["weights", "index_per_layer", f"{wm['index_per_layer']:.0f}", fmt_bytes(wm['index_per_layer'])])
        writer.writerow(["weights", "moe_per_layer", f"{wm['moe_per_layer']:.0f}", fmt_bytes(wm['moe_per_layer'])])
        writer.writerow(["weights", "total_attn", f"{wm['total_attn']:.0f}", fmt_bytes(wm['total_attn'])])
        writer.writerow(["weights", "total_moe", f"{wm['total_moe']:.0f}", fmt_bytes(wm['total_moe'])])
        writer.writerow(["weights", "total_other", f"{wm['total_other']:.0f}", fmt_bytes(wm['total_other'])])
        writer.writerow(["weights", "total", f"{wm['total']:.0f}", fmt_bytes(wm['total'])])

        # Total HBM
        total_hbm = (wm["total"] + wm.get("scale_overhead_bytes", 0)
                 + kv["total_bytes"] + kv.get("scale_overhead_bytes", 0))
        writer.writerow(["total", "hbm_usage", f"{total_hbm:.0f}", fmt_bytes(total_hbm)])
        capacity = cfg.hw.hbm_capacity_gb * 1e9
        writer.writerow(["total", "hbm_capacity", f"{capacity:.0f}", fmt_bytes(capacity)])
        usable_capacity = cfg.hw.usable_hbm_capacity_gb * 1e9
        writer.writerow(["total", "hbm_reserved_pct", f"{cfg.hw.hbm_reserved_pct:.3f}", f"{cfg.hw.hbm_reserved_pct:.3f}%"])
        writer.writerow(["total", "usable_hbm_capacity", f"{usable_capacity:.0f}", fmt_bytes(usable_capacity)])


def export_summary_csv(filepath: str, metrics: dict):
    """Export end-to-end summary metrics as key-value CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def export_config_json(filepath: str, cfg: Config):
    """Export merged config to JSON."""
    d = dataclasses.asdict(cfg)
    with open(filepath, "w") as f:
        json.dump(d, f, indent=2)
