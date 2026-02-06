#!/usr/bin/env python3
"""CLI entry point for DeepSeek V4 Performance Model."""

import io
import os
import sys
from datetime import datetime

from perf_model import Config, prefill_model, decode_step, decode_model
from perf_model.report import (
    fmt_ms, print_config_summary, print_phase_report, print_memory_report,
    export_ops_csv, export_layer_summary_csv, export_memory_csv,
    export_summary_csv, export_config_json,
)


class TeeOutput:
    """Write to both stdout and a StringIO buffer."""
    def __init__(self, original):
        self.original = original
        self.buffer = io.StringIO()

    def write(self, s):
        self.original.write(s)
        self.buffer.write(s)

    def flush(self):
        self.original.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def main():
    if len(sys.argv) < 5:
        print("Usage: python main.py <device.json> <network.json> <model.json> <runtime.json>")
        sys.exit(1)

    cfg = Config.from_json(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    # Validate compress_ratios length
    if len(cfg.model.compress_ratios) != cfg.model.num_layers:
        print(f"ERROR: compress_ratios length ({len(cfg.model.compress_ratios)}) "
              f"!= num_layers ({cfg.model.num_layers})")
        sys.exit(1)

    # Validate DP
    DP = cfg.rt.dp
    if cfg.rt.batch_size % DP != 0:
        print(f"ERROR: batch_size ({cfg.rt.batch_size}) must be divisible by dp ({DP})")
        sys.exit(1)

    TP = cfg.rt.tp
    EP = cfg.rt.ep
    total_gpus = TP * DP

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Tee stdout to capture console output
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        print_config_summary(cfg)

        # --- Prefill ---
        prefill = prefill_model(cfg)
        print_phase_report(prefill, cfg)

        # --- Decode ---
        decode_first_step = decode_step(cfg.rt.seq_len, cfg)
        decode_total = decode_model(cfg)
        print_phase_report(decode_total, cfg, detailed_step=decode_first_step)

        # --- Memory ---
        print_memory_report(cfg)

        # --- End-to-end summary ---
        per_rank_batch = cfg.rt.batch_size // DP
        edp = total_gpus / EP

        print("=" * 80)
        print("  END-TO-END SUMMARY")
        print("=" * 80)
        print()
        edp_str = str(int(edp)) if edp == int(edp) else f"{edp:.2f}"
        print(f"  Config: TP={TP}, DP={DP}, EP={EP}, EDP={edp_str}, Total GPUs={total_gpus}")
        print(f"  Global batch={cfg.rt.batch_size}, Per-rank batch={per_rank_batch}")
        print()
        print(f"  Prefill ({per_rank_batch}x{cfg.rt.seq_len} tokens per rank):")
        print(f"    Time:       {fmt_ms(prefill.total_time_s)} ms")
        if prefill.total_time_s > 0:
            per_rank_tps = prefill.total_tokens / prefill.total_time_s
            print(f"    Throughput: {per_rank_tps:.1f} tokens/s (per rank)")
            print(f"    Throughput: {per_rank_tps * DP:.1f} tokens/s (total, x{DP} DP)")
        print()
        print(f"  Decode ({cfg.rt.output_len} tokens):")
        print(f"    Total time: {fmt_ms(decode_total.total_time_s)} ms")
        print(f"    First step: {fmt_ms(decode_first_step.total_time_s)} ms")
        if decode_total.total_time_s > 0:
            per_rank_dec_tps = decode_total.total_tokens / decode_total.total_time_s
            print(f"    Avg tok/s:  {per_rank_dec_tps:.1f} (per rank)")
            print(f"    Avg tok/s:  {per_rank_dec_tps * DP:.1f} (total, x{DP} DP)")
        print()
        total_time = prefill.total_time_s + decode_total.total_time_s
        per_rank_output_tokens = per_rank_batch * cfg.rt.output_len
        total_output_tokens = cfg.rt.batch_size * cfg.rt.output_len
        print(f"  Total (prefill + decode): {fmt_ms(total_time)} ms")
        if total_time > 0:
            per_rank_otps = per_rank_output_tokens / total_time
            print(f"  Output tokens/s:          {per_rank_otps:.1f} (per rank)")
            print(f"  Output tokens/s:          {per_rank_otps * DP:.1f} (total, x{DP} DP)")
        print()

        # --- Build summary metrics dict ---
        # Comm vs compute breakdown
        def _comm_stats(pp):
            comm = sum(op.comm_time_s for lp in pp.layer_profiles for op in lp.ops)
            comm += sum(op.comm_time_s for op in pp.extra_ops)
            comp = pp.total_time_s - comm
            pct = comm / pp.total_time_s * 100 if pp.total_time_s > 0 else 0.0
            return comp, comm, pct

        pf_comp, pf_comm, pf_cpct = _comm_stats(prefill)
        ds_comp, ds_comm, ds_cpct = _comm_stats(decode_first_step)

        metrics = {
            "total_gpus": total_gpus,
            "dp": DP,
            "tp": TP,
            "ep": EP,
            "edp": edp if edp == int(edp) else f"{edp:.2f}",
            "batch_size": cfg.rt.batch_size,
            "per_rank_batch": per_rank_batch,
            "seq_len": cfg.rt.seq_len,
            "output_len": cfg.rt.output_len,
            "prefill_time_ms": f"{prefill.total_time_s * 1000:.3f}",
            "prefill_tps": f"{prefill.total_tokens / prefill.total_time_s:.1f}" if prefill.total_time_s > 0 else "0",
            "prefill_compute_ms": f"{pf_comp * 1000:.3f}",
            "prefill_comm_ms": f"{pf_comm * 1000:.3f}",
            "prefill_comm_pct": f"{pf_cpct:.2f}",
            "decode_time_ms": f"{decode_total.total_time_s * 1000:.3f}",
            "decode_first_step_ms": f"{decode_first_step.total_time_s * 1000:.3f}",
            "decode_tps": f"{decode_total.total_tokens / decode_total.total_time_s:.1f}" if decode_total.total_time_s > 0 else "0",
            "decode_step_compute_ms": f"{ds_comp * 1000:.3f}",
            "decode_step_comm_ms": f"{ds_comm * 1000:.3f}",
            "decode_step_comm_pct": f"{ds_cpct:.2f}",
            "total_time_ms": f"{total_time * 1000:.3f}",
            "output_tps_per_rank": f"{per_rank_output_tokens / total_time:.1f}" if total_time > 0 else "0",
            "output_tps_total": f"{total_output_tokens / total_time:.1f}" if total_time > 0 else "0",
        }

        # --- Export CSV files ---
        export_ops_csv(os.path.join(output_dir, "prefill_ops.csv"), prefill)
        export_ops_csv(os.path.join(output_dir, "decode_ops.csv"), decode_first_step)
        export_layer_summary_csv(os.path.join(output_dir, "layer_summary.csv"),
                                  prefill, decode_first_step)
        export_memory_csv(os.path.join(output_dir, "memory.csv"), cfg)
        export_summary_csv(os.path.join(output_dir, "summary.csv"), metrics)
        export_config_json(os.path.join(output_dir, "config.json"), cfg)

    finally:
        sys.stdout = tee.original

    # Write console output
    with open(os.path.join(output_dir, "console_output.txt"), "w") as f:
        f.write(tee.getvalue())

    print(f"\nOutput saved to: {output_dir}/")


if __name__ == "__main__":
    main()
