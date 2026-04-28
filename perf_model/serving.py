"""Serving-oriented metric helpers."""

import math
from dataclasses import replace

from .config import Config
from .layers import decode_step, prefill_model
from .quantization import (
    quantize_phase_profile,
    quantized_kv_cache_memory,
    quantized_weight_memory_per_rank,
)


def _validate_serving_config(cfg: Config) -> None:
    cfg.rt.validate_serving_fields()

    if cfg.rt.seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    if cfg.rt.request_input_len < 0:
        raise ValueError("request_input_len must be >= 0")
    if cfg.rt.decode_context_len_effective < 0:
        raise ValueError("decode_context_len_effective must be >= 0")
    if cfg.rt.output_len < 0:
        raise ValueError("output_len must be >= 0")

    if cfg.rt.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.rt.tp <= 0:
        raise ValueError("tp must be > 0")
    if cfg.rt.dp <= 0:
        raise ValueError("dp must be > 0")
    if cfg.rt.ep <= 0:
        raise ValueError("ep must be > 0")

    if cfg.rt.batch_size % cfg.rt.dp != 0:
        raise ValueError("batch_size must be divisible by dp")

    physical_gpus = cfg.rt.tp * cfg.rt.dp
    if physical_gpus % cfg.rt.ep != 0:
        raise ValueError("tp * dp must be divisible by ep")

    if cfg.model.num_attention_heads % cfg.rt.tp != 0:
        raise ValueError("num_attention_heads must be divisible by tp")
    if cfg.model.num_attention_heads % cfg.model.o_groups != 0:
        raise ValueError("num_attention_heads must be divisible by o_groups")
    if cfg.model.o_groups % cfg.rt.tp != 0:
        raise ValueError("o_groups must be divisible by tp")
    if cfg.model.index_n_heads % cfg.rt.tp != 0:
        raise ValueError("index_n_heads must be divisible by tp")
    if cfg.model.vocab_size % cfg.rt.tp != 0:
        raise ValueError("vocab_size must be divisible by tp")
    if cfg.model.n_routed_experts % cfg.rt.ep != 0:
        raise ValueError("n_routed_experts must be divisible by ep")


def make_prefill_compute_config(cfg: Config) -> Config:
    _validate_serving_config(cfg)
    seq_len = cfg.rt.effective_prefill_len
    return replace(cfg, rt=replace(cfg.rt, seq_len=seq_len))


def make_prefill_memory_config(cfg: Config) -> Config:
    _validate_serving_config(cfg)
    input_len = cfg.rt.request_input_len
    return replace(cfg, rt=replace(cfg.rt, seq_len=input_len))


def make_decode_compute_config(cfg: Config) -> Config:
    _validate_serving_config(cfg)
    input_len = cfg.rt.decode_context_len_effective
    return replace(cfg, rt=replace(cfg.rt, seq_len=input_len))


def make_decode_memory_config(cfg: Config) -> Config:
    _validate_serving_config(cfg)
    context_len = cfg.rt.decode_context_len_effective + cfg.rt.output_len
    return replace(cfg, rt=replace(cfg.rt, seq_len=context_len))


def tokens_per_forward(mtp: int, mtp_accept_ratio: float) -> float:
    if mtp < 0:
        raise ValueError("mtp must be >= 0")
    if not 0 <= mtp_accept_ratio <= 1:
        raise ValueError("mtp_accept_ratio must be in [0, 1]")
    return 1.0 + mtp * mtp_accept_ratio


def decode_forward_count(output_len: int, mtp: int, mtp_accept_ratio: float) -> int:
    if output_len < 0:
        raise ValueError("output_len must be >= 0")
    return math.ceil(output_len / tokens_per_forward(mtp, mtp_accept_ratio))


def _parallelism_metrics(cfg: Config) -> dict:
    physical_gpus = cfg.rt.tp * cfg.rt.dp
    return {
        "physical_gpus": physical_gpus,
        "batch_size": cfg.rt.batch_size,
        "batch_per_card": cfg.rt.batch_size / physical_gpus,
        "batch_per_rank": cfg.rt.batch_size / cfg.rt.dp,
    }


def _hbm_metrics(cfg: Config) -> dict:
    wm = quantized_weight_memory_per_rank(cfg)
    kv = quantized_kv_cache_memory(cfg)
    weight_hbm_gb = wm["total"] / 1e9
    kv_hbm_gb = kv["total_bytes"] / 1e9
    hbm_total_gb = weight_hbm_gb + kv_hbm_gb
    return {
        "weight_hbm_gb": weight_hbm_gb,
        "kv_hbm_gb": kv_hbm_gb,
        "hbm_total_gb": hbm_total_gb,
        "hbm_margin_gb": cfg.hw.usable_hbm_capacity_gb - hbm_total_gb,
    }


def evaluate_prefill_serving(cfg: Config) -> dict:
    compute_cfg = make_prefill_compute_config(cfg)
    memory_cfg = make_prefill_memory_config(cfg)
    if compute_cfg.rt.seq_len == 0:
        prefill_time_s = 0.0
    else:
        phase = quantize_phase_profile(prefill_model(compute_cfg), compute_cfg)
        prefill_time_s = phase.total_time_s

    physical_gpus = cfg.rt.tp * cfg.rt.dp
    prefill_qps_instance = None
    prefill_tps_per_card = None
    if prefill_time_s > 0:
        prefill_qps_instance = cfg.rt.batch_size / prefill_time_s
        prefill_tps_per_card = (
            cfg.rt.batch_size * cfg.rt.request_input_len / prefill_time_s / physical_gpus
        )

    return {
        **_parallelism_metrics(cfg),
        "input_len": cfg.rt.request_input_len,
        "effective_prefill_len": compute_cfg.rt.seq_len,
        **_hbm_metrics(memory_cfg),
        "prefill_time_ms": prefill_time_s * 1000,
        "prefill_qps_instance": prefill_qps_instance,
        "prefill_tps_per_card": prefill_tps_per_card,
    }


def evaluate_decode_serving(cfg: Config) -> dict:
    compute_cfg = make_decode_compute_config(cfg)
    memory_cfg = make_decode_memory_config(cfg)
    first_context = compute_cfg.rt.seq_len
    num_forwards = decode_forward_count(cfg.rt.output_len, cfg.rt.mtp, cfg.rt.mtp_accept_ratio)
    tpf = tokens_per_forward(cfg.rt.mtp, cfg.rt.mtp_accept_ratio)

    if num_forwards == 0:
        decode_total_s = 0.0
    else:
        last_context = math.ceil(first_context + max(0, num_forwards - 1) * tpf)
        first_phase = quantize_phase_profile(decode_step(first_context, compute_cfg), compute_cfg)
        last_phase = quantize_phase_profile(decode_step(last_context, compute_cfg), compute_cfg)
        decode_total_s = num_forwards * (first_phase.total_time_s + last_phase.total_time_s) / 2

    physical_gpus = cfg.rt.tp * cfg.rt.dp
    tpot_ms = None
    decode_qps_instance = None
    decode_tps_per_card = None
    if decode_total_s > 0:
        tpot_ms = decode_total_s * 1000 / cfg.rt.output_len
        decode_qps_instance = cfg.rt.batch_size / decode_total_s
        decode_tps_per_card = cfg.rt.batch_size * cfg.rt.output_len / decode_total_s / physical_gpus

    return {
        **_parallelism_metrics(cfg),
        "input_len": cfg.rt.request_input_len,
        "output_len": cfg.rt.output_len,
        "decode_hbm_context_len": memory_cfg.rt.seq_len,
        "tokens_per_forward": tpf,
        "decode_forward_count": num_forwards,
        **_hbm_metrics(memory_cfg),
        "decode_total_time_ms": decode_total_s * 1000,
        "tpot_ms": tpot_ms,
        "decode_qps_instance": decode_qps_instance,
        "decode_tps_per_card": decode_tps_per_card,
    }


def compute_pd_ratio(
    p_qps_instance: float,
    d_qps_instance: float,
    tolerance: float = 0.1,
    max_instances: int = 10000,
) -> dict:
    """Find the smallest integer P:D ratio that balances aggregate QPS."""
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
