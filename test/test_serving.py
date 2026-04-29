import unittest
from dataclasses import replace

from test.helpers import make_config
from perf_model.serving import (
    tokens_per_forward,
    decode_forward_count,
    make_prefill_compute_config,
    make_prefill_memory_config,
    make_decode_compute_config,
    make_decode_memory_config,
    evaluate_prefill_serving,
    evaluate_decode_serving,
    compute_pd_ratio,
)


class TestServingHelpers(unittest.TestCase):
    def test_mtp_tokens_per_forward(self):
        self.assertEqual(tokens_per_forward(0, 0.9), 1.0)
        self.assertAlmostEqual(tokens_per_forward(1, 0.9), 1.9)
        self.assertEqual(decode_forward_count(output_len=1024, mtp=1, mtp_accept_ratio=0.9), 539)

    def test_prefix_cache_compute_and_memory_lengths(self):
        cfg = make_config(
            seq_len=8192,
            input_len=8192,
            output_len=1024,
            prefix_cache_hit_rate=0.9,
            ep=2,
        )
        self.assertEqual(make_prefill_compute_config(cfg).rt.seq_len, 820)
        self.assertEqual(make_prefill_memory_config(cfg).rt.seq_len, 8192)
        self.assertEqual(make_decode_memory_config(cfg).rt.seq_len, 9216)

    def test_decode_mtp_improves_tpot_same_batch(self):
        base = make_config(seq_len=256, input_len=256, output_len=32, batch_size=8, dp=4, tp=2)
        no_mtp = evaluate_decode_serving(base)
        mtp_cfg = make_config(
            seq_len=256,
            input_len=256,
            output_len=32,
            batch_size=8,
            dp=4,
            tp=2,
            mtp=1,
            mtp_accept_ratio=0.9,
        )
        mtp = evaluate_decode_serving(mtp_cfg)
        self.assertLess(mtp["tpot_ms"], no_mtp["tpot_ms"])

    def test_pd_ratio_uses_instance_qps(self):
        ratio = compute_pd_ratio(10.0, 25.0, tolerance=0.0)
        self.assertEqual(ratio["prefill_instances"], 5)
        self.assertEqual(ratio["decode_instances"], 2)

    def test_full_prefix_cache_hit_returns_zero_prefill_time_and_no_throughput(self):
        cfg = make_config(input_len=1024, prefix_cache_hit_rate=1.0, ep=2)
        metrics = evaluate_prefill_serving(cfg)
        self.assertEqual(metrics["effective_prefill_len"], 0)
        self.assertEqual(metrics["prefill_time_ms"], 0.0)
        self.assertIsNone(metrics["prefill_qps_instance"])
        self.assertIsNone(metrics["prefill_tps_per_card"])

    def test_decode_serving_reports_mtp_batch_and_hbm_fields(self):
        cfg = make_config(
            seq_len=256,
            input_len=256,
            output_len=32,
            batch_size=8,
            dp=4,
            tp=2,
            mtp=1,
            mtp_accept_ratio=0.9,
        )
        metrics = evaluate_decode_serving(cfg)
        self.assertAlmostEqual(metrics["tokens_per_forward"], 1.9)
        self.assertEqual(metrics["decode_forward_count"], 17)
        self.assertEqual(metrics["batch_per_card"], 1.0)
        self.assertEqual(metrics["batch_per_rank"], 2.0)
        self.assertEqual(metrics["decode_hbm_context_len"], 288)
        for key in ("weight_hbm_gb", "kv_hbm_gb", "hbm_total_gb", "hbm_margin_gb"):
            self.assertIn(key, metrics)

    def test_invalid_runtime_fields_raise_through_serving_helpers(self):
        cfg = make_config(mtp_accept_ratio=1.1, ep=2)
        with self.assertRaises(ValueError):
            evaluate_decode_serving(cfg)

        cfg = make_config(kv_cache_quant_mode="bad", ep=2)
        with self.assertRaises(ValueError):
            evaluate_prefill_serving(cfg)

        cfg = make_config(ep=2)
        cfg = replace(cfg, rt=replace(cfg.rt, mtp=-1))
        with self.assertRaises(ValueError):
            make_prefill_compute_config(cfg)

        cfg = make_config(input_len=100, prefix_cache_hit_rate=-0.1, ep=2)
        with self.assertRaises(ValueError):
            make_prefill_compute_config(cfg)

        cfg = make_config(input_len=100, prefix_cache_hit_rate=1.1, ep=2)
        with self.assertRaises(ValueError):
            evaluate_prefill_serving(cfg)

    def test_invalid_serving_runtime_shapes_raise_through_helpers(self):
        invalid_cases = [
            (make_config(seq_len=-1, ep=2), make_prefill_compute_config, "seq_len"),
            (make_config(input_len=-1, ep=2), make_prefill_compute_config, "request_input_len"),
            (make_config(decode_context_len=-1, ep=2), make_decode_compute_config, "decode_context_len_effective"),
            (make_config(output_len=-1, ep=2), make_decode_memory_config, "output_len"),
            (make_config(batch_size=0, ep=2), make_prefill_compute_config, "batch_size"),
            (make_config(tp=0, ep=2), evaluate_decode_serving, "tp"),
            (make_config(dp=0, ep=2), evaluate_prefill_serving, "dp"),
            (make_config(ep=0), make_prefill_memory_config, "ep"),
            (make_config(batch_size=3, dp=2, tp=2, ep=4), evaluate_decode_serving, "batch_size.*dp"),
            (make_config(tp=2, dp=1, ep=4), make_prefill_compute_config, "tp \\* dp.*ep"),
        ]
        for cfg, helper, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    helper(cfg)

    def test_invalid_serving_model_shapes_raise_through_helpers(self):
        invalid_cases = [
            (make_config(num_attention_heads=7, ep=2), "num_attention_heads.*tp"),
            (make_config(num_attention_heads=6, o_groups=3, ep=2), "o_groups.*tp"),
            (make_config(index_n_heads=7, ep=2), "index_n_heads.*tp"),
            (make_config(vocab_size=1025, ep=2), "vocab_size.*tp"),
            (make_config(n_routed_experts=15, ep=2), "n_routed_experts.*ep"),
            (make_config(num_attention_heads=7, o_groups=2, tp=1, ep=1), "num_attention_heads.*o_groups"),
        ]
        for cfg, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_prefill_serving(cfg)


# ── Quantization integration (AC-8 & AC-9) ───────────────────────────────────
# AC-8: serving outputs match golden values (≤1e-9 rel tol); no double-quant path.
# AC-9: import perf_model.quantization raises ImportError after deletion.
#
# Golden values were captured from the pre-refactor serving path on 2026-04-29
# using make_config(ep=2) with input_len=128, output_len=8.

_GOLDEN_BF16_PREFILL_MS        = 0.386219406802721
_GOLDEN_BF16_WEIGHT_HBM_GB     = 0.030251008
_GOLDEN_BF16_KV_HBM_GB         = 4.1216e-05

_GOLDEN_W8A8_PREFILL_MS        = 0.37440017124716546
_GOLDEN_W8A8_WEIGHT_HBM_GB     = 0.015125504

_GOLDEN_KV8_DECODE_TOTAL_MS    = 1.7520919378684803
_GOLDEN_KV8_KV_HBM_GB          = 2.1376e-05

_REL_TOL = 1e-9   # numerical tolerance for golden-value checks


class TestServingQuantizationIntegration(unittest.TestCase):
    """AC-8 golden-value regression and AC-9 module-deletion guard."""

    def setUp(self):
        self.cfg_bf16 = make_config(input_len=128, output_len=8, ep=2)
        self.cfg_w8a8 = make_config(input_len=128, output_len=8, ep=2, quant_mode="w8a8")
        self.cfg_kv8  = make_config(input_len=128, output_len=8, ep=2, kv_cache_quant_mode="kv8")

    # ── AC-8: Golden-value regression ─────────────────────────────────────

    def _assert_rel(self, actual, expected, msg=""):
        """Assert relative difference ≤ _REL_TOL."""
        if expected == 0:
            self.assertEqual(actual, 0.0, msg)
        else:
            rel = abs(actual - expected) / abs(expected)
            self.assertLessEqual(rel, _REL_TOL, f"{msg}: got {actual}, expected {expected}, rel={rel:.2e}")

    def test_bf16_prefill_time_matches_golden(self):
        result = evaluate_prefill_serving(self.cfg_bf16)
        self._assert_rel(result["prefill_time_ms"], _GOLDEN_BF16_PREFILL_MS, "BF16 prefill_time_ms")

    def test_bf16_prefill_weight_hbm_matches_golden(self):
        result = evaluate_prefill_serving(self.cfg_bf16)
        self._assert_rel(result["weight_hbm_gb"], _GOLDEN_BF16_WEIGHT_HBM_GB, "BF16 weight_hbm_gb")

    def test_bf16_prefill_kv_hbm_matches_golden(self):
        result = evaluate_prefill_serving(self.cfg_bf16)
        self._assert_rel(result["kv_hbm_gb"], _GOLDEN_BF16_KV_HBM_GB, "BF16 kv_hbm_gb")

    def test_w8a8_prefill_time_matches_golden(self):
        result = evaluate_prefill_serving(self.cfg_w8a8)
        self._assert_rel(result["prefill_time_ms"], _GOLDEN_W8A8_PREFILL_MS, "W8A8 prefill_time_ms")

    def test_w8a8_prefill_weight_hbm_matches_golden(self):
        result = evaluate_prefill_serving(self.cfg_w8a8)
        self._assert_rel(result["weight_hbm_gb"], _GOLDEN_W8A8_WEIGHT_HBM_GB, "W8A8 weight_hbm_gb")

    def test_w8a8_prefill_faster_than_bf16(self):
        """W8A8 must be strictly faster than BF16 for the same input."""
        r_bf16 = evaluate_prefill_serving(self.cfg_bf16)
        r_w8a8 = evaluate_prefill_serving(self.cfg_w8a8)
        self.assertLess(r_w8a8["prefill_time_ms"], r_bf16["prefill_time_ms"])

    def test_kv8_decode_time_matches_golden(self):
        result = evaluate_decode_serving(self.cfg_kv8)
        self._assert_rel(result["decode_total_time_ms"], _GOLDEN_KV8_DECODE_TOTAL_MS, "KV8 decode_total_time_ms")

    def test_kv8_decode_kv_hbm_matches_golden(self):
        result = evaluate_decode_serving(self.cfg_kv8)
        self._assert_rel(result["kv_hbm_gb"], _GOLDEN_KV8_KV_HBM_GB, "KV8 kv_hbm_gb")

    def test_kv8_kv_hbm_less_than_bf16(self):
        """KV8 must use strictly less KV HBM than BF16."""
        r_bf16 = evaluate_decode_serving(make_config(input_len=128, output_len=8, ep=2))
        r_kv8  = evaluate_decode_serving(self.cfg_kv8)
        self.assertLess(r_kv8["kv_hbm_gb"], r_bf16["kv_hbm_gb"])

    # ── AC-8: Static assertion — no quantize_phase_profile in serving/init ──

    def test_no_quantize_phase_profile_in_serving_py(self):
        """serving.py must not reference quantize_phase_profile."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "perf_model" / "serving.py"
        self.assertNotIn("quantize_phase_profile", src.read_text())

    def test_no_quantize_phase_profile_in_init_py(self):
        """perf_model/__init__.py must not export quantize_phase_profile."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "perf_model" / "__init__.py"
        self.assertNotIn("quantize_phase_profile", src.read_text())

    # ── AC-9: Module deleted ───────────────────────────────────────────────

    def test_quantization_module_raises_import_error(self):
        """After refactor perf_model.quantization must not exist."""
        import importlib
        with self.assertRaises(ImportError):
            importlib.import_module("perf_model.quantization")

    def test_removed_exports_raise_import_error(self):
        """Removed public exports must raise ImportError when imported from perf_model."""
        removed = (
            "infer_op_kind",
            "quantize_op_profile",
            "quantize_phase_profile",
            "quantized_weight_memory_per_rank",
            "quantized_kv_cache_memory",
        )
        for name in removed:
            with self.subTest(name=name):
                with self.assertRaises(ImportError):
                    exec(f"from perf_model import {name}")


if __name__ == "__main__":
    unittest.main()
