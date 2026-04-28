import unittest
from dataclasses import replace

from test.helpers import make_config
from perf_model.serving import (
    tokens_per_forward,
    decode_forward_count,
    make_prefill_compute_config,
    make_prefill_memory_config,
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
        cfg = make_config(seq_len=8192, input_len=8192, output_len=1024, prefix_cache_hit_rate=0.9)
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
        cfg = make_config(input_len=1024, prefix_cache_hit_rate=1.0)
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
        cfg = make_config(mtp_accept_ratio=1.1)
        with self.assertRaises(ValueError):
            evaluate_decode_serving(cfg)

        cfg = make_config(kv_cache_quant_mode="bad")
        with self.assertRaises(ValueError):
            evaluate_prefill_serving(cfg)

        cfg = make_config()
        cfg = replace(cfg, rt=replace(cfg.rt, mtp=-1))
        with self.assertRaises(ValueError):
            make_prefill_compute_config(cfg)

        cfg = make_config(input_len=100, prefix_cache_hit_rate=-0.1)
        with self.assertRaises(ValueError):
            make_prefill_compute_config(cfg)

        cfg = make_config(input_len=100, prefix_cache_hit_rate=1.1)
        with self.assertRaises(ValueError):
            evaluate_prefill_serving(cfg)


if __name__ == "__main__":
    unittest.main()
