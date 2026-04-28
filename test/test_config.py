"""Tests for perf_model.config — dataclasses and JSON loader."""

import json
import os
import tempfile
import unittest

from test.helpers import make_config, make_prod_config, ATOL, RTOL

from perf_model.config import (
    Config,
    HardwareConfig,
    ModelConfig,
    NetworkConfig,
    RuntimeConfig,
)


class TestHardwareConfig(unittest.TestCase):
    """HardwareConfig default values and custom overrides."""

    def test_defaults(self):
        hw = HardwareConfig()
        self.assertEqual(hw.cube_tflops, 376)
        self.assertEqual(hw.vec_tflops, 24)
        self.assertEqual(hw.hbm_capacity_gb, 64)
        self.assertEqual(hw.hbm_bandwidth_gbps, 1800)
        self.assertEqual(hw.flops_utilization, 0.5)
        self.assertEqual(hw.hbm_bw_utilization, 0.8)

    def test_custom_values(self):
        hw = HardwareConfig(cube_tflops=500, vec_tflops=32, hbm_capacity_gb=128)
        self.assertEqual(hw.cube_tflops, 500)
        self.assertEqual(hw.vec_tflops, 32)
        self.assertEqual(hw.hbm_capacity_gb, 128)
        # unchanged defaults
        self.assertEqual(hw.hbm_bandwidth_gbps, 1800)


class TestNetworkConfig(unittest.TestCase):
    """NetworkConfig default values."""

    def test_defaults(self):
        net = NetworkConfig()
        self.assertEqual(net.tp_bandwidth_gbps, 392)
        self.assertEqual(net.ep_bandwidth_gbps, 392)
        self.assertEqual(net.latency_us, 10)
        self.assertEqual(net.bandwidth_utilization, 0.8)


class TestModelConfig(unittest.TestCase):
    """ModelConfig defaults and computed properties."""

    def test_defaults(self):
        m = ModelConfig()
        self.assertEqual(m.hidden_size, 4096)
        self.assertEqual(m.num_hidden_layers, 43)
        self.assertEqual(m.vocab_size, 129280)
        self.assertEqual(m.num_attention_heads, 64)
        self.assertEqual(m.head_dim, 512)
        self.assertEqual(m.n_routed_experts, 256)
        self.assertEqual(m.num_experts_per_tok, 6)

    def test_num_kv_heads(self):
        m = ModelConfig()
        self.assertEqual(m.num_kv_heads, 1)

    def test_kv_dim(self):
        m = ModelConfig()
        self.assertEqual(m.kv_dim, m.head_dim)

    def test_compress_c_kv(self):
        m = ModelConfig()
        self.assertEqual(m.compress_c_kv, m.head_dim)

    def test_compress_coeff(self):
        m = ModelConfig()
        self.assertEqual(m.compress_coeff(1), 0.0)
        self.assertEqual(m.compress_coeff(4), 1.0)
        self.assertEqual(m.compress_coeff(128), 0.5)

    def test_o_mid_dim(self):
        m = ModelConfig()
        self.assertEqual(m.o_mid_dim, m.o_groups * m.o_lora_rank)
        self.assertEqual(m.o_mid_dim, 8 * 1024)

    def test_properties_with_custom_values(self):
        m = ModelConfig(head_dim=256, o_groups=4, o_lora_rank=512)
        self.assertEqual(m.kv_dim, 256)
        self.assertEqual(m.compress_c_kv, 256)
        self.assertEqual(m.o_mid_dim, 4 * 512)


class TestRuntimeConfig(unittest.TestCase):
    """RuntimeConfig defaults, especially mHC flags."""

    def test_defaults(self):
        rt = RuntimeConfig()
        self.assertTrue(rt.mhc_kernel_fused)
        self.assertTrue(rt.shared_expert_overlapped)
        self.assertFalse(rt.mhc_sp)
        self.assertFalse(rt.mhc_fused_bf16)
        self.assertTrue(rt.sp)
        self.assertEqual(rt.seq_len, 4096)
        self.assertEqual(rt.batch_size, 1)
        self.assertEqual(rt.tp, 4)
        self.assertEqual(rt.ep, 64)
        self.assertEqual(rt.dp, 1)
        self.assertIsNone(rt.input_len)
        self.assertIsNone(rt.decode_context_len)
        self.assertEqual(rt.prefix_cache_hit_rate, 0.0)

    def test_helper_semantics(self):
        rt = RuntimeConfig(
            seq_len=4096,
            input_len=3000,
            decode_context_len=None,
            prefix_cache_hit_rate=0.25,
        )
        self.assertEqual(rt.request_input_len, 3000)
        self.assertEqual(rt.effective_prefill_len, 2250)
        self.assertEqual(rt.decode_context_len_effective, 3000)

        rt = RuntimeConfig(
            seq_len=4096,
            input_len=None,
            decode_context_len=512,
            prefix_cache_hit_rate=0.1,
        )
        self.assertEqual(rt.request_input_len, 4096)
        self.assertEqual(rt.effective_prefill_len, 3687)
        self.assertEqual(rt.decode_context_len_effective, 512)

    def test_effective_prefill_len_rounding_boundaries(self):
        self.assertEqual(
            RuntimeConfig(seq_len=10, input_len=10, prefix_cache_hit_rate=0.7).effective_prefill_len,
            3,
        )
        self.assertEqual(
            RuntimeConfig(seq_len=10, input_len=10, prefix_cache_hit_rate=0.0).effective_prefill_len,
            10,
        )
        self.assertEqual(
            RuntimeConfig(seq_len=10, input_len=10, prefix_cache_hit_rate=1.0).effective_prefill_len,
            0,
        )


class TestConfigFromJson(unittest.TestCase):
    """Config.from_json loading, normalization, and error handling."""

    def test_load_prod_configs(self):
        cfg = make_prod_config()
        self.assertIsInstance(cfg, Config)
        self.assertEqual(cfg.hw.cube_tflops, 376)
        self.assertEqual(cfg.model.num_hidden_layers, 43)
        self.assertEqual(cfg.model.vocab_size, 129280)

    def test_network_gbps_case_normalization(self):
        """Network JSON uses GBps keys; from_json lowercases them."""
        cfg = make_prod_config()
        self.assertEqual(cfg.net.tp_bandwidth_gbps, 392)
        self.assertEqual(cfg.net.ep_bandwidth_gbps, 392)

    def test_unknown_model_fields_filtered(self):
        """Model JSON has extra fields (architectures, rope_scaling, etc.) that
        are silently ignored rather than causing an error."""
        cfg = make_prod_config()
        self.assertFalse(hasattr(cfg.model, "architectures"))
        self.assertFalse(hasattr(cfg.model, "rope_scaling"))
        self.assertFalse(hasattr(cfg.model, "model_type"))
        self.assertFalse(hasattr(cfg.model, "dtype"))

    def test_compress_ratios_structure(self):
        """compress_ratios: 2x ratio=1 + 21x ratio=4 + 20x ratio=128 = 43 total."""
        cfg = make_prod_config()
        ratios = cfg.model.compress_ratios
        self.assertEqual(len(ratios), 43)
        self.assertEqual(ratios.count(1), 2)
        self.assertEqual(ratios.count(4), 21)
        self.assertEqual(ratios.count(128), 20)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            Config.from_json(
                "/nonexistent/device.json",
                "/nonexistent/network.json",
                "/nonexistent/model.json",
                "/nonexistent/runtime.json",
            )

    def test_roundtrip_custom_json(self):
        """Write minimal JSON files, load them, verify values."""
        with tempfile.TemporaryDirectory() as d:
            hw_path = os.path.join(d, "hw.json")
            net_path = os.path.join(d, "net.json")
            model_path = os.path.join(d, "model.json")
            rt_path = os.path.join(d, "rt.json")

            with open(hw_path, "w") as f:
                json.dump({"cube_tflops": 100, "vec_tflops": 10,
                           "hbm_capacity_gb": 32, "hbm_bandwidth_gbps": 900,
                           "flops_utilization": 0.6, "hbm_bw_utilization": 0.7}, f)
            with open(net_path, "w") as f:
                json.dump({"tp_bandwidth_GBps": 200, "ep_bandwidth_GBps": 200,
                           "latency_us": 5, "bandwidth_utilization": 0.9}, f)
            with open(model_path, "w") as f:
                json.dump({"hidden_size": 512, "num_hidden_layers": 2,
                           "vocab_size": 1000, "num_attention_heads": 4,
                           "head_dim": 64, "rope_head_dim": 16, "q_lora_rank": 64,
                           "o_groups": 2, "o_lora_rank": 32,
                           "index_n_heads": 4, "index_head_dim": 32,
                           "index_topk": 16, "window_size": 32,
                           "compress_ratios": [1, 4], "hc_mult": 4,
                           "n_routed_experts": 8, "num_experts_per_tok": 2,
                           "n_shared_experts": 1, "moe_inter_dim": 256,
                           "n_hash_layers": 1,
                           "extra_field": "ignored"}, f)
            with open(rt_path, "w") as f:
                json.dump({"seq_len": 64, "batch_size": 1, "dp": 1, "tp": 2,
                           "ep": 4, "sp": True, "moe_load_balance_factor": 1.0,
                           "output_len": 8, "shared_expert_overlapped": False,
                           "mhc_sp": True, "mhc_kernel_fused": False,
                           "mhc_fused_bf16": False,
                           "input_len": 48, "decode_context_len": 32,
                           "prefix_cache_hit_rate": 0.25}, f)

            cfg = Config.from_json(hw_path, net_path, model_path, rt_path)

            self.assertEqual(cfg.hw.cube_tflops, 100)
            self.assertEqual(cfg.net.tp_bandwidth_gbps, 200)
            self.assertEqual(cfg.model.hidden_size, 512)
            self.assertFalse(hasattr(cfg.model, "extra_field"))
            self.assertEqual(cfg.rt.seq_len, 64)
            self.assertEqual(cfg.rt.input_len, 48)
            self.assertEqual(cfg.rt.decode_context_len, 32)
            self.assertEqual(cfg.rt.prefix_cache_hit_rate, 0.25)
            self.assertEqual(cfg.rt.request_input_len, 48)
            self.assertEqual(cfg.rt.effective_prefill_len, 36)
            self.assertEqual(cfg.rt.decode_context_len_effective, 32)
            self.assertFalse(cfg.rt.shared_expert_overlapped)
            self.assertTrue(cfg.rt.mhc_sp)
            self.assertFalse(cfg.rt.mhc_kernel_fused)


class TestMakeConfig(unittest.TestCase):
    """Verify the test helper make_config itself."""

    def test_defaults(self):
        cfg = make_config()
        self.assertEqual(cfg.model.num_hidden_layers, 4)
        self.assertEqual(cfg.model.hidden_size, 256)
        self.assertEqual(cfg.model.num_attention_heads, 8)
        self.assertEqual(cfg.model.head_dim, 64)
        self.assertEqual(cfg.rt.tp, 2)
        self.assertEqual(cfg.rt.ep, 4)
        self.assertEqual(cfg.rt.dp, 1)
        self.assertEqual(cfg.rt.batch_size, 2)
        self.assertEqual(cfg.rt.seq_len, 128)
        self.assertEqual(cfg.model.compress_ratios, [1, 4, 128, 4])

    def test_overrides(self):
        cfg = make_config(tp=8, batch_size=16, hidden_size=512,
                          input_len=96, prefix_cache_hit_rate=0.5)
        self.assertEqual(cfg.rt.tp, 8)
        self.assertEqual(cfg.rt.batch_size, 16)
        self.assertEqual(cfg.model.hidden_size, 512)
        self.assertEqual(cfg.rt.input_len, 96)
        self.assertEqual(cfg.rt.request_input_len, 96)

    def test_unknown_override_raises(self):
        with self.assertRaises(ValueError):
            make_config(nonexistent_field=42)


if __name__ == "__main__":
    unittest.main()
