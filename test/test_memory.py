"""Tests for perf_model.memory — KV cache and weight memory analysis."""

import unittest

from test.helpers import make_config

from perf_model.roofline import bytes2
from perf_model.memory import (
    kv_cache_memory,
    kv_cache_total_bytes,
    weight_memory_per_rank,
    weight_memory_total_bytes,
)


# ── KV Cache Memory ──────────────────────────────────────────────────────

class TestKVCacheMemory(unittest.TestCase):
    """kv_cache_memory per-layer and total KV cache sizing."""

    def test_ratio1_swa_cache(self):
        """ratio=1 layer: SWA cache = B * W * kv_dim * 2 (K=V shared, window only)."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        W = cfg.model.window_size
        kv_dim = cfg.model.kv_dim
        expected = B * W * kv_dim * 2
        layer0 = result["layers"][0]
        self.assertEqual(layer0["type"], "SWA")
        self.assertEqual(layer0["bytes"], expected)

    def test_ratio4_compressed_with_index(self):
        """ratio=4 layer: compressed + SWA + index (S//4=32 > topK=16). K=V shared."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        S = cfg.rt.seq_len
        m = cfg.model
        ratio = 4
        S_comp = S // ratio

        # Verify use_index = S_comp > index_topk => 32 > 16 => True
        self.assertGreater(S_comp, m.index_topk)

        layer1 = result["layers"][1]
        self.assertEqual(layer1["type"], "C4A")

        comp_bytes = B * S_comp * m.compress_c_kv * 2
        swa_bytes = B * m.window_size * m.kv_dim * 2
        idx_bytes = B * S_comp * m.index_head_dim * 2

        self.assertEqual(layer1["comp_bytes"], comp_bytes)
        self.assertEqual(layer1["swa_bytes"], swa_bytes)
        self.assertIn("idx_bytes", layer1)
        self.assertEqual(layer1["idx_bytes"], idx_bytes)
        self.assertEqual(layer1["bytes"], comp_bytes + swa_bytes + idx_bytes)

    def test_ratio128_no_index(self):
        """ratio=128 layer: S//128=1, not > topK=16, so NO index cache. K=V shared."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        B = cfg.rt.batch_size // cfg.rt.dp
        S = cfg.rt.seq_len
        m = cfg.model
        ratio = 128
        S_comp = S // ratio

        # Verify use_index = S_comp > index_topk => 1 > 16 => False
        self.assertFalse(S_comp > m.index_topk)

        layer2 = result["layers"][2]
        self.assertEqual(layer2["type"], "C128A")

        comp_bytes = B * S_comp * m.compress_c_kv * 2
        swa_bytes = B * m.window_size * m.kv_dim * 2

        self.assertEqual(layer2["comp_bytes"], comp_bytes)
        self.assertEqual(layer2["swa_bytes"], swa_bytes)
        self.assertNotIn("idx_bytes", layer2)
        self.assertEqual(layer2["bytes"], comp_bytes + swa_bytes)

    def test_total_sums_all_layers(self):
        """Total bytes is sum across all layers."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        expected_total = sum(result["layers"][i]["bytes"]
                            for i in range(cfg.model.num_hidden_layers))
        self.assertEqual(result["total_bytes"], expected_total)

    def test_dp_splits_batch(self):
        """dp splits batch: B = batch_size // dp."""
        cfg_dp1 = make_config(batch_size=4, dp=1)
        cfg_dp2 = make_config(batch_size=4, dp=2)
        r1 = kv_cache_memory(cfg_dp1)
        r2 = kv_cache_memory(cfg_dp2)
        # dp=2 halves effective batch, so total should be half
        self.assertEqual(r1["total_bytes"], 2 * r2["total_bytes"])

    def test_all_layers_present(self):
        """All layers present in result."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        for i in range(cfg.model.num_hidden_layers):
            self.assertIn(i, result["layers"])

    def test_layer_type_labels(self):
        """Layer type labels correct: 'SWA', 'C4A', 'C128A'."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        self.assertEqual(result["layers"][0]["type"], "SWA")
        self.assertEqual(result["layers"][1]["type"], "C4A")
        self.assertEqual(result["layers"][2]["type"], "C128A")
        self.assertEqual(result["layers"][3]["type"], "C4A")

    def test_all_bytes_positive(self):
        """All per-layer bytes are positive."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        for i in range(cfg.model.num_hidden_layers):
            self.assertGreater(result["layers"][i]["bytes"], 0)


# ── Weight Memory ─────────────────────────────────────────────────────────

class TestWeightMemory(unittest.TestCase):
    """weight_memory_per_rank formula verification."""

    def test_attn_per_layer_formula(self):
        """attn_per_layer = bytes2(w_dq + w_uq + w_kv + w_wo_a + w_wo_b) with TP splitting."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        m = cfg.model
        TP = cfg.rt.tp
        H = m.hidden_size

        w_dq = H * m.q_lora_rank
        w_uq = m.q_lora_rank * (m.num_attention_heads // TP) * (m.head_dim + m.rope_head_dim)
        w_kv = H * m.kv_dim
        Ng = m.o_groups
        w_wo_a = (Ng // TP) * (m.num_attention_heads // Ng) * m.head_dim * m.o_lora_rank
        w_wo_b = (m.o_mid_dim // TP) * H

        expected = bytes2(w_dq + w_uq + w_kv + w_wo_a + w_wo_b)
        self.assertEqual(result["attn_per_layer"], expected)

    def test_moe_per_layer_formula(self):
        """moe_per_layer = bytes2(gate + routed/EP + shared)."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        m = cfg.model
        H = m.hidden_size
        EP = cfg.rt.ep

        w_gate = H * m.n_routed_experts
        experts_per_rank = m.n_routed_experts // EP
        w_routed = experts_per_rank * 3 * H * m.moe_inter_dim
        w_shared = m.n_shared_experts * 3 * H * m.moe_inter_dim

        expected = bytes2(w_gate + w_routed + w_shared)
        self.assertEqual(result["moe_per_layer"], expected)

    def test_tp_scaling_attn(self):
        """Higher TP reduces attn_per_layer (Q proj and wo split by TP)."""
        cfg_tp2 = make_config(tp=2, ep=4)
        cfg_tp4 = make_config(tp=4, ep=4)
        r2 = weight_memory_per_rank(cfg_tp2)
        r4 = weight_memory_per_rank(cfg_tp4)
        self.assertGreater(r2["attn_per_layer"], r4["attn_per_layer"])

    def test_ep_scaling_moe(self):
        """Higher EP reduces moe_per_layer (routed experts split by EP)."""
        cfg_ep2 = make_config(tp=2, ep=2)
        cfg_ep4 = make_config(tp=2, ep=4)
        r2 = weight_memory_per_rank(cfg_ep2)
        r4 = weight_memory_per_rank(cfg_ep4)
        self.assertGreater(r2["moe_per_layer"], r4["moe_per_layer"])

    def test_total_includes_all_components(self):
        """Total includes attn + moe + mhc + norm + embed + lm_head + final_norm."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        m = cfg.model
        H = m.hidden_size
        TP = cfg.rt.tp

        total_attn = result["total_attn"]
        total_moe = result["total_moe"]
        total_other = result["total_other"]

        self.assertEqual(result["total"], total_attn + total_moe + total_other)
        self.assertGreater(result["total"], 0)

    def test_embedding_and_lm_head(self):
        """embedding = bytes2(vocab * H), lm_head = bytes2(H * vocab//TP)."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        m = cfg.model
        H = m.hidden_size
        TP = cfg.rt.tp

        self.assertEqual(result["embedding"], bytes2(m.vocab_size * H))
        self.assertEqual(result["lm_head"], bytes2(H * (m.vocab_size // TP)))

    def test_layer_type_counts(self):
        """n_swa_layers and n_comp_layers counts correct for [1,4,128,4]."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        # compress_ratios = [1, 4, 128, 4]
        self.assertEqual(result["n_swa_layers"], 1)
        self.assertEqual(result["n_comp_layers"], 3)


# ── Quantization-aware memory (AC-6 & AC-7) ──────────────────────────────────
# Frozen golden values captured from make_config() (default BF16, zero overhead).
_GOLDEN_BF16_WEIGHT_TOTAL      = 17668096.0
_GOLDEN_BF16_WEIGHT_ATTN_LAYER = 245760.0
_GOLDEN_BF16_WEIGHT_MOE_LAYER  = 3940352.0
_GOLDEN_BF16_KV_TOTAL          = 41216.0

# AC-6: Full frozen fixture dicts — literal public output of make_config() with BF16.
# These are independent of the implementation; a regression in any helper changes
# the public output and breaks the test even if both implementation sides regress together.
_FROZEN_BF16_WEIGHT_DICT = {
    "attn_per_layer":  245760.0,
    "index_per_layer": 65536.0,
    "moe_per_layer":   3940352.0,
    "mhc_per_layer":   384.0,
    "norm_per_layer":  1024.0,
    "n_swa_layers":    1,
    "n_comp_layers":   3,
    "total_attn":      1114112.0,
    "total_moe":       15761408.0,
    "total_other":     792576.0,
    "embedding":       524288.0,
    "lm_head":         262144.0,
    "total":           17668096.0,
}

_FROZEN_BF16_KV_DICT = {
    "layers": {
        0: {"type": "SWA", "bytes": 4096},
        1: {"type": "C4A", "comp_bytes": 8192, "swa_bytes": 4096, "bytes": 16384, "idx_bytes": 4096},
        2: {"type": "C128A", "comp_bytes": 256, "swa_bytes": 4096, "bytes": 4352},
        3: {"type": "C4A", "comp_bytes": 8192, "swa_bytes": 4096, "bytes": 16384, "idx_bytes": 4096},
    },
    "total_bytes": 41216,
}

# AC-6 tests: BF16 identity (same keys, same values, no extra metadata keys).
# AC-7 tests: W8A8/KV8/KV4 exact ratios + overhead; metadata keys present.

class TestMemoryQuantization(unittest.TestCase):
    """weight_memory_per_rank and kv_cache_memory with quantization ratios."""

    # ── AC-6: BF16 identity ───────────────────────────────────────────────

    def test_bf16_weight_memory_identical_keys(self):
        """BF16 weight_memory_per_rank: same keys as baseline, no extra metadata keys."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        expected_keys = {
            "attn_per_layer", "index_per_layer", "moe_per_layer",
            "mhc_per_layer", "norm_per_layer",
            "n_swa_layers", "n_comp_layers",
            "total_attn", "total_moe", "total_other",
            "embedding", "lm_head", "total",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_bf16_kv_cache_memory_identical_keys(self):
        """BF16 kv_cache_memory: keys are exactly {'layers', 'total_bytes'}, no extras."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        self.assertEqual(set(result.keys()), {"layers", "total_bytes"})

    def test_bf16_weight_memory_identical_values(self):
        """BF16 weight_memory_per_rank total matches frozen golden."""
        cfg = make_config()
        result = weight_memory_per_rank(cfg)
        self.assertEqual(result["total"], _GOLDEN_BF16_WEIGHT_TOTAL)
        self.assertEqual(result["attn_per_layer"], _GOLDEN_BF16_WEIGHT_ATTN_LAYER)
        self.assertEqual(result["moe_per_layer"], _GOLDEN_BF16_WEIGHT_MOE_LAYER)
        # BF16 ratio = 1.0, so total must equal the raw formula sum
        self.assertEqual(result["total"], result["total_attn"] + result["total_moe"] + result["total_other"])

    def test_bf16_kv_cache_memory_identical_values(self):
        """BF16 kv_cache_memory total_bytes matches frozen golden."""
        cfg = make_config()
        result = kv_cache_memory(cfg)
        self.assertEqual(result["total_bytes"], _GOLDEN_BF16_KV_TOTAL)

    def test_bf16_weight_memory_overhead_ignored(self):
        """BF16 + non-zero weight overhead returns same total as zero-overhead BF16."""
        r_base = weight_memory_per_rank(make_config())
        r_overhead = weight_memory_per_rank(make_config(weight_scale_overhead_bytes=123.0))
        self.assertEqual(r_overhead["total"], r_base["total"])

    def test_bf16_kv_cache_memory_overhead_ignored(self):
        """BF16 + non-zero KV overhead returns same total_bytes as zero-overhead BF16."""
        r_base = kv_cache_memory(make_config())
        r_overhead = kv_cache_memory(make_config(kv_scale_overhead_bytes=45.0))
        self.assertEqual(r_overhead["total_bytes"], r_base["total_bytes"])

    def test_bf16_weight_memory_full_dict(self):
        """BF16 weight_memory_per_rank matches frozen fixture — all keys and values."""
        self.assertEqual(weight_memory_per_rank(make_config()), _FROZEN_BF16_WEIGHT_DICT)

    def test_bf16_kv_cache_memory_full_dict(self):
        """BF16 kv_cache_memory matches frozen fixture — all keys and nested per-layer values."""
        self.assertEqual(kv_cache_memory(make_config()), _FROZEN_BF16_KV_DICT)

    # ── AC-7: W8A8 weight memory exact ratios ─────────────────────────────

    def test_w8a8_weight_total_is_half_bf16(self):
        """W8A8: weight_memory_per_rank total == bf16_total * 0.5 (exact)."""
        r_w8a8 = weight_memory_per_rank(make_config(quant_mode="w8a8"))
        self.assertEqual(r_w8a8["total"], _GOLDEN_BF16_WEIGHT_TOTAL * 0.5)

    def test_w8a8_weight_per_layer_fields_scaled(self):
        """W8A8: numeric per-layer fields are scaled by exactly 0.5."""
        r_bf16 = weight_memory_per_rank(make_config())
        r_w8a8 = weight_memory_per_rank(make_config(quant_mode="w8a8"))
        for key in ("attn_per_layer", "moe_per_layer", "mhc_per_layer"):
            with self.subTest(key=key):
                self.assertEqual(r_w8a8[key], r_bf16[key] * 0.5)

    def test_w8a8_weight_with_scale_overhead(self):
        """W8A8 + overhead: total == bf16_total * 0.5 (data only); overhead in scale_overhead_bytes."""
        overhead = 1_000_000.0
        r_w8a8 = weight_memory_per_rank(make_config(quant_mode="w8a8", weight_scale_overhead_bytes=overhead))
        self.assertEqual(r_w8a8["total"], _GOLDEN_BF16_WEIGHT_TOTAL * 0.5)
        self.assertEqual(r_w8a8["scale_overhead_bytes"], overhead)
        self.assertEqual(weight_memory_total_bytes(r_w8a8), _GOLDEN_BF16_WEIGHT_TOTAL * 0.5 + overhead)

    def test_w8a8_weight_has_quant_mode_key(self):
        """W8A8: returned dict has 'quant_mode' metadata key."""
        result = weight_memory_per_rank(make_config(quant_mode="w8a8"))
        self.assertIn("quant_mode", result)
        self.assertEqual(result["quant_mode"], "w8a8")

    # ── AC-7: KV cache exact ratios ────────────────────────────────────────

    def test_kv8_total_bytes_is_half_bf16(self):
        """KV8: kv_cache_memory total_bytes == bf16_total * 0.5 (exact)."""
        r_kv8 = kv_cache_memory(make_config(kv_cache_quant_mode="kv8"))
        self.assertEqual(r_kv8["total_bytes"], _GOLDEN_BF16_KV_TOTAL * 0.5)

    def test_kv4_total_bytes_is_quarter_bf16(self):
        """KV4: kv_cache_memory total_bytes == bf16_total * 0.25 (exact)."""
        r_kv4 = kv_cache_memory(make_config(kv_cache_quant_mode="kv4"))
        self.assertEqual(r_kv4["total_bytes"], _GOLDEN_BF16_KV_TOTAL * 0.25)

    def test_kv8_per_layer_bytes_scaled(self):
        """KV8: per-layer byte fields all scaled by exactly 0.5."""
        r_bf16 = kv_cache_memory(make_config())
        r_kv8  = kv_cache_memory(make_config(kv_cache_quant_mode="kv8"))
        for i in range(make_config().model.num_hidden_layers):
            with self.subTest(layer=i):
                self.assertEqual(r_kv8["layers"][i]["bytes"], r_bf16["layers"][i]["bytes"] * 0.5)

    def test_kv8_with_scale_overhead(self):
        """KV8 + overhead: total_bytes == bf16_total * 0.5 (data only); overhead in scale_overhead_bytes."""
        overhead = 500_000.0
        r_kv8 = kv_cache_memory(make_config(kv_cache_quant_mode="kv8", kv_scale_overhead_bytes=overhead))
        self.assertEqual(r_kv8["total_bytes"], _GOLDEN_BF16_KV_TOTAL * 0.5)
        self.assertEqual(r_kv8["scale_overhead_bytes"], overhead)
        self.assertEqual(kv_cache_total_bytes(r_kv8), _GOLDEN_BF16_KV_TOTAL * 0.5 + overhead)

    def test_kv4_with_scale_overhead(self):
        """KV4 + overhead: total_bytes == bf16_total * 0.25 (data only); overhead in scale_overhead_bytes."""
        overhead = 200_000.0
        r_kv4 = kv_cache_memory(make_config(kv_cache_quant_mode="kv4", kv_scale_overhead_bytes=overhead))
        self.assertEqual(r_kv4["total_bytes"], _GOLDEN_BF16_KV_TOTAL * 0.25)
        self.assertEqual(r_kv4["scale_overhead_bytes"], overhead)
        self.assertEqual(kv_cache_total_bytes(r_kv4), _GOLDEN_BF16_KV_TOTAL * 0.25 + overhead)

    def test_kv8_has_kv_cache_quant_mode_key(self):
        """KV8: returned dict has 'kv_cache_quant_mode' metadata key."""
        result = kv_cache_memory(make_config(kv_cache_quant_mode="kv8"))
        self.assertIn("kv_cache_quant_mode", result)
        self.assertEqual(result["kv_cache_quant_mode"], "kv8")

    def test_kv4_has_kv_cache_quant_mode_key(self):
        """KV4: returned dict has 'kv_cache_quant_mode' metadata key."""
        result = kv_cache_memory(make_config(kv_cache_quant_mode="kv4"))
        self.assertIn("kv_cache_quant_mode", result)
        self.assertEqual(result["kv_cache_quant_mode"], "kv4")


if __name__ == "__main__":
    unittest.main()
