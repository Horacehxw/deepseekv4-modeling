"""Comprehensive tests for all per-operation cost functions in perf_model.ops."""

import math
import unittest

from test.helpers import make_config, assert_op_valid
from perf_model.roofline import bytes2, bytes4
from perf_model import ops


class TestAttentionProjections(unittest.TestCase):
    """Tests for Q/K/V projections and output projections."""

    def test_q_proj_dq_flops(self):
        cfg = make_config()
        T = cfg.rt.batch_size * cfg.rt.seq_len  # 2*128 = 256
        op = ops.op_q_proj_dq(T, cfg)
        assert_op_valid(self, op)
        H = cfg.model.hidden_size       # 256
        qlr = cfg.model.q_lora_rank     # 128
        expected_flops = T * H * qlr * 2
        self.assertEqual(op.flops, expected_flops)
        self.assertEqual(op.vec_ops, 0)

    def test_q_proj_uq_flops(self):
        cfg = make_config()
        T = cfg.rt.batch_size * cfg.rt.seq_len
        op = ops.op_q_proj_uq(T, cfg)
        assert_op_valid(self, op)
        qlr = cfg.model.q_lora_rank     # 128
        Nq = cfg.model.num_attention_heads  # 8
        Dqc = cfg.model.head_dim            # 64
        TP = cfg.rt.tp                      # 2
        out_dim = (Nq // TP) * Dqc          # 4 * 64 = 256
        expected_flops = T * qlr * out_dim * 2
        self.assertEqual(op.flops, expected_flops)
        self.assertEqual(op.vec_ops, 0)

    def test_q_proj_uq_tp_scaling(self):
        """TP scaling halves the output dim."""
        cfg_tp2 = make_config(tp=2)
        cfg_tp4 = make_config(tp=4, num_attention_heads=8)
        T = 256
        op_tp2 = ops.op_q_proj_uq(T, cfg_tp2)
        op_tp4 = ops.op_q_proj_uq(T, cfg_tp4)
        # TP=4 should have half the flops of TP=2
        self.assertEqual(op_tp4.flops, op_tp2.flops / 2)

    def test_kv_proj_flops(self):
        cfg = make_config()
        T = 256
        op = ops.op_kv_proj(T, cfg)
        assert_op_valid(self, op)
        H = cfg.model.hidden_size   # 256
        kv_d = cfg.model.kv_dim     # head_dim = 64
        expected_flops = T * H * kv_d * 2
        self.assertEqual(op.flops, expected_flops)

    def test_wo_a_flops(self):
        cfg = make_config()
        T = 256
        op = ops.op_wo_a(T, cfg)
        assert_op_valid(self, op)
        Nq = cfg.model.num_attention_heads  # 8
        Dqc = cfg.model.head_dim            # 64
        olr = cfg.model.o_lora_rank         # 64
        TP = cfg.rt.tp                      # 2
        expected_flops = T * (Nq // TP) * Dqc * olr * 2
        self.assertEqual(op.flops, expected_flops)

    def test_wo_b_flops(self):
        cfg = make_config()
        T = 256
        op = ops.op_wo_b(T, cfg)
        assert_op_valid(self, op)
        H = cfg.model.hidden_size   # 256
        omd = cfg.model.o_mid_dim   # o_groups * o_lora_rank = 2*64 = 128
        TP = cfg.rt.tp              # 2
        in_dim = omd // TP          # 64
        expected_flops = T * in_dim * H * 2
        self.assertEqual(op.flops, expected_flops)

    def test_attn_tp_allreduce_tp1_zero(self):
        cfg = make_config(tp=1)
        T = 256
        op = ops.op_attn_tp_allreduce(T, cfg)
        self.assertEqual(op.time_s, 0.0)

    def test_attn_tp_allreduce_tp2_nonzero(self):
        cfg = make_config(tp=2)
        T = 256
        op = ops.op_attn_tp_allreduce(T, cfg)
        self.assertGreater(op.time_s, 0.0)
        H = cfg.model.hidden_size
        expected_vol = bytes2(T * H)
        self.assertEqual(op.comm_bytes, expected_vol)


class TestLightningIndex(unittest.TestCase):
    """Tests for Lightning Index ops."""

    def test_iq_proj_flops(self):
        cfg = make_config()
        T = 256
        op = ops.op_index_iq_proj(T, cfg)
        assert_op_valid(self, op)
        in_dim = cfg.model.q_lora_rank        # 128
        TP = cfg.rt.tp                        # 2
        nh = cfg.model.index_n_heads          # 8
        hd = cfg.model.index_head_dim         # 32
        out_dim = (nh // TP) * hd             # 4 * 32 = 128
        expected_flops = T * in_dim * out_dim * 2
        self.assertEqual(op.flops, expected_flops)

    def test_index_kv_compression_prefill(self):
        cfg = make_config()
        B, S, ratio = 2, 128, 4
        op = ops.op_index_kv_compression_prefill(B, S, ratio, cfg)
        assert_op_valid(self, op)
        H = cfg.model.hidden_size       # 256
        d = cfg.model.index_head_dim    # 32
        g = ratio                       # 4
        S_comp = S // g                 # 32
        coeff = cfg.model.compress_coeff(ratio)  # 1.0
        expected_cube = (8 * B * S * H * d)
        expected_vec = (14 * g - 1) * d * B * S_comp
        expected_mem = bytes2(int((B * S * H + 4 * H * d + 2 * g * d + B * S_comp * d)))
        self.assertEqual(op.flops, expected_cube)
        self.assertEqual(op.vec_ops, expected_vec)
        self.assertEqual(op.mem_bytes, expected_mem)

    def test_index_kv_compression_decode_aligned(self):
        """S_total % ratio == 0: has group compression (vec > 0)."""
        cfg = make_config()
        B, S_total, ratio = 2, 128, 4  # 128 % 4 == 0
        op = ops.op_index_kv_compression_decode(B, S_total, ratio, cfg)
        assert_op_valid(self, op)
        self.assertGreater(op.vec_ops, 0)

    def test_index_kv_compression_decode_unaligned(self):
        """S_total % ratio != 0: projections only (vec == 0)."""
        cfg = make_config()
        B, S_total, ratio = 2, 129, 4  # 129 % 4 != 0
        op = ops.op_index_kv_compression_decode(B, S_total, ratio, cfg)
        assert_op_valid(self, op)
        self.assertEqual(op.vec_ops, 0)

    def test_index_score_flops(self):
        cfg = make_config()
        B, S, ratio = 2, 128, 4
        op = ops.op_index_score(B, S, ratio, cfg)
        assert_op_valid(self, op)
        TP = cfg.rt.tp
        nh = cfg.model.index_n_heads // TP   # 8//2 = 4
        hd = cfg.model.index_head_dim        # 32
        S_comp = S // ratio                  # 32
        expected_flops = B * nh * S * S_comp * hd * 2 + B * nh * S * S_comp
        self.assertEqual(op.flops, expected_flops)
        expected_vec_ops = B * nh * S * S_comp * 3 + B * S * S_comp * math.log(S)
        self.assertAlmostEqual(op.vec_ops, expected_vec_ops)

    def test_index_score_output_fp32(self):
        """Output scores are FP32 (4 bytes per element)."""
        cfg = make_config()
        B, S, ratio = 2, 128, 4
        op = ops.op_index_score(B, S, ratio, cfg)
        S_comp = S // ratio
        TP = cfg.rt.tp
        nh = cfg.model.index_n_heads // TP
        hd = cfg.model.index_head_dim
        act_in = bytes2(B * nh * S * hd) + bytes2(B * S_comp * hd) + bytes2(B * S * nh)
        act_out = bytes4(B * S * S_comp)  # FP32 scores
        expected_mem = act_in + act_out
        self.assertEqual(op.mem_bytes, expected_mem)

    def test_index_score_allreduce_tp1_zero(self):
        cfg = make_config(tp=1)
        B, S, ratio = 2, 128, 4
        op = ops.op_index_score_allreduce(B, S, ratio, cfg)
        self.assertEqual(op.time_s, 0.0)

    def test_index_score_allreduce_volume_bf16(self):
        """Volume is BF16 = bytes2(B*S*S_comp)."""
        cfg = make_config(tp=2)
        B, S, ratio = 2, 128, 4
        op = ops.op_index_score_allreduce(B, S, ratio, cfg)
        S_comp = S // ratio
        expected_vol = bytes2(B * S * S_comp)
        self.assertEqual(op.comm_bytes, expected_vol)
        self.assertGreater(op.time_s, 0.0)


class TestKVCompression(unittest.TestCase):
    """Tests for KV compression ops (K=V shared, coeff-scaled)."""

    def test_prefill_formula(self):
        """KV compression prefill: coeff * (H*c_kv projection + group compression)."""
        cfg = make_config()
        B, S, ratio = 2, 128, 4
        kv_op = ops.op_kv_compression_prefill(B, S, ratio, cfg)
        assert_op_valid(self, kv_op)
        H = cfg.model.hidden_size        # 256
        c_kv = cfg.model.compress_c_kv   # 64
        g = ratio                        # 4
        S_comp = S // g                  # 32
        coeff = cfg.model.compress_coeff(ratio)  # 1.0
        expected_cube = coeff * (8 * B * S * H * c_kv)
        expected_vec = coeff * (14 * g - 1) * c_kv * B * S_comp
        expected_mem = bytes2(int((B * S * H) + coeff * (4 * H * c_kv) + coeff * (2 * g * c_kv) + B * S_comp * c_kv))
        self.assertEqual(kv_op.flops, expected_cube)
        self.assertEqual(kv_op.vec_ops, expected_vec)
        self.assertEqual(kv_op.mem_bytes, expected_mem)

    def test_decode_aligned_has_vec(self):
        """Aligned decode (S_total % ratio == 0) has vec > 0."""
        cfg = make_config()
        B = 2
        op = ops.op_kv_compression_decode(B, 128, 4, cfg)  # 128 % 4 == 0
        assert_op_valid(self, op)
        self.assertGreater(op.vec_ops, 0)

    def test_decode_unaligned_has_no_vec(self):
        """Unaligned decode (S_total % ratio != 0) has vec == 0."""
        cfg = make_config()
        B = 2
        op = ops.op_kv_compression_decode(B, 129, 4, cfg)  # 129 % 4 != 0
        assert_op_valid(self, op)
        self.assertEqual(op.vec_ops, 0)

    def test_aligned_more_flops_than_unaligned(self):
        """Aligned has strictly more cube flops than unaligned."""
        cfg = make_config()
        B = 2
        aligned = ops.op_kv_compression_decode(B, 128, 4, cfg)
        unaligned = ops.op_kv_compression_decode(B, 129, 4, cfg)
        self.assertEqual(aligned.flops, unaligned.flops)


class TestAttentionCompute(unittest.TestCase):
    """Tests for attention score computation (prefill and decode)."""

    def test_prefill_swa_flops(self):
        cfg = make_config()
        B, S = 2, 128
        op = ops.op_attention_prefill_swa(B, S, cfg)
        assert_op_valid(self, op)
        TP = cfg.rt.tp
        Nq = cfg.model.num_attention_heads // TP   # 8//2 = 4
        kv_d = cfg.model.kv_dim                    # 64
        W = cfg.model.window_size                  # 16
        flops_qk = B * Nq * S * W * kv_d * 2
        flops_sv = B * Nq * S * W * kv_d * 2
        self.assertEqual(op.flops, (flops_qk + flops_sv) * 1.3)

    def test_prefill_swa_flash_attn_mem(self):
        """Flash attention memory: Q + full KV read + O write.
        head_dim includes rope_head_dim. Prefill reads full KV (not just window)."""
        cfg = make_config()
        B, S = 2, 128
        op = ops.op_attention_prefill_swa(B, S, cfg)
        TP = cfg.rt.tp
        Nq = cfg.model.num_attention_heads // TP
        Dqc = cfg.model.head_dim
        kv_d = cfg.model.kv_dim
        q_bytes = bytes2(B * Nq * S * Dqc)
        kv_bytes = bytes2(B * S * kv_d)  # full KV read in prefill
        o_bytes = bytes2(B * Nq * S * Dqc)
        expected_mem = q_bytes + kv_bytes + o_bytes
        self.assertEqual(op.mem_bytes, expected_mem * 2)

    def test_prefill_compressed_with_index(self):
        """C4A: use_index=True -> n_attend=topK. Compressed part only (no SWA)."""
        cfg = make_config()
        B, S, ratio = 2, 128, 4
        op = ops.op_attention_prefill_compressed(B, S, ratio, cfg, use_index=True)
        assert_op_valid(self, op)
        TP = cfg.rt.tp
        Nq = cfg.model.num_attention_heads // TP
        topK = cfg.model.index_topk          # 16
        c_kv = cfg.model.compress_c_kv       # 64
        # Compressed attention only (SWA is a separate op now)
        flops_comp_qk = B * Nq * S * topK * c_kv * 2
        flops_comp_sv = B * Nq * S * topK * c_kv * 2
        expected_flops = flops_comp_qk + flops_comp_sv
        self.assertEqual(op.flops, expected_flops * 1.3)

    def test_prefill_compressed_without_index(self):
        """C128A: use_index=False -> n_attend=S//ratio. Compressed part only (no SWA)."""
        cfg = make_config()
        B, S, ratio = 2, 128, 128
        op = ops.op_attention_prefill_compressed(B, S, ratio, cfg, use_index=False)
        assert_op_valid(self, op)
        TP = cfg.rt.tp
        Nq = cfg.model.num_attention_heads // TP
        S_comp = S // ratio                  # 1
        c_kv = cfg.model.compress_c_kv
        flops_comp_qk = B * Nq * S * S_comp * c_kv * 2
        flops_comp_sv = B * Nq * S * S_comp * c_kv * 2
        expected_flops = flops_comp_qk + flops_comp_sv
        self.assertEqual(op.flops, expected_flops * 1.3)

    def test_decode_swa(self):
        """Decode SWA: S_query=1, attends to min(W, S_total). K=V shared."""
        cfg = make_config()
        B, S_total = 2, 128
        op = ops.op_attention_decode_swa(B, S_total, cfg)
        assert_op_valid(self, op)
        TP = cfg.rt.tp
        Nq = cfg.model.num_attention_heads // TP
        kv_d = cfg.model.kv_dim
        W = min(cfg.model.window_size, S_total)
        flops_qk = B * Nq * 1 * W * kv_d * 2
        flops_sv = B * Nq * 1 * W * kv_d * 2
        self.assertEqual(op.flops, (flops_qk + flops_sv) * 1.3)

    def test_decode_compressed_with_index(self):
        """Decode compressed with index: n_attend=topK. Compressed part only (no SWA)."""
        cfg = make_config()
        B, S_total, ratio = 2, 128, 4
        op = ops.op_attention_decode_compressed(B, S_total, ratio, cfg, use_index=True)
        assert_op_valid(self, op)
        TP = cfg.rt.tp
        Nq = cfg.model.num_attention_heads // TP
        topK = cfg.model.index_topk
        c_kv = cfg.model.compress_c_kv
        flops_comp_qk = B * Nq * 1 * topK * c_kv * 2
        flops_comp_sv = B * Nq * 1 * topK * c_kv * 2
        expected_flops = flops_comp_qk + flops_comp_sv
        self.assertEqual(op.flops, expected_flops * 1.3)

    def test_decode_compressed_without_index(self):
        """Decode compressed without index: n_attend=S_total//ratio. Compressed part only."""
        cfg = make_config()
        B, S_total, ratio = 2, 128, 4
        op = ops.op_attention_decode_compressed(B, S_total, ratio, cfg, use_index=False)
        assert_op_valid(self, op)
        TP = cfg.rt.tp
        Nq = cfg.model.num_attention_heads // TP
        S_comp = S_total // ratio             # 32
        c_kv = cfg.model.compress_c_kv
        flops_comp_qk = B * Nq * 1 * S_comp * c_kv * 2
        flops_comp_sv = B * Nq * 1 * S_comp * c_kv * 2
        expected_flops = flops_comp_qk + flops_comp_sv
        self.assertEqual(op.flops, expected_flops * 1.3)


class TestMHCUnfused(unittest.TestCase):
    """Tests for unfused mHC ops."""

    def test_pre_cube_formula(self):
        cfg = make_config()
        T = 256
        op = ops.op_mhc_pre(T, cfg)
        assert_op_valid(self, op)
        n = cfg.model.hc_mult          # 4
        D = cfg.model.hidden_size      # 256
        expected_cube = 2 * T * (n**2 + 2*n) * n * D + 5 * T * n + 2 * T * n**2
        self.assertEqual(op.flops, expected_cube)

    def test_pre_fp32_mem(self):
        """Pre uses FP32 (4 bytes per element)."""
        cfg = make_config()
        T = 256
        op = ops.op_mhc_pre(T, cfg)
        n = cfg.model.hc_mult
        D = cfg.model.hidden_size
        expected_mem = bytes4(3 * T * n * D + T * (n**2 + 2*n) * n * D + T * (n**2 + 2*n))
        self.assertEqual(op.mem_bytes, expected_mem)

    def test_sinkhorn_no_cube(self):
        """Sinkhorn has no cube flops."""
        cfg = make_config()
        T = 256
        op = ops.op_mhc_sinkhorn(T, cfg)
        assert_op_valid(self, op)
        self.assertEqual(op.flops, 0)

    def test_sinkhorn_vec_formula(self):
        cfg = make_config()
        T = 256
        op = ops.op_mhc_sinkhorn(T, cfg)
        n = cfg.model.hc_mult
        expected_vec = T * n**2 + 40 * T * n * (2*n - 1)
        self.assertEqual(op.vec_ops, expected_vec)

    def test_sinkhorn_fp32_mem(self):
        cfg = make_config()
        T = 256
        op = ops.op_mhc_sinkhorn(T, cfg)
        n = cfg.model.hc_mult
        expected_mem = bytes4(2 * T * n**2 + 20 * (2 * T * n**2 + 2 * T * n**2))
        self.assertEqual(op.mem_bytes, expected_mem)

    def test_post_cube_formula(self):
        cfg = make_config()
        T = 256
        op = ops.op_mhc_post(T, cfg)
        assert_op_valid(self, op)
        n = cfg.model.hc_mult
        D = cfg.model.hidden_size
        expected_cube = 2 * T * n**2 * D + 2 * T * n * D
        self.assertEqual(op.flops, expected_cube)

    def test_post_vec_is_nD(self):
        cfg = make_config()
        T = 256
        op = ops.op_mhc_post(T, cfg)
        n = cfg.model.hc_mult
        D = cfg.model.hidden_size
        self.assertEqual(op.vec_ops, T * n * D)


class TestMHCFused(unittest.TestCase):
    """Tests for fused mHC ops (kernel fusion)."""

    def test_pre_fused_cube_matches_unfused(self):
        """Fused pre cube should match unfused pre cube (same formula)."""
        cfg = make_config()
        T = 256
        unfused_pre = ops.op_mhc_pre(T, cfg)
        fused_pre = ops.op_mhc_pre_fused(T, cfg)
        assert_op_valid(self, fused_pre)
        self.assertEqual(fused_pre.flops, unfused_pre.flops)

    def test_pre_fused_mem_much_less_than_unfused(self):
        """Fused pre mem << unfused pre + sinkhorn mem (10x+ reduction)."""
        cfg = make_config()
        T = 256
        unfused_pre = ops.op_mhc_pre(T, cfg)
        unfused_sink = ops.op_mhc_sinkhorn(T, cfg)
        fused_pre = ops.op_mhc_pre_fused(T, cfg)
        unfused_total_mem = unfused_pre.mem_bytes + unfused_sink.mem_bytes
        # Fused should be at least 10x less
        self.assertLess(fused_pre.mem_bytes * 10, unfused_total_mem)

    def test_pre_fused_fp32_vs_bf16(self):
        """FP32 vs BF16 mem difference (bpe=4 vs bpe=2)."""
        cfg_fp32 = make_config(mhc_fused_bf16=False)
        cfg_bf16 = make_config(mhc_fused_bf16=True)
        T = 256
        op_fp32 = ops.op_mhc_pre_fused(T, cfg_fp32)
        op_bf16 = ops.op_mhc_pre_fused(T, cfg_bf16)
        # BF16 mem should be approximately half of FP32 mem
        # (exact ratio not 2x due to weight terms always being FP32)
        self.assertLess(op_bf16.mem_bytes, op_fp32.mem_bytes)
        # The dominant terms scale by bpe, so roughly 2x ratio
        n = cfg_fp32.model.hc_mult
        D = cfg_fp32.model.hidden_size
        # Dominant term: bpe * T * D * (2*n + 1)
        fp32_dominant = 4 * T * D * (2 * n + 1)
        bf16_dominant = 2 * T * D * (2 * n + 1)
        self.assertAlmostEqual(op_fp32.mem_bytes - op_bf16.mem_bytes,
                               fp32_dominant - bf16_dominant, places=0)

    def test_post_fused_cube_matches_unfused(self):
        """Fused post cube same as unfused post."""
        cfg = make_config()
        T = 256
        unfused_post = ops.op_mhc_post(T, cfg)
        fused_post = ops.op_mhc_post_fused(T, cfg)
        assert_op_valid(self, fused_post)
        self.assertEqual(fused_post.flops, unfused_post.flops + unfused_post.vec_ops)

    def test_post_pre_fused_cube(self):
        """post_pre_fused cube = post cube + pre cube."""
        cfg = make_config()
        T = 256
        fused_post = ops.op_mhc_post_fused(T, cfg)
        fused_pre = ops.op_mhc_pre_fused(T, cfg)
        fused_pp = ops.op_mhc_post_pre_fused(T, cfg)
        assert_op_valid(self, fused_pp)
        self.assertEqual(fused_pp.flops, fused_post.flops + fused_pre.flops)

    def test_post_pre_fused_mem_less_than_separate(self):
        """post_pre_fused mem < separate post_fused + pre_fused (deep fusion savings)."""
        cfg = make_config()
        T = 256
        fused_post = ops.op_mhc_post_fused(T, cfg)
        fused_pre = ops.op_mhc_pre_fused(T, cfg)
        fused_pp = ops.op_mhc_post_pre_fused(T, cfg)
        separate_mem = fused_post.mem_bytes + fused_pre.mem_bytes
        self.assertLess(fused_pp.mem_bytes, separate_mem)


class TestMoE(unittest.TestCase):
    """Tests for MoE ops (gate, dispatch, combine, experts)."""

    def test_gate_flops(self):
        cfg = make_config()
        T = 256
        op = ops.op_moe_gate(T, cfg)
        assert_op_valid(self, op)
        H = cfg.model.hidden_size          # 256
        n_exp = cfg.model.n_routed_experts # 16
        expected_flops = T * H * n_exp * 2
        self.assertEqual(op.flops, expected_flops)

    def test_dispatch_hash_layer_load_1(self):
        """Hash layer (layer_idx < n_hash) uses load_factor=1.0."""
        cfg = make_config(n_hash_layers=1, moe_load_balance_factor=1.2)
        T = 256
        op_hash = ops.op_moe_ep_dispatch(T, layer_idx=0, cfg=cfg)
        op_nonhash = ops.op_moe_ep_dispatch(T, layer_idx=1, cfg=cfg)
        # Hash layer has smaller volume (load=1.0 vs 1.2)
        self.assertLess(op_hash.comm_bytes, op_nonhash.comm_bytes)

    def test_dispatch_ep1_zero_time(self):
        """EP=1 -> zero communication time."""
        cfg = make_config(ep=1)
        T = 256
        op = ops.op_moe_ep_dispatch(T, layer_idx=0, cfg=cfg)
        self.assertEqual(op.time_s, 0.0)

    def test_combine_same_volume_as_dispatch(self):
        """Combine has same volume as dispatch."""
        cfg = make_config()
        T = 256
        dispatch = ops.op_moe_ep_dispatch(T, layer_idx=1, cfg=cfg)
        combine = ops.op_moe_ep_combine(T, layer_idx=1, cfg=cfg)
        self.assertEqual(dispatch.comm_bytes, combine.comm_bytes)

    def test_routed_experts_returns_4_ops(self):
        """Routed experts returns list of 4 ops."""
        cfg = make_config()
        T = 256
        result = ops.op_moe_routed_experts(T, layer_idx=1, cfg=cfg)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        for op in result:
            assert_op_valid(self, op)

    def test_routed_experts_t_rank_scaling(self):
        """T_rank scales by top_k * load_factor / EP."""
        cfg = make_config(ep=4, num_experts_per_tok=2, moe_load_balance_factor=1.2)
        T = 256
        result = ops.op_moe_routed_experts(T, layer_idx=1, cfg=cfg)
        H = cfg.model.hidden_size       # 256
        inter = cfg.model.moe_inter_dim # 512
        EP = cfg.rt.ep                  # 4
        top_k = cfg.model.num_experts_per_tok  # 2
        load = cfg.rt.moe_load_balance_factor  # 1.2
        T_rank = T * top_k * load / EP
        # gate_proj flops = T_rank * H * inter * 2
        expected_gate_flops = T_rank * H * inter * 2
        self.assertAlmostEqual(result[0].flops, expected_gate_flops, places=0)

    def test_shared_expert_returns_4_ops(self):
        """Shared expert returns 4 ops."""
        cfg = make_config()
        T = 256
        result = ops.op_moe_shared_expert(T, cfg)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        for op in result:
            assert_op_valid(self, op)

    def test_shared_expert_no_ep_split(self):
        """Shared expert uses full T tokens (no EP split)."""
        cfg = make_config()
        T = 256
        result = ops.op_moe_shared_expert(T, cfg)
        H = cfg.model.hidden_size
        inter = cfg.model.moe_inter_dim
        # gate_proj flops = T * H * inter * 2 (uses full T, no EP)
        expected_gate_flops = T * H * inter * 2
        self.assertEqual(result[0].flops, expected_gate_flops)


class TestMiscOps(unittest.TestCase):
    """Tests for RMSNorm, embedding, LM head, SP allgather."""

    def test_rmsnorm_no_cube(self):
        cfg = make_config()
        T = 256
        op = ops.op_rmsnorm(T, cfg)
        assert_op_valid(self, op)
        self.assertEqual(op.flops, 0)

    def test_rmsnorm_vec_and_mem(self):
        cfg = make_config()
        T = 256
        op = ops.op_rmsnorm(T, cfg)
        H = cfg.model.hidden_size
        expected_vec = T * H * 3
        expected_mem = bytes2(T * H) * 2
        self.assertEqual(op.vec_ops, expected_vec)
        self.assertEqual(op.mem_bytes, expected_mem)

    def test_embedding_no_flops_no_vec(self):
        cfg = make_config()
        B, S = 2, 128
        op = ops.op_embedding(B, S, cfg)
        assert_op_valid(self, op)
        self.assertEqual(op.flops, 0)
        self.assertEqual(op.vec_ops, 0)

    def test_embedding_mem(self):
        cfg = make_config()
        B, S = 2, 128
        op = ops.op_embedding(B, S, cfg)
        H = cfg.model.hidden_size
        expected_mem = bytes2(B * S * H)
        self.assertEqual(op.mem_bytes, expected_mem)

    def test_lm_head_flops(self):
        cfg = make_config()
        T = 256
        op = ops.op_lm_head(T, cfg)
        assert_op_valid(self, op)
        H = cfg.model.hidden_size       # 256
        V = cfg.model.vocab_size        # 1024
        TP = cfg.rt.tp                  # 2
        v_per_rank = V // TP            # 512
        expected_flops = T * H * v_per_rank * 2
        self.assertEqual(op.flops, expected_flops)

    def test_sp_allgather_disabled(self):
        """SP disabled -> empty op."""
        cfg = make_config(sp=False)
        T_sp = 128
        op = ops.op_sp_allgather(T_sp, cfg)
        self.assertEqual(op.time_s, 0.0)
        self.assertEqual(op.comm_bytes, 0.0)

    def test_sp_allgather_tp1_empty(self):
        """TP=1 -> empty op even with sp=True."""
        cfg = make_config(sp=True, tp=1)
        T_sp = 128
        op = ops.op_sp_allgather(T_sp, cfg)
        self.assertEqual(op.time_s, 0.0)

    def test_sp_allgather_enabled(self):
        """SP enabled with TP>1 -> volume > 0."""
        cfg = make_config(sp=True, tp=2)
        T_sp = 128
        op = ops.op_sp_allgather(T_sp, cfg)
        self.assertGreater(op.comm_bytes, 0)
        self.assertGreater(op.time_s, 0.0)
        H = cfg.model.hidden_size
        TP = cfg.rt.tp
        T_full = T_sp * TP
        expected_vol = bytes2(T_full * H)
        self.assertEqual(op.comm_bytes, expected_vol)


class TestDecodeScoreOps(unittest.TestCase):
    """Tests for decode-specific index score ops."""

    def test_index_score_decode_flops(self):
        cfg = make_config()
        B, S_total, ratio = 2, 128, 4
        op = ops.op_index_score_decode(B, S_total, ratio, cfg)
        assert_op_valid(self, op)
        TP = cfg.rt.tp
        nh = cfg.model.index_n_heads // TP
        hd = cfg.model.index_head_dim
        S_comp = S_total // ratio
        expected_flops = B * nh * S_comp * hd * 2 + B * nh * S_comp
        self.assertEqual(op.flops, expected_flops)
        expected_vec_ops = B * nh * S_comp * 3 + B * S_comp * math.log(S_total)
        self.assertAlmostEqual(op.vec_ops, expected_vec_ops)

    def test_index_score_decode_output_fp32(self):
        """Decode output scores are FP32 and include indexer weight reads."""
        cfg = make_config()
        B, S_total, ratio = 2, 128, 4
        op = ops.op_index_score_decode(B, S_total, ratio, cfg)
        TP = cfg.rt.tp
        nh = cfg.model.index_n_heads // TP
        hd = cfg.model.index_head_dim
        S_comp = S_total // ratio
        act_in = bytes2(B * nh * hd) + bytes2(B * S_comp * hd) + bytes2(B * nh)
        act_out = bytes4(B * S_comp)  # FP32 scores
        expected_mem = act_in + act_out
        self.assertEqual(op.mem_bytes, expected_mem)

    def test_index_score_allreduce_decode_tp1(self):
        cfg = make_config(tp=1)
        B, S_total, ratio = 2, 128, 4
        op = ops.op_index_score_allreduce_decode(B, S_total, ratio, cfg)
        self.assertEqual(op.time_s, 0.0)

    def test_index_score_allreduce_decode_volume(self):
        cfg = make_config(tp=2)
        B, S_total, ratio = 2, 128, 4
        op = ops.op_index_score_allreduce_decode(B, S_total, ratio, cfg)
        S_comp = S_total // ratio
        expected_vol = bytes2(B * S_comp)  # BF16
        self.assertEqual(op.comm_bytes, expected_vol)
        self.assertGreater(op.time_s, 0.0)


if __name__ == "__main__":
    unittest.main()
