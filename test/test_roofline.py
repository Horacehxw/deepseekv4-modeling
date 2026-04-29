"""Tests for perf_model.roofline — OpProfile, roofline engine, and comm helpers."""

import unittest

from test.helpers import make_config, assert_op_valid, ATOL, RTOL

from perf_model.config import HardwareConfig
from perf_model.roofline import (
    OpProfile,
    allgather_time,
    allreduce_time,
    alltoall_time,
    bytes2,
    roofline_time,
    sum_ops,
)

# ── Quant-policy tests (AC-1 through AC-5, AC-10) ────────────────────────────
# These tests use the NEW roofline_time(cfg, op_kind) signature.
# They fail with TypeError/AttributeError before the refactor.

_HW_KWARGS = dict(
    cube_tflops=100, vec_tflops=10,
    hbm_bandwidth_gbps=1000,
    flops_utilization=0.5, hbm_bw_utilization=0.8,
    vec_static_latency_us=0.0,
)


class TestRooflineTimeQuantPolicy(unittest.TestCase):
    """New roofline_time signature: cfg + op_kind with inline quantization policy."""

    def setUp(self):
        # Configs share the same hw settings so BF16 identity comparisons are exact.
        self.hw = HardwareConfig(**_HW_KWARGS)
        self.cfg_bf16 = make_config(**_HW_KWARGS, quant_mode="bf16", kv_cache_quant_mode="bf16")
        self.cfg_w8a8 = make_config(**_HW_KWARGS, quant_mode="w8a8", kv_cache_quant_mode="bf16")
        self.cfg_kv8  = make_config(**_HW_KWARGS, quant_mode="bf16", kv_cache_quant_mode="kv8")
        self.cfg_kv4  = make_config(**_HW_KWARGS, quant_mode="bf16", kv_cache_quant_mode="kv4")

    # ── AC-1: BF16 numerical identity ─────────────────────────────────────

    def _bf16_identity(self, op_kind):
        """BF16 config + any op_kind must produce the same numbers as direct hw formula."""
        flops, vec_ops, mem_bytes = 1e12, 1e11, 1e9
        hw = self.cfg_bf16.hw
        expected_cube = flops  / (hw.cube_tflops  * 1e12 * hw.effective_cube_utilization)
        expected_vec  = vec_ops / (hw.vec_tflops   * 1e12 * hw.effective_vec_utilization)
        expected_mem  = mem_bytes / (hw.hbm_bandwidth_gbps * 1e9 * hw.hbm_bw_utilization)
        op = roofline_time("x", flops=flops, vec_ops=vec_ops, mem_bytes=mem_bytes,
                           cfg=self.cfg_bf16, op_kind=op_kind)
        self.assertAlmostEqual(op.cube_time_s, expected_cube, places=15)
        self.assertAlmostEqual(op.vec_time_s,  expected_vec,  places=15)
        self.assertAlmostEqual(op.mem_time_s,  expected_mem,  places=15)
        self.assertAlmostEqual(op.time_s,      max(expected_cube + expected_vec, expected_mem), places=15)

    def test_bf16_identity_gemm(self):      self._bf16_identity("gemm")
    def test_bf16_identity_attention(self): self._bf16_identity("attention")
    def test_bf16_identity_vector(self):    self._bf16_identity("vector")
    def test_bf16_identity_other(self):     self._bf16_identity("other")
    def test_bf16_identity_comm(self):      self._bf16_identity("comm")

    # ── AC-2: W8A8 GEMM policy ────────────────────────────────────────────

    def test_w8a8_gemm_uses_w8a8_tflops(self):
        """W8A8 + gemm: cube_time uses effective_w8a8_tflops (= cube_tflops*2 when unset)."""
        flops = 1e12
        op = roofline_time("g", flops=flops, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg_w8a8, op_kind="gemm")
        w8a8_tflops = self.cfg_bf16.hw.effective_w8a8_tflops  # 200 (2×cube)
        expected_cube = flops / (w8a8_tflops * 1e12 * self.cfg_bf16.hw.flops_utilization)
        self.assertAlmostEqual(op.cube_time_s, expected_cube, places=15)

    def test_w8a8_gemm_scales_mem_bytes_by_half(self):
        """W8A8 + gemm: op.mem_bytes in returned profile == raw_mem_bytes * 0.5."""
        raw_mem = 1e9
        op = roofline_time("g", flops=0, vec_ops=0, mem_bytes=raw_mem,
                           cfg=self.cfg_w8a8, op_kind="gemm")
        self.assertAlmostEqual(op.mem_bytes, raw_mem * 0.5, places=6)

    def test_w8a8_gemm_cube_faster_than_bf16(self):
        """W8A8 cube must be strictly faster than BF16 for identical flops."""
        flops = 1e12
        op_bf16 = roofline_time("g", flops=flops, vec_ops=0, mem_bytes=0,
                                cfg=self.cfg_bf16, op_kind="gemm")
        op_w8a8 = roofline_time("g", flops=flops, vec_ops=0, mem_bytes=0,
                                cfg=self.cfg_w8a8, op_kind="gemm")
        self.assertLess(op_w8a8.cube_time_s, op_bf16.cube_time_s)

    # ── AC-3: Attention KV quantization policy ────────────────────────────

    def test_kv8_attention_scales_mem_by_half(self):
        """kv8 + attention: mem_bytes in profile == raw * 0.5."""
        raw_mem = 1e9
        op = roofline_time("a", flops=0, vec_ops=0, mem_bytes=raw_mem,
                           cfg=self.cfg_kv8, op_kind="attention")
        self.assertAlmostEqual(op.mem_bytes, raw_mem * 0.5, places=6)

    def test_kv4_attention_scales_mem_by_quarter(self):
        """kv4 + attention: mem_bytes in profile == raw * 0.25."""
        raw_mem = 1e9
        op = roofline_time("a", flops=0, vec_ops=0, mem_bytes=raw_mem,
                           cfg=self.cfg_kv4, op_kind="attention")
        self.assertAlmostEqual(op.mem_bytes, raw_mem * 0.25, places=6)

    def test_kv8_attention_compute_throughput_unchanged(self):
        """kv8 + attention: cube_time == BF16 cube_time (KV mode doesn't change compute)."""
        flops = 1e12
        op_bf16 = roofline_time("a", flops=flops, vec_ops=0, mem_bytes=0,
                                cfg=self.cfg_bf16, op_kind="attention")
        op_kv8  = roofline_time("a", flops=flops, vec_ops=0, mem_bytes=0,
                                cfg=self.cfg_kv8,  op_kind="attention")
        self.assertAlmostEqual(op_kv8.cube_time_s, op_bf16.cube_time_s, places=15)

    # ── AC-4: Vector / other unchanged under any quant mode ──────────────

    def _quant_unchanged(self, op_kind, quant_cfg):
        flops, vec_ops, mem_bytes = 1e12, 1e11, 1e9
        op_bf16 = roofline_time("v", flops=flops, vec_ops=vec_ops, mem_bytes=mem_bytes,
                                cfg=self.cfg_bf16, op_kind=op_kind)
        op_q    = roofline_time("v", flops=flops, vec_ops=vec_ops, mem_bytes=mem_bytes,
                                cfg=quant_cfg, op_kind=op_kind)
        self.assertAlmostEqual(op_q.cube_time_s, op_bf16.cube_time_s, places=15)
        self.assertAlmostEqual(op_q.mem_time_s,  op_bf16.mem_time_s,  places=15)
        self.assertAlmostEqual(op_q.time_s,      op_bf16.time_s,      places=15)

    def test_vector_unchanged_under_w8a8(self):  self._quant_unchanged("vector", self.cfg_w8a8)
    def test_vector_unchanged_under_kv8(self):   self._quant_unchanged("vector", self.cfg_kv8)
    def test_other_unchanged_under_w8a8(self):   self._quant_unchanged("other",  self.cfg_w8a8)
    def test_other_unchanged_under_kv4(self):    self._quant_unchanged("other",  self.cfg_kv4)

    # ── AC-5: Comm unchanged ──────────────────────────────────────────────

    def test_comm_unchanged_under_w8a8(self):    self._quant_unchanged("comm", self.cfg_w8a8)
    def test_comm_unchanged_under_kv8(self):     self._quant_unchanged("comm", self.cfg_kv8)

    # ── AC-10: Invalid op_kind raises ValueError ──────────────────────────

    def test_invalid_op_kind_raises_value_error(self):
        with self.assertRaises(ValueError):
            roofline_time("x", flops=0, vec_ops=0, mem_bytes=0,
                          cfg=self.cfg_bf16, op_kind="bad_kind")

    def test_all_valid_op_kinds_accepted(self):
        for kind in ("gemm", "attention", "vector", "other", "comm"):
            with self.subTest(op_kind=kind):
                op = roofline_time("x", flops=0, vec_ops=0, mem_bytes=0,
                                   cfg=self.cfg_bf16, op_kind=kind)
                assert_op_valid(self, op)


class TestBytes2(unittest.TestCase):
    """bytes2: BF16 byte count = count * 2."""

    def test_basic(self):
        self.assertEqual(bytes2(10), 20)

    def test_zero(self):
        self.assertEqual(bytes2(0), 0)

    def test_large(self):
        self.assertEqual(bytes2(1_000_000_000), 2_000_000_000)


class TestRooflineTime(unittest.TestCase):
    """roofline_time: bottleneck detection and time computation."""

    def setUp(self):
        self.hw = HardwareConfig(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
            vec_static_latency_us=0.0,
        )
        self.cfg = make_config(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
            vec_static_latency_us=0.0,
        )

    def test_cube_bound(self):
        """Large flops, small vec/mem -> CUBE bottleneck."""
        op = roofline_time("cube_test", flops=1e15, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg, op_kind="gemm")
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "CUBE")
        self.assertGreater(op.cube_time_s, 0)
        self.assertEqual(op.vec_time_s, 0)
        self.assertEqual(op.mem_time_s, 0)

    def test_vec_bound(self):
        """Large vec_ops, small flops/mem -> VEC bottleneck."""
        op = roofline_time("vec_test", flops=0, vec_ops=1e15, mem_bytes=0,
                           cfg=self.cfg, op_kind="vector")
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "VEC")

    def test_mem_bound(self):
        """Large mem_bytes, small flops/vec -> MEM bottleneck."""
        op = roofline_time("mem_test", flops=0, vec_ops=0, mem_bytes=1e15,
                           cfg=self.cfg, op_kind="other")
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "MEM")

    def test_comm_bound(self):
        """comm_time_s > compute_time -> COMM bottleneck."""
        op = roofline_time("comm_test", flops=0, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg, op_kind="comm", comm_time_s=1.0)
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "COMM")
        self.assertAlmostEqual(op.time_s, 1.0, places=12)

    def test_total_time_formula(self):
        """total = max(cube+vec, mem) + comm."""
        op = roofline_time("formula_test", flops=1e12, vec_ops=1e11,
                           mem_bytes=1e9, cfg=self.cfg, op_kind="gemm", comm_time_s=0.001)
        assert_op_valid(self, op)
        compute = max(op.cube_time_s + op.vec_time_s, op.mem_time_s)
        self.assertAlmostEqual(op.time_s, compute + op.comm_time_s, places=12)

    def test_all_zero(self):
        """All zeros -> empty bottleneck."""
        op = roofline_time("zero", flops=0, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg, op_kind="other")
        assert_op_valid(self, op)
        self.assertEqual(op.bottleneck, "")
        self.assertEqual(op.time_s, 0.0)

    def test_exact_cube_time(self):
        """cube_time = flops / (tflops * 1e12 * util)."""
        flops = 1e12
        expected = flops / (self.hw.cube_tflops * 1e12 * self.hw.flops_utilization)
        op = roofline_time("exact_cube", flops=flops, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg, op_kind="gemm")
        self.assertAlmostEqual(op.cube_time_s, expected, places=15)

    def test_exact_vec_time(self):
        """vec_time = vec_ops / (vec_tflops * 1e12 * util)."""
        vec_ops = 1e12
        expected = vec_ops / (self.hw.vec_tflops * 1e12 * self.hw.flops_utilization)
        op = roofline_time("exact_vec", flops=0, vec_ops=vec_ops, mem_bytes=0,
                           cfg=self.cfg, op_kind="vector")
        self.assertAlmostEqual(op.vec_time_s, expected, places=15)

    def test_separate_cube_and_vec_utilization(self):
        cfg = make_config(
            cube_tflops=100,
            vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5,
            cube_utilization=0.25,
            vec_utilization=0.1,
            hbm_bw_utilization=0.8,
            vec_static_latency_us=0.0,
        )
        op = roofline_time("split_util", flops=1e12, vec_ops=1e12, mem_bytes=0,
                           cfg=cfg, op_kind="gemm")
        self.assertAlmostEqual(op.cube_time_s, 1e12 / (100 * 1e12 * 0.25), places=15)
        self.assertAlmostEqual(op.vec_time_s, 1e12 / (10 * 1e12 * 0.1), places=15)

    def test_exact_mem_time(self):
        """mem_time = mem_bytes / (bw_gbps * 1e9 * bw_util)."""
        mem_bytes = 1e12
        expected = mem_bytes / (self.hw.hbm_bandwidth_gbps * 1e9 * self.hw.hbm_bw_utilization)
        op = roofline_time("exact_mem", flops=0, vec_ops=0, mem_bytes=mem_bytes,
                           cfg=self.cfg, op_kind="other")
        self.assertAlmostEqual(op.mem_time_s, expected, places=15)

    def test_name_passthrough(self):
        op = roofline_time("my_op_name", flops=0, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg, op_kind="other")
        self.assertEqual(op.name, "my_op_name")

    def test_comm_bytes_passthrough(self):
        op = roofline_time("cb", flops=0, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg, op_kind="comm", comm_bytes=12345)
        self.assertEqual(op.comm_bytes, 12345)

    def test_cube_beats_vec_and_mem(self):
        """When cube > vec > mem, bottleneck is CUBE."""
        op = roofline_time("priority", flops=1e14, vec_ops=1e12, mem_bytes=1e9,
                           cfg=self.cfg, op_kind="gemm")
        self.assertEqual(op.bottleneck, "CUBE")

    def test_vec_beats_mem_but_not_cube(self):
        """vec_time > mem_time, both > 0, but no cube -> VEC."""
        op = roofline_time("vec_vs_mem", flops=0, vec_ops=1e13, mem_bytes=1e6,
                           cfg=self.cfg, op_kind="vector")
        self.assertEqual(op.bottleneck, "VEC")


class TestAllReduceTime(unittest.TestCase):
    """allreduce_time: 2*(n-1)/n * vol / (bw*1e9*util) + 2*(n-1)*lat*1e-6."""

    def test_n1_returns_zero(self):
        self.assertEqual(allreduce_time(1000, 1, 100, 10, 0.8), 0.0)

    def test_n2_formula(self):
        vol, n, bw, lat, util = 1e9, 2, 100, 10, 0.8
        factor = 2.0 * (n - 1) / n  # 1.0
        steps = 2 * (n - 1)          # 2
        expected = factor * vol / (bw * 1e9 * util) + steps * lat * 1e-6
        result = allreduce_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)

    def test_n4_formula(self):
        vol, n, bw, lat, util = 1e9, 4, 200, 5, 0.9
        factor = 2.0 * (n - 1) / n  # 1.5
        steps = 2 * (n - 1)          # 6
        expected = factor * vol / (bw * 1e9 * util) + steps * lat * 1e-6
        result = allreduce_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)

    def test_volume_scaling(self):
        """Doubling volume should roughly double the bandwidth term."""
        t1 = allreduce_time(1e9, 4, 100, 0, 1.0)
        t2 = allreduce_time(2e9, 4, 100, 0, 1.0)
        self.assertAlmostEqual(t2, 2 * t1, places=12)

    def test_zero_volume(self):
        """Zero volume -> only latency term."""
        n, bw, lat, util = 4, 100, 10, 0.8
        steps = 2 * (n - 1)
        expected = steps * lat * 1e-6
        result = allreduce_time(0, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)


class TestAllToAllTime(unittest.TestCase):
    """alltoall_time: (n-1)/n * vol / (bw*1e9*util) + lat*1e-6."""

    def test_n1_returns_zero(self):
        self.assertEqual(alltoall_time(1000, 1, 100, 10, 0.8), 0.0)

    def test_n2_formula(self):
        vol, n, bw, lat, util = 1e9, 2, 100, 10, 0.8
        factor = (n - 1) / n  # 0.5
        expected = factor * vol / (bw * 1e9 * util) + lat * 1e-6
        result = alltoall_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)

    def test_n4_formula(self):
        vol, n, bw, lat, util = 1e9, 4, 200, 5, 0.9
        factor = (n - 1) / n  # 0.75
        expected = factor * vol / (bw * 1e9 * util) + lat * 1e-6
        result = alltoall_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)


class TestAllGatherTime(unittest.TestCase):
    """allgather_time: (n-1)/n * vol / (bw*1e9*util) + (n-1)*lat*1e-6."""

    def test_n1_returns_zero(self):
        self.assertEqual(allgather_time(1000, 1, 100, 10, 0.8), 0.0)

    def test_n2_formula(self):
        vol, n, bw, lat, util = 1e9, 2, 100, 10, 0.8
        factor = (n - 1) / n  # 0.5
        steps = n - 1          # 1
        expected = factor * vol / (bw * 1e9 * util) + steps * lat * 1e-6
        result = allgather_time(vol, n, bw, lat, util)
        self.assertAlmostEqual(result, expected, places=15)

    def test_allgather_vs_alltoall_latency(self):
        """allgather has (n-1) latency steps vs alltoall has 1 step.
        With zero volume, difference is purely in latency."""
        n, bw, lat, util = 8, 100, 10, 0.8
        ag = allgather_time(0, n, bw, lat, util)
        a2a = alltoall_time(0, n, bw, lat, util)
        # allgather: (n-1) * lat * 1e-6 = 7 * 10 * 1e-6 = 7e-5
        # alltoall:  1 * lat * 1e-6 = 1e-5
        expected_ag = (n - 1) * lat * 1e-6
        expected_a2a = lat * 1e-6
        self.assertAlmostEqual(ag, expected_ag, places=15)
        self.assertAlmostEqual(a2a, expected_a2a, places=15)
        self.assertGreater(ag, a2a)

    def test_bandwidth_term_same_as_alltoall(self):
        """allgather and alltoall share the same bandwidth factor (n-1)/n."""
        vol, n, bw, util = 1e9, 4, 100, 0.8
        # With zero latency, bandwidth terms should be equal
        ag = allgather_time(vol, n, bw, 0, util)
        a2a = alltoall_time(vol, n, bw, 0, util)
        self.assertAlmostEqual(ag, a2a, places=15)


class TestSumOps(unittest.TestCase):
    """sum_ops: aggregate OpProfiles."""

    def setUp(self):
        self.cfg = make_config(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
            vec_static_latency_us=0.0,
        )

    def test_empty_list(self):
        total = sum_ops([], "empty")
        self.assertEqual(total.name, "empty")
        self.assertEqual(total.time_s, 0.0)
        self.assertEqual(total.bottleneck, "")

    def test_single_op(self):
        op = roofline_time("single", flops=1e12, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg, op_kind="gemm")
        total = sum_ops([op], "agg")
        self.assertEqual(total.name, "agg")
        self.assertAlmostEqual(total.flops, op.flops, places=12)
        self.assertAlmostEqual(total.time_s, op.time_s, places=12)
        self.assertAlmostEqual(total.cube_time_s, op.cube_time_s, places=12)

    def test_two_ops_aggregation(self):
        op1 = roofline_time("a", flops=1e12, vec_ops=0, mem_bytes=0,
                            cfg=self.cfg, op_kind="gemm")
        op2 = roofline_time("b", flops=0, vec_ops=0, mem_bytes=1e12,
                            cfg=self.cfg, op_kind="other")
        total = sum_ops([op1, op2], "both")
        self.assertAlmostEqual(total.flops, op1.flops + op2.flops, places=12)
        self.assertAlmostEqual(total.mem_bytes, op1.mem_bytes + op2.mem_bytes, places=12)
        self.assertAlmostEqual(total.cube_time_s, op1.cube_time_s + op2.cube_time_s, places=12)
        self.assertAlmostEqual(total.mem_time_s, op1.mem_time_s + op2.mem_time_s, places=12)
        self.assertAlmostEqual(total.time_s, op1.time_s + op2.time_s, places=12)

    def test_bottleneck_determination(self):
        """Bottleneck = component with largest aggregate time."""
        op_cube = roofline_time("c", flops=1e15, vec_ops=0, mem_bytes=0,
                                cfg=self.cfg, op_kind="gemm")
        op_mem  = roofline_time("m", flops=0, vec_ops=0, mem_bytes=1e9,
                                cfg=self.cfg, op_kind="other")
        total = sum_ops([op_cube, op_mem], "cube_dom")
        self.assertEqual(total.bottleneck, "CUBE")

    def test_comm_dominant(self):
        """When comm_time_s dominates, bottleneck is COMM."""
        op = OpProfile(name="comm_heavy", comm_time_s=10.0, time_s=10.0,
                       cube_time_s=0.001, vec_time_s=0.001, mem_time_s=0.001)
        total = sum_ops([op], "comm_dom")
        self.assertEqual(total.bottleneck, "COMM")

    def test_comm_bytes_aggregated(self):
        op1 = OpProfile(name="a", comm_bytes=100)
        op2 = OpProfile(name="b", comm_bytes=200)
        total = sum_ops([op1, op2], "comm_bytes")
        self.assertEqual(total.comm_bytes, 300)

    def test_vec_ops_aggregated(self):
        op1 = roofline_time("v1", flops=0, vec_ops=1e12, mem_bytes=0,
                            cfg=self.cfg, op_kind="vector")
        op2 = roofline_time("v2", flops=0, vec_ops=2e12, mem_bytes=0,
                            cfg=self.cfg, op_kind="vector")
        total = sum_ops([op1, op2], "vec_agg")
        self.assertAlmostEqual(total.vec_ops, 3e12, places=6)


class TestVecStaticLatency(unittest.TestCase):
    """vec_static_latency_us is added to vec_time only when vec_ops > 0."""

    def setUp(self):
        self.cfg = make_config(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
            vec_static_latency_us=10.0,
        )
        self.hw = self.cfg.hw

    def test_static_latency_not_added_when_vec_ops_zero(self):
        op = roofline_time("no_vec", flops=1e12, vec_ops=0, mem_bytes=0,
                           cfg=self.cfg, op_kind="gemm")
        self.assertEqual(op.vec_time_s, 0.0)

    def test_static_latency_added_when_vec_ops_positive(self):
        vec_ops = 1e12
        op = roofline_time("with_vec", flops=0, vec_ops=vec_ops, mem_bytes=0,
                           cfg=self.cfg, op_kind="vector")
        expected_compute = vec_ops / (self.hw.vec_tflops * 1e12 * self.hw.effective_vec_utilization)
        expected_vec_time = expected_compute + 10.0 * 1e-6
        self.assertAlmostEqual(op.vec_time_s, expected_vec_time, places=15)

    def test_static_latency_is_phase_independent(self):
        """vec_static_latency_us is not scaled by for_phase()."""
        cfg = make_config(vec_static_latency_us=10.0, prefill_utilization=0.5)
        scaled = cfg.for_phase("prefill")
        self.assertAlmostEqual(scaled.hw.vec_static_latency_us, 10.0)


class TestNewBottleneckFormula(unittest.TestCase):
    """Bottleneck uses cube+vec vs mem, not max(cube, vec, mem)."""

    def setUp(self):
        self.cfg = make_config(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
            vec_static_latency_us=0.0,
        )
        self.hw = self.cfg.hw

    def test_cube_plus_vec_exceeds_mem_gives_cube_bottleneck(self):
        """cube+vec > mem, and cube >= vec -> CUBE."""
        # cube = 1e13/(100*1e12*0.5) = 0.2s, vec = 1e11/(10*1e12*0.5) = 0.02s
        # cube+vec = 0.22s > mem = 1e6/(1000*1e9*0.8) = 1.25e-6s
        op = roofline_time("cube_dom", flops=1e13, vec_ops=1e11, mem_bytes=1e6,
                           cfg=self.cfg, op_kind="other")
        self.assertEqual(op.bottleneck, "CUBE")
        self.assertGreater(op.cube_time_s + op.vec_time_s, op.mem_time_s)

    def test_cube_plus_vec_exceeds_mem_but_vec_larger_gives_vec(self):
        """cube+vec > mem, and vec > cube -> VEC."""
        # cube = 1e11/(100*1e12*0.5) = 0.002s, vec = 1e13/(10*1e12*0.5) = 2.0s
        op = roofline_time("vec_dom", flops=1e11, vec_ops=1e13, mem_bytes=1e6,
                           cfg=self.cfg, op_kind="other")
        self.assertEqual(op.bottleneck, "VEC")

    def test_mem_exceeds_cube_plus_vec_gives_mem(self):
        """mem > cube+vec -> MEM."""
        # cube = 1e12/(100*1e12*0.5) = 0.02s, vec = 1e11/(10*1e12*0.5) = 0.02s
        # cube+vec = 0.04s
        # mem_time = X / (1000*1e9*0.8), want mem_time > 0.04 → X > 0.04 * 8e11 = 3.2e10
        op = roofline_time("mem_dom", flops=1e12, vec_ops=1e11, mem_bytes=4e10,
                           cfg=self.cfg, op_kind="other")
        self.assertEqual(op.bottleneck, "MEM")
        self.assertGreater(op.mem_time_s, op.cube_time_s + op.vec_time_s)

    def test_old_formula_disagrees_with_new_when_cube_plus_vec_exceeds_mem_gt_cube(self):
        """Case where old argmax(cube, vec, mem) would say MEM, new says CUBE/VEC.
        Setup: mem_time > cube_time, but mem_time < cube_time + vec_time.
        """
        # cube = 0.01s, vec = 0.02s, mem = 0.015s (mem > cube, but mem < cube+vec)
        # Old: argmax(0.01, 0.02, 0.015) = VEC (vec_time is max)
        # New: cube+vec=0.03 > mem=0.015 → VEC (cube < vec)
        # Actually in this case both agree on VEC since vec > cube and vec > mem.
        # Let's find a case where they disagree:
        # cube=0.01, vec=0.005, mem=0.012 → old: argmax=MEM (mem is largest);
        # new: cube+vec=0.015 > mem=0.012 → compute bound, cube>=vec → CUBE
        # cube = 0.01 → flops = 0.01 * 100*1e12*0.5 = 5e11
        # vec = 0.005 → vec_ops = 0.005 * 10*1e12*0.5 = 2.5e10
        # mem = 0.012 → mem_bytes = 0.012 * 1000*1e9*0.8 = 9.6e9
        op = roofline_time("disagree", flops=5e11, vec_ops=2.5e10, mem_bytes=9.6e9,
                           cfg=self.cfg, op_kind="other")
        self.assertEqual(op.bottleneck, "CUBE")
        # Verify: old formula would have given MEM
        self.assertGreater(op.mem_time_s, op.cube_time_s)
        self.assertGreater(op.cube_time_s + op.vec_time_s, op.mem_time_s)

    def test_comm_dominates(self):
        """comm > compute_time -> COMM."""
        op = roofline_time("comm_dom", flops=1e12, vec_ops=1e11, mem_bytes=1e9,
                           cfg=self.cfg, op_kind="comm", comm_time_s=100.0)
        self.assertEqual(op.bottleneck, "COMM")


class TestSumOpsNewBottleneck(unittest.TestCase):
    """sum_ops() aggregate bottleneck uses cube+vec vs mem."""

    def setUp(self):
        self.cfg = make_config(
            cube_tflops=100, vec_tflops=10,
            hbm_bandwidth_gbps=1000,
            flops_utilization=0.5, hbm_bw_utilization=0.8,
            vec_static_latency_us=0.0,
        )

    def test_aggregate_cube_plus_vec_gt_mem_gives_cube(self):
        op1 = roofline_time("a", flops=1e13, vec_ops=0, mem_bytes=0,
                            cfg=self.cfg, op_kind="other")
        op2 = roofline_time("b", flops=0, vec_ops=1e12, mem_bytes=1e6,
                            cfg=self.cfg, op_kind="vector")
        total = sum_ops([op1, op2], "agg")
        self.assertGreater(total.cube_time_s + total.vec_time_s, total.mem_time_s)
        self.assertIn(total.bottleneck, {"CUBE", "VEC"})

    def test_aggregate_mem_gt_cube_plus_vec_gives_mem(self):
        op1 = roofline_time("a", flops=1e9, vec_ops=0, mem_bytes=0,
                            cfg=self.cfg, op_kind="other")
        op2 = roofline_time("b", flops=0, vec_ops=0, mem_bytes=1e13,
                            cfg=self.cfg, op_kind="other")
        total = sum_ops([op1, op2], "agg")
        self.assertEqual(total.bottleneck, "MEM")

    def test_aggregate_comm_dominant(self):
        op = OpProfile(name="c", comm_time_s=10.0, time_s=10.0,
                       cube_time_s=0.01, vec_time_s=0.01, mem_time_s=0.01)
        total = sum_ops([op], "comm")
        self.assertEqual(total.bottleneck, "COMM")


class TestPhaseScaling(unittest.TestCase):
    """prefill_model and decode_step apply phase utilization multipliers."""

    def test_prefill_model_slower_with_lower_prefill_utilization(self):
        from perf_model.layers import prefill_model
        cfg_full = make_config(prefill_utilization=1.0)
        cfg_low = make_config(prefill_utilization=0.5)
        p_full = prefill_model(cfg_full)
        p_low = prefill_model(cfg_low)
        self.assertGreater(p_low.total_time_s, p_full.total_time_s)

    def test_decode_step_slower_with_lower_decode_utilization(self):
        from perf_model.layers import decode_step
        cfg_full = make_config(decode_utilization=1.0)
        cfg_low = make_config(decode_utilization=0.5)
        d_full = decode_step(512, cfg_full)
        d_low = decode_step(512, cfg_low)
        self.assertGreater(d_low.total_time_s, d_full.total_time_s)

    def test_prefill_decode_differ_when_utilizations_differ(self):
        from perf_model.layers import prefill_model, decode_step
        cfg = make_config(prefill_utilization=0.9, decode_utilization=0.5)
        cfg_equal = make_config(prefill_utilization=0.9, decode_utilization=0.9)
        d_scaled = decode_step(512, cfg)
        d_equal = decode_step(512, cfg_equal)
        self.assertGreater(d_scaled.total_time_s, d_equal.total_time_s)

    def test_comm_ops_not_scaled_by_phase(self):
        from perf_model.layers import prefill_layer
        cfg_full = make_config(prefill_utilization=1.0)
        cfg_low = make_config(prefill_utilization=0.5)
        lp_full = prefill_layer(0, cfg_full.for_phase("prefill"))
        lp_low = prefill_layer(0, cfg_low.for_phase("prefill"))
        comm_names = {"attn_tp_allreduce", "moe_ep_dispatch", "moe_ep_combine",
                      "sp_ag_before_attn", "sp_ag_before_moe", "sp_ag_after_moe"}
        for op_full, op_low in zip(lp_full.ops, lp_low.ops):
            if op_full.name in comm_names:
                self.assertAlmostEqual(op_full.time_s, op_low.time_s, places=12,
                                       msg=f"Comm op {op_full.name} should not be scaled by phase")


if __name__ == "__main__":
    unittest.main()
