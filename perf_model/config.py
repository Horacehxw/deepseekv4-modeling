"""Config dataclasses and JSON loader."""

import json
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import List


@dataclass
class HardwareConfig:
    cube_tflops: float = 376
    vec_tflops: float = 24
    hbm_capacity_gb: float = 64
    hbm_bandwidth_gbps: float = 1800
    flops_utilization: float = 0.5
    hbm_bw_utilization: float = 0.8


@dataclass
class NetworkConfig:
    tp_bandwidth_gbps: float = 392
    ep_bandwidth_gbps: float = 392
    latency_us: float = 10
    bandwidth_utilization: float = 0.8


@dataclass
class ModelConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    vocab_size: int = 129280
    num_attention_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    window_size: int = 128
    compress_ratios: List[int] = field(default_factory=list)
    hc_mult: int = 4
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 1
    moe_inter_dim: int = 2048
    n_hash_layers: int = 3

    @property
    def num_kv_heads(self) -> int:
        return 1  # MQA

    @property
    def k_dim(self) -> int:
        # FIXME: now use head dim directly, the rope is contained inside the head dim
        # return self.head_dim + self.rope_head_dim  # 512 + 64 = 576
        return self.head_dim  # 512

    @property
    def v_dim(self) -> int:
        return self.head_dim  # 512

    @property
    def compress_c_k(self) -> int:
        # return self.head_dim + self.rope_head_dim  # = k_dim = 576
        return self.head_dim  # = k_dim = 512

    @property
    def compress_c_v(self) -> int:
        return self.head_dim  # = v_dim = 512

    @property
    def o_mid_dim(self) -> int:
        return self.o_groups * self.o_lora_rank  # 8 * 1024 = 8192


@dataclass
class RuntimeConfig:
    seq_len: int = 4096
    batch_size: int = 1
    dp: int = 1
    tp: int = 4
    ep: int = 64
    sp: bool = True
    moe_load_balance_factor: float = 1.2
    output_len: int = 256
    shared_expert_overlapped: bool = True
    mhc_sp: bool = False
    mhc_kernel_fused: bool = True    # Fuse mHC pre+sinkhorn+post into single kernels (FP32)
    mhc_fused_bf16: bool = False     # Use BF16 activations in fused mHC (inference only)


@dataclass
class Config:
    hw: HardwareConfig = field(default_factory=HardwareConfig)
    net: NetworkConfig = field(default_factory=NetworkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rt: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_json(cls, device_path: str, network_path: str,
                  model_path: str, runtime_path: str) -> "Config":
        with open(device_path) as f:
            hw = HardwareConfig(**json.load(f))
        with open(network_path) as f:
            # Normalize keys to lowercase (JSON uses GBps for clarity, fields use gbps)
            net = NetworkConfig(**{k.lower(): v for k, v in json.load(f).items()})
        with open(model_path) as f:
            data = json.load(f)
            known = {f.name for f in dataclass_fields(ModelConfig)}
            model = ModelConfig(**{k: v for k, v in data.items() if k in known})
        with open(runtime_path) as f:
            rt = RuntimeConfig(**json.load(f))
        return cls(hw=hw, net=net, model=model, rt=rt)
