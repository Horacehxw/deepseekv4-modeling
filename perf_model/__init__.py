"""DeepSeek V4 Inference Performance Model — Roofline-based estimation."""

from .config import Config, HardwareConfig, NetworkConfig, ModelConfig, RuntimeConfig
from .roofline import OpProfile, roofline_time
from .layers import LayerProfile, PhaseProfile, prefill_model, decode_step, decode_model, _compression_period
from .memory import kv_cache_memory, weight_memory_per_rank
