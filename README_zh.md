# DeepSeek V4 推理性能模型

基于 Roofline 模型的 DeepSeek V4 推理延迟估算工具，针对昇腾 910C 硬件。支持逐算子、逐层和端到端延迟估算，覆盖 Prefill 和 Decode 两个阶段。

## 功能特性

- **Roofline 模型**：每个算子分别追踪 Cube（矩阵乘）、Vector、Memory 三种耗时，瓶颈 = argmax
- **并行策略建模**：支持 TP（张量并行）、DP（数据并行）、EP（专家并行）、SP（序列并行）
- **通信分析**：AllReduce 和 AllToAll 通信代价估算；逐层和逐阶段的通信 vs 计算分解
- **逐算子分析**：约 30 个独立算子代价函数，覆盖注意力投影、Lightning Index、MoE、mHC 等
- **显存分析**：KV Cache 大小和每卡权重显存
- **CSV 导出**：带时间戳的输出目录，包含逐算子、逐层、显存和汇总 CSV
- **零依赖**：仅使用 Python 标准库

## 快速开始

```bash
python main.py configs/device_910C.json configs/network_910C.json configs/model_deepseekv4.json configs/runtime_deepseekv4.json
```

输出保存至 `output/<timestamp>/`，包含 CSV 导出文件和控制台日志。

输出内容包括：
- 配置摘要（硬件、网络、模型、运行时）
- Prefill 阶段：代表层逐算子分析、层汇总（含通信占比）、通信 vs 计算分析、总延迟
- Decode 阶段：逐算子分析、通信 vs 计算分析、单步和总延迟、吞吐量（tokens/s）
- 显存分析：逐层 KV Cache、每卡权重显存、HBM 总使用量
- 端到端汇总及吞吐量

## 配置文件

| 文件 | 说明 |
|------|------|
| `configs/device_910C.json` | 硬件参数：BF16 算力、Vector 算力、HBM 容量/带宽、利用率 |
| `configs/network_910C.json` | 网络参数：TP/EP 带宽、延迟、带宽利用率 |
| `configs/model_deepseekv4.json` | 模型架构：隐藏层大小、层数、注意力头数、MoE 配置、压缩比 |
| `configs/runtime_deepseekv4.json` | 运行时配置：序列长度、批大小、dp、TP/EP/SP、负载均衡因子、输出长度 |

## 项目结构

```
configs/                  # JSON 配置文件
perf_model/               # 核心包
  __init__.py             # 公共 API 导出
  config.py               # 配置数据类 + JSON 加载器
  roofline.py             # OpProfile、Roofline 引擎、通信辅助函数
  ops.py                  # 逐算子代价函数（约 30 个）
  layers.py               # 层和阶段聚合
  memory.py               # KV Cache + 权重显存分析
  report.py               # 格式化、打印、CSV 导出、通信 vs 计算分析
main.py                   # CLI 入口
output/                   # 自动生成：带时间戳的运行结果，包含 CSV 和控制台输出
```

## 架构

数据流遵循简单的管线模式：

1. **config.py** — 从 JSON 加载配置到类型化数据类
2. **roofline.py** — Roofline 核心引擎：根据 FLOPs/向量运算/显存字节数计算耗时分解
3. **ops.py** — 每个算子函数计算其 FLOPs/显存并调用 roofline
4. **layers.py** — 将算子聚合为层，将层聚合为阶段（prefill/decode）
5. **memory.py** — 计算 KV Cache 和权重显存需求
6. **report.py** — 格式化并打印所有结果；导出 CSV 文件；通信 vs 计算分析

## 自定义指南

### 添加新硬件配置
创建新的 `configs/device_xxx.json`，字段对应 `HardwareConfig`：
- `bf16_tflops`、`vec_tflops`、`hbm_capacity_gb`、`hbm_bandwidth_gbps`
- `flops_utilization`、`hbm_bw_utilization`

### 添加新模型配置
创建新的 `configs/model_xxx.json`。关键字段：`compress_ratios` 必须是长度为 `num_layers` 的列表，指定每层的压缩比（1 = 全注意力）。

### 填充 KV 压缩占位符
在 `perf_model/ops.py` 中，以下三个函数返回零开销，需用户根据实际压缩算法填充：
- `op_kv_compression_prefill()` — Prefill 阶段的压缩算法开销
- `op_kv_compression_decode()` — Decode 阶段的每步摊销开销
- `op_index_kv_compression()` — 索引键压缩开销

## 关键假设

- 所有权重和激活使用 BF16（2 字节）
- Flash Attention 显存模型（中间结果不写回 HBM）
- 哈希路由层的 MoE 负载均衡因子 = 1.0，其他层可配置
- 共享专家可与路由专家完全重叠计算（可配置）
- 通信建模为叠加模式（不与计算重叠）
- 单批次 Decode 时 SP 无收益（T=1）
- DP 将全局批次均匀分配；每卡批次 = batch_size / dp
