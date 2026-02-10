# DeepSeek V4 推理性能模型

基于 Roofline 模型的 DeepSeek V4 推理延迟估算工具，针对昇腾 910C 硬件。支持逐算子、逐层和端到端延迟估算，覆盖 Prefill 和 Decode 两个阶段。

## 功能特性

- **Roofline 模型**：每个算子分别追踪 Cube（矩阵乘）、Vector、Memory 三种耗时，瓶颈 = argmax
- **并行策略建模**：支持 TP（张量并行）、DP（数据并行）、EP（专家并行）、SP（序列并行）
- **通信分析**：AllReduce、AllToAll 和 AllGather 通信代价估算；逐层和逐阶段的通信 vs 计算分解
- **逐算子分析**：约 30 个独立算子代价函数，覆盖注意力投影、Lightning Index、MoE、mHC 等
- **mHC 内核融合**（默认开启）：融合 mHC 算子将 HBM 流量减少约 10 倍，中间结果保留在寄存器/SRAM 中
- **共享专家重叠**（默认开启）：共享专家计算与 MoE 调度/汇总通信重叠
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
param_search/             # 参数搜索工具
  search.py               # 网格搜索 TP/EP/DP/BS/seq，覆盖 4 个场景
  analyze.py              # 分析结果并生成 search_report.md
  report.md               # 搜索结果详细分析报告
  results/                # 自动生成：带时间戳的搜索结果 CSV
report/                   # 分析报告
  analyze_scenarios.py    # 综合分析：搜索、P/D 比例、算子分析、V3 对比
  report_en.md            # 主分析报告（英文，8 章结构）
  report_zh.md            # 主分析报告（中文翻译）
  ppt_outline_en.md       # PPT 提纲（英文）
  ppt_outline_zh.md       # PPT 提纲（中文）
  data/                   # 自动生成：10 个 JSON 数据文件
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
- `cube_tflops`、`vec_tflops`、`hbm_capacity_gb`、`hbm_bandwidth_gbps`
- `flops_utilization`、`hbm_bw_utilization`

### 添加新模型配置
创建新的 `configs/model_xxx.json`。关键字段：`compress_ratios` 必须是长度为 `num_layers` 的列表，指定每层的压缩比（1 = 全注意力）。

## 参数搜索

通过网格搜索并行策略、批大小和序列长度，寻找最优部署配置。

```bash
python param_search/search.py     # 运行搜索（约 30 秒）
python param_search/analyze.py    # 分析结果并生成报告
```

搜索独立评估 4 个场景：

| 场景 | 优化目标 | 指标 |
|:---|:---|:---|
| Prefill 延迟 | 首 Token 时间 | `prefill_time_ms`（最小化） |
| Decode 延迟 | 单步生成速度 | `decode_first_step_ms`（最小化） |
| Prefill 吞吐 | 每 GPU 每秒处理 Token 数 | `B*S / prefill_s / GPUs`（最大化） |
| Decode 吞吐 | 每 GPU 每秒生成 Token 数 | `B*output_len / decode_s / GPUs`（最大化） |

**搜索空间：** TP ∈ {1,2,4,8,16,32,64}，EP ∈ {1,2,4,...,256}，DP ∈ {1,2,4,8}，BS ∈ {1,...,512}，seq ∈ {1K,...,32K}
**GPU 公式：** `physical_gpus = TP * DP`，约束 `(TP*DP) % EP == 0`
**约束条件：** GPU 数量 ∈ [8, 64]，HBM ≤ 64 GB

**关键结果（昇腾 910C，8K/4K）：**

| 场景 | 最优配置 | 关键指标 | GPU 数 |
|:---|:---|---:|---:|
| Prefill 延迟 | TP=8, EP=64, DP=8, BS=8 | 325 ms | 64 |
| Decode 延迟 | TP=4, EP=32, DP=8, BS=8 | 19.4 ms/步 | 32 |
| Prefill 吞吐 | TP=8, EP=16, DP=2, BS=512 | 1,679 tok/s/GPU | 16 |
| Decode 吞吐 | TP=8, EP=16, DP=2, BS=512 | 307 tok/s/GPU | 16 |

详细搜索分析请参见 [`param_search/report.md`](param_search/report.md)；综合 8 章分析报告（V4 vs V3 对比、瓶颈分析、4 个服务场景：8K/32K/128K/256K、mHC 优化、KV Cache 缩放、部署建议）请参见 [`report/report_en.md`](report/report_en.md)。

## 关键假设

- 所有权重和激活使用 BF16（2 字节）
- Flash Attention 显存模型（中间结果不写回 HBM）
- `head_dim`（512）已包含 `rope_head_dim`（64）— RoPE 嵌入在头维度内，不需要单独投影。Q 字节计算仅使用 `Dqc`。
- Prefill 注意力读取完整 KV 缓存（`B * S * kv_d`），而非逐 Query 窗口读取 — 由于每个 Q 位置都需要其局部窗口，单次顺序读取比逐 Q 随机窗口读取更高效。Decode 仅读取窗口（SWA）或 top-K 条目（压缩注意力）。
- 哈希路由层的 MoE 负载均衡因子 = 1.0，其他层可配置
- 共享专家可与路由专家完全重叠计算（可配置，默认开启）
- mHC 内核融合默认开启：融合后 mHC 前/后投影的中间结果保留在寄存器/SRAM 中，不写回 HBM
- 通信建模为叠加模式（不与计算重叠）
- SP（序列并行）在 Prefill 阶段的每个 T_sp -> T_full 转换点插入 AllGather（注意力之前、MoE 前后、LM Head 之前）
- 单批次 Decode 时 SP 无收益（T=1）
- DP 将全局批次均匀分配；每卡批次 = batch_size / dp
- Decode 聚合使用周期采样 + 梯形插值：单步代价为 `常量 + 线性(S) + 周期(S)`，周期 `P = LCM(compress_ratios)` = 128 步；仅评估 2P 步而非全部 N 步，最高可获得 16 倍加速且结果数学精确
