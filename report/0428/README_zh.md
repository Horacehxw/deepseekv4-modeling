# 0428 PD 分离分析报告 — 使用说明

本报告分析 DeepSeek V4 在 Ascend 910C 上的 Prefill/Decode 分离部署性能，覆盖 8K/32K/128K/1M 四个场景。

---

## 一、如何修改配置参数

### 硬件利用率参数（最常调整）

文件：`configs/device_910C.json`

| 字段 | 含义 | 当前值 |
|------|------|--------|
| `cube_utilization` | 矩阵乘法（Cube）单元利用率 | 0.4 |
| `vec_utilization` | 向量单元利用率 | 0.2 |
| `hbm_bw_utilization` | HBM 带宽有效利用率 | 0.3 |
| `prefill_utilization` | Prefill 整体利用率系数 | 1.0 |
| `decode_utilization` | Decode 整体利用率系数 | 0.9 |
| `hbm_bandwidth_gbps` | HBM 峰值带宽（GB/s） | 1800 |
| `hbm_capacity_gb` | 单卡 HBM 容量（GB） | 64 |
| `hbm_reserved_pct` | HBM 系统预留比例（%） | 10.0 |
| `cube_tflops` | Cube 峰值算力（TFLOPS） | 376 |
| `vec_tflops` | 向量峰值算力（TFLOPS） | 24 |

文件 `configs/network_910C.json`

| 字段 | 含义 | 当前值 |
|------|------|--------|
| `bandwidth_utilization` | 通信带宽利用率 | 0.26 （比 A3 原本默认的 0.8 折算为 1/3） |

### 报告专用参数（场景与默认值）

文件：`report/0428/script/common.py`

| 变量 / 字段 | 含义 | 当前值 |
|------------|------|--------|
| `PREFIX_CACHE_HIT_RATES` | 枚举的 prefix cache 命中率 | `[0.0, 0.9, 0.99]` |
| `DECODE_INSTANCE_SIZES` | 枚举的 Decode 实例卡数 | `[8, 16, 32, 64]` |
| `REPORT_DEFAULTS["quant_mode"]` | 权重量化模式 | `"w8a8"` |
| `REPORT_DEFAULTS["kv_cache_quant_mode"]` | KV Cache 量化模式 | `"kv8"` |
| `REPORT_DEFAULTS["mtp_accept_ratio"]` | MTP token 接收率 | `0.9` |
| `REPORT_DEFAULTS["tpot_target_ms"]` | TPOT 约束（ms） | `50.0` |
| `REPORT_DEFAULTS["w8a8_tflops"]` | W8A8 GEMM 有效算力（TFLOPS） | `752.0` |
| `SCENARIOS` | 四个分析场景（输入/输出长度） | 8K/32K/128K/1M + 1K |

修改以上参数后，重新运行生成脚本即可看到更新结果（见第二节）。

---

## 二、手动重新运行生成脚本

在仓库根目录执行：

```bash
python report/0428/script/generate_report.py
```

该脚本会依次完成：

1. **搜索 Prefill 配置**：按 `batch_size=1` 确定最小卡数，再在该卡数内搜索最优 `{TP, EP, DP, batch_size}`。
2. **搜索 Decode 配置**：枚举 `[8, 16, 32, 64]` 卡的 HBM 上限与 TPOT 约束下的最大 batch。
3. **计算 P/D 配比**：按实例 QPS 求整数比，允许 10% 不平衡。
4. **输出数据文件**（`report/0428/data/*.json`）：
   - `prefill_results.json` — Prefill 各场景结果
   - `decode_results.json` — Decode 各场景/卡数结果
   - `pd_ratio_results.json` — P/D 配比与总卡数
   - `manifest.json` — 生成时间戳与配置快照
5. **输出图表**（`report/0428/figure/*.svg`）：HBM、TPS/card、P/D 总卡数堆叠图。

> 脚本直接覆盖输出文件，无需额外参数。完成后可用 `git diff --stat -- report/0428` 确认变更范围。

运行测试验证：

```bash
python -m unittest test.test_report_0428 test.test_param_search test.test_serving -v
```

---

## 三、使用 /refresh_0428 技能自动生成报告

`/refresh_0428` 是一个多阶段自动化技能，适合在模型/配置更改后一键同步整个报告。它在不同 AI 编程助手平台的调用方式如下：

### Claude Code（CLI / IDE 扩展）

在对话框直接输入：

```
/refresh_0428
```

### 其他支持 Skill 工具的平台（Copilot CLI、Gemini CLI 等）

参照各平台的技能调用语法，例如：

```
/skill refresh_0428
```

或在 Gemini CLI 中：

```
/activate_skill refresh_0428
```

如果找不到，他的位置在 `.claude/commands/refresh_0428.md`, 可以直接告诉 agent `请 load .claude/commands/refresh_0428.md 的 SKILL 然后严格执行`

### 技能执行流程

| 阶段 | 内容 |
|------|------|
| Phase 0：预检 | 查看 git 状态，确认脏文件，定位生成入口脚本 |
| Phase 1：重新生成 | 运行 `generate_report.py`，刷新数据 JSON 与 SVG 图表 |
| Phase 2：LLM 改写 | 读取新数据，以技术报告风格重写 `report/0428/report.md` |
| Phase 3：图表审查 | 检查 SVG 标签、图例、数据标注的可读性 |
| Phase 4：验证 | 运行单测、whitespace check、输出最终 diff |
| Phase 5：汇总 | 报告关键变化与未提交的文件，不自动 commit |

> 技能**不会自动提交**，所有修改仅在工作区，需用户确认后手动 `git commit`。
