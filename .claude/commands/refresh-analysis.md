# Refresh All Analysis Data and Reports

Regenerate all analysis data, parameter search results, and update reports using a multi-agent pipeline.
Run this after changing any config defaults (e.g., mhc_kernel_fused, shared_expert_overlapped) or model parameters.

**Bilingual rule:** Every `*.md` report/doc/README must have a `*_zh.md` Chinese translation with identical data/tables and translated prose/headings.

---

## Phase 1: Run Experiments & Gather Data

Run all data-generating scripts **sequentially** before any report writing begins. All agents depend on this data.

> **Model assignment (cost optimization):** Phase 1 data gathering, Agent A (recorder), Agent D (verifier), translator, and doc-updater use **Sonnet** (`model: "sonnet"`). Only Agent B (analyzer) and Agent C (researcher) use **Opus 4.6** (`model: "opus"`) for their deep analytical writing.

### 1.1 Run scenario analysis

Regenerates all `report/data/*.json` files (10 files). Takes ~30s. Covers 4 combos: 8K, 32K, 128K, 256K.

```bash
python report/analyze_scenarios.py
```

### 1.2 Run parameter search

Grid search across TP/EP/DP/BS/seq for 4 scenarios. Takes ~5 minutes.

```bash
python param_search/search.py
```

### 1.3 Run parameter search analysis

Generates search_report.md from latest results.

```bash
python param_search/analyze.py
```

Then copy reports to standard location:
```bash
cp param_search/results/$(ls -t param_search/results/ | head -1)/search_report.md param_search/report.md
cp param_search/results/$(ls -t param_search/results/ | head -1)/search_report_zh.md param_search/report_zh.md
```

### 1.4 Verify main.py output

```bash
python main.py configs/device_910C.json configs/network_910C.json configs/model_deepseekv4.json configs/runtime_deepseekv4.json
```

---

## Phase 2: Multi-Agent Report Writing

After Phase 1 completes, create a **team** with 4 agents working in a coordinated pipeline.

Use `TeamCreate` to create a team named `report-refresh`, then spawn the agents below using the `Task` tool with `team_name: "report-refresh"`.

### Agent A — Recorder (runs first)

**Name:** `recorder`
**Type:** `general-purpose`
**Model:** `sonnet`
**Role:** Generate the report skeleton and update all data tables from JSON.

**Instructions for Agent A:**
1. Read all 10 JSON data files in `report/data/` and all config files in `configs/`
2. Read the current `report/report_en.md` to understand existing structure
3. Generate/update the **report outline** — the 8-section heading skeleton with markdown-style section names:
   - `## 1. Executive Summary` (leave body as placeholder `<!-- Agent C will write this -->`)
   - `## 2. Model Structure` (subsections 2.1, 2.2, 2.3)
   - `## 3. Bottleneck Analysis` (subsections 3.1–3.4)
   - `## 4. Parameter & Scenario Optimization` (subsections 4.1–4.4)
   - `## 5. Key Module Analysis`
   - `### 5.1 mHC Optimization` (subsections)
   - `### 5.2 Attention & KV Cache Analysis` (subsections)
   - `## 6. Deployment Recommendations` (subsections 6.1–6.5)
   - `## 7. Industry Implications` (leave body as placeholder `<!-- Agent C will write this -->`)
   - `## 8. Appendix` (subsections 8.1–8.3)
4. **Update all data tables** in each section with fresh numbers from the JSON data files. Use the data-to-section mapping:

| Section | Data Source |
|---|---|
| **2. Model Structure** | `configs/model_deepseekv4.json`, `v3_comparison.json` |
| **3. Bottleneck Analysis** | `op_analysis.json`, `hardware_comparison.json` |
| **4. Parameter & Scenario Optimization** | `search_results_910C.json`, `search_results_H20.json`, `pd_ratio_analysis.json`, `hardware_comparison.json` |
| **5.1 mHC Analysis** | `mhc_optimization_comparison.json`, `sp_comparison.json` |
| **5.2 Attention & KV Cache** | `kv_cache_scaling.json`, `attention_analysis.json` |
| **6. Deployment Recommendations** | `search_results_*.json`, `pd_ratio_analysis.json` |
| **8. Appendix** | `configs/*.json`, methodology description |

5. Write the result to `report/report_en.md`. Leave analysis prose as placeholders (e.g., `<!-- Agent B: write analysis here -->`) for sections Agent B owns, and `<!-- Agent C will write this -->` for sections 1 and 7.
6. Mark your task as completed when done.

### Agent B — Analyzer (runs after Agent A)

**Name:** `analyzer`
**Type:** `general-purpose`
**Model:** `opus` (deep analytical writing)
**Role:** Act as an **experienced engineer in AI/LLM/AI Infra**. Write expert analysis prose for technical sections.

**Instructions for Agent B:**
1. Read `report/report_en.md` (updated by Agent A with fresh data tables)
2. Read relevant JSON data files for each section you're writing
3. Read the perf model source code (`perf_model/ops.py`, `perf_model/layers.py`, `perf_model/roofline.py`) as needed to understand the underlying calculations
4. Write **analysis prose** for each of these sections, replacing Agent A's placeholders:
   - **Section 2 (Model Structure)**: Explain architectural choices, V4 vs V3 tradeoffs, hardware platform characteristics
   - **Section 3 (Bottleneck Analysis)**: Analyze per-category bottlenecks, interpret op breakdown tables, explain 910C vs H20 differences
   - **Section 4 (Parameter & Scenario Optimization)**: Analyze optimal configs, explain why certain TP/EP/DP work best, P/D ratio trends, hardware comparison insights
   - **Section 5.1 (mHC Analysis)**: Explain mHC optimization levels, kernel fusion impact, SP comparison findings
   - **Section 5.2 (Attention & KV Cache)**: Analyze KV cache scaling, attention breakdown, compression effectiveness
   - **Section 6 (Deployment Recommendations)**: Write actionable deployment guidance for each context length scenario, general guidance
   - **Section 8 (Appendix)**: Update methodology description, data source references
5. **Writing style**: Technical but accessible. Ground every claim in specific numbers from the data tables. Explain *why* results look the way they do, not just *what* they are.
6. Save to `report/report_en.md` and mark task as completed.

### Agent C — Researcher (runs after Agent B)

**Name:** `researcher`
**Type:** `general-purpose`
**Model:** `opus` (deep analytical writing + web research)
**Role:** Act as an **expert and ideaful researcher in AI/LLM/AI Infra**. Write the industry implications and executive summary.

**Instructions for Agent C:**
1. Read the complete `report/report_en.md` (with Agent B's analysis)
2. Read all 10 JSON data files in `report/data/` for quantitative grounding
3. **Write Section 7 (Industry Implications)**:
   - Use **web search** for current industry trends (2025-2026) in: LLM inference optimization, MoE serving systems, KV cache management, P/D disaggregation, hardware for AI inference
   - You may run **additional experiments** using the perf model (e.g., `python main.py` with modified configs) to test hypotheses about performance tradeoffs
   - Write 6 subsections:
     - **7.1 KV Cache Management & Tiered Caching**: How V4's compression changes the KV cache landscape
     - **7.2 P/D Disaggregation Architecture**: What our P/D ratio analysis means for serving system design
     - **7.3 Hardware Design Tradeoffs**: Compute vs bandwidth lessons from 910C vs H20 comparison
     - **7.4 Network Bandwidth for MoE**: EP bandwidth implications for datacenter networking
     - **7.5 mHC as New Paradigm**: What kernel fusion results imply for future model architectures
     - **7.6 Ultra-Long Context Serving**: Practical challenges revealed by 128K/256K scenarios
   - Ground all claims in specific numbers from the analysis data
   - Connect findings to broader industry trends with web search citations
4. **Write Section 1 (Executive Summary)**:
   - Read the complete report (all sections 2-8)
   - Synthesize the 5-6 most important findings with specific numbers
   - Include the analysis method line
   - Keep it concise (15-20 lines)
5. Save to `report/report_en.md` and mark task as completed.

### Agent D — Verifier (runs after Agent C)

**Name:** `verifier`
**Type:** `general-purpose`
**Model:** `sonnet`
**Role:** Quality assurance — verify data accuracy and catch hallucinations.

**Instructions for Agent D:**
1. Read the complete `report/report_en.md`
2. Read **every** JSON data file in `report/data/` and config files in `configs/`
3. **Verify data tables**: For every table in the report, cross-reference each number against the source JSON. Flag any mismatches.
4. **Verify analysis claims**: For every quantitative claim in the prose (e.g., "mHC reduces prefill 4x"), verify the number exists in or can be derived from the data. Flag any unsupported claims.
5. **Check completeness**:
   - [ ] All 10 JSON files referenced appropriately
   - [ ] All 4 combos (8K, 32K, 128K, 256K) appear in Sections 4 and 6
   - [ ] V3 comparison table present in Section 2
   - [ ] KV cache scaling data present in Section 5.2
   - [ ] Attention analysis data present in Section 5.2
   - [ ] No placeholder text remaining (no `<!-- Agent ... -->` comments)
   - [ ] Section numbering is consistent and complete
6. **Fix any issues found** directly in `report/report_en.md`. If a number is wrong, correct it from the JSON source. If a claim is unsupported, either add the supporting data or soften the claim.
7. Report a summary of findings (issues found + fixed) and mark task as completed.

---

## Phase 3: Translation

After Agent D completes, run a **translation agent**:

**Name:** `translator`
**Type:** `general-purpose`
**Model:** `sonnet`

**Instructions:**
1. Read the verified `report/report_en.md`
2. Read the existing `report/report_zh.md` to understand the translation style
3. Translate `report/report_en.md` → `report/report_zh.md`:
   - Translate all prose and headings to Chinese
   - Keep all data tables, numbers, code blocks, and formulas **identical**
   - Keep technical terms in English where standard (e.g., "MoE", "KV cache", "roofline")
4. Similarly update:
   - `report/ppt_outline_en.md` → `report/ppt_outline_zh.md`
5. Mark task as completed.

### 3.1 Generate PDF

After translation completes, convert the Chinese report to PDF:

```bash
python report/md2pdf.py
```

This generates `report/DeepseekV4建模分析报告.pdf` from `report/report_zh.md`.

---

## Phase 4: Update Documentation

After translation, update all remaining documentation:

**Name:** `doc-updater`
**Type:** `general-purpose`
**Model:** `sonnet`

**Instructions:**
1. Read the final `report/report_en.md` and `report/report_zh.md`
2. Update `README.md`:
   - Update key results tables with latest numbers
   - Ensure report section descriptions match the 8-section structure
   - Update any performance numbers referenced
3. Update `README_zh.md`:
   - Mirror all changes from README.md with Chinese prose
   - Keep data/tables identical
4. Update `CLAUDE.md`:
   - Update "Parameter Search" key results section with latest numbers
   - Update any other sections that reference specific performance numbers
5. Verify all bilingual pairs are in sync:
   - `report/report_en.md` ↔ `report/report_zh.md`
   - `report/ppt_outline_en.md` ↔ `report/ppt_outline_zh.md`
   - `README.md` ↔ `README_zh.md`
   - `param_search/report.md` ↔ `param_search/report_zh.md`
6. Mark task as completed.

---

## Agent Dependency Graph

```
Phase 1: Run experiments (sequential bash commands)              [sonnet]
    │
    ▼
Agent A (recorder) ── skeleton + data tables                     [sonnet]
    │
    ▼
Agent B (analyzer) ── technical analysis prose (Sec 2,3,4,5,6,8) [opus]
    │
    ▼
Agent C (researcher) ── industry implications (Sec 7) + summary  [opus]
    │
    ▼
Agent D (verifier) ── data verification + fix issues             [sonnet]
    │
    ▼
Phase 3: translator ── EN → ZH translation                      [sonnet]
    │
    ▼
Phase 3.1: md2pdf.py ── report_zh.md → PDF                      [bash]
    │
    ▼
Phase 4: doc-updater ── README, CLAUDE.md, bilingual sync        [sonnet]
```

## Task Setup

Create these tasks with dependencies using `TaskCreate` and `TaskUpdate`. Use the `model` parameter on each `Task` tool call:

1. **"Run experiments and gather data"** — Phase 1, `model: "sonnet"`
2. **"Generate report skeleton and update data tables"** — Agent A, `model: "sonnet"` (blocked by task 1)
3. **"Write technical analysis for sections 2,3,4,5,6,8"** — Agent B, `model: "opus"` (blocked by task 2)
4. **"Write industry implications and executive summary"** — Agent C, `model: "opus"` (blocked by task 3)
5. **"Verify data accuracy and fix issues"** — Agent D, `model: "sonnet"` (blocked by task 4)
6. **"Translate reports to Chinese"** — translator, `model: "sonnet"` (blocked by task 5)
7. **"Update README, CLAUDE.md, and sync bilingual docs"** — doc-updater, `model: "sonnet"` (blocked by task 6)

---

## Data-to-Section Mapping (Reference)

| Section | Data Source |
|---|---|
| **1. Executive Summary** | All JSON (synthesize key findings) |
| **2. Model Structure** | `configs/model_deepseekv4.json`, `v3_comparison.json` |
| **3. Bottleneck Analysis** | `op_analysis.json`, `hardware_comparison.json` |
| **4. Parameter & Scenario Optimization** | `search_results_910C.json`, `search_results_H20.json`, `pd_ratio_analysis.json`, `hardware_comparison.json` |
| **5.1 mHC Analysis** | `mhc_optimization_comparison.json`, `sp_comparison.json` |
| **5.2 Attention & KV Cache** | `kv_cache_scaling.json`, `attention_analysis.json` |
| **6. Deployment Recommendations** | `search_results_*.json`, `pd_ratio_analysis.json` |
| **7. Industry Implications** | **AI-generated narrative** — analysis data + web search for current industry context |
| **8. Appendix** | `configs/*.json`, methodology description |

## Files Updated by This Command

- `report/report_en.md` ↔ `report/report_zh.md` — Main analysis report (8-section structure)
- `report/DeepseekV4建模分析报告.pdf` — PDF version of `report_zh.md` (generated by `md2pdf.py`)
- `report/ppt_outline_en.md` ↔ `report/ppt_outline_zh.md` — PPT outline
- `README.md` ↔ `README_zh.md` — Project README
- `param_search/report.md` ↔ `param_search/report_zh.md` — Parameter search report
- `CLAUDE.md` — Parameter search key results (no Chinese mirror)
