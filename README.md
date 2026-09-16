# AI Customer Support Agent & Evaluation Suite

An AI Customer Support Agent and Evaluation Suite built on **55,552 real-world customer support records**, designed to investigate how standard evaluation metrics misrepresent production readiness.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Engine](https://img.shields.io/badge/Ollama-qwen3.5%20%7C%20gemma4-orange)
![Architecture](https://img.shields.io/badge/RAG-TF--IDF%20Vector%20Search-green)
![Dataset](https://img.shields.io/badge/Dataset-55%2C552%20Tickets-purple)
![Tests](https://img.shields.io/badge/Tests-22%2F22%20Passing-brightgreen)

---

## ⚡ Quickstart (Run in 30 Seconds)

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/P-Sushanth/AI_Support_Agent.git
cd AI_Support_Agent
pip install -e .

# 2. Launch interactive terminal CLI with local Ollama models
python -m scripts.cli
```

> **CLI Commands**:
> - `/model` — Switch local Ollama models (`qwen3.5:2b`, `qwen3.5:9b`, `gemma4:12b`)
> - `/mode` — Toggle between `[RAG AGENT]` (with document citations) and `[BASELINE LLM]`
> - `/sample` — Load a random ticket from the isolated golden evaluation set

---

## 🎯 Executive Summary & Investigation Focus

A single aggregate metric can hide serious failure modes. A support agent may perform well on routine requests while failing disproportionately on ambiguous, difficult, or high-risk tickets.

This project investigates that problem empirically by building an AI support agent on **55,552 real records**, evaluating it on an isolated golden set ($n = 200$), analyzing its failure modes, and stress-testing the headline metric against different evaluation slices.

When evaluated on our 200-ticket golden set, our RAG agent achieved a **43.0% headline accuracy**. However, subgroup analysis reveals three critical evaluation insights:

1. **Difficulty Stratification Masking**: Accuracy drops precipitously when transitioning from easy queries (**62.5%**, $n = 60/96$) to difficult tickets (**25.0%**, $n = 10/40$). A high proportion of routine tickets in a dataset artificially inflates the overall score ($96 + 64 + 40 = 200$ tickets).
2. **High-Severity Failure Risk**: An aggregate score of 43% conceals failure distribution. Unstratified metrics do not distinguish between an auto-resolved routine question and a missed mandatory escalation for account takeover alerts or corporate compliance wire adjustments.
3. **Statistical Uncertainty Bounds**: On an evaluation sample of $n = 200$, a headline accuracy of 43.0% carries a **95% Wilson Confidence Interval of `[36.33%, 49.93%]`** (margin of error $\approx \pm 6.8$ percentage points).
4. **Escalation Performance**: Grounding the agent with RAG knowledge retrieval substantially improved escalation performance on evaluated high-risk cases (**+15.2 pp Escalation F1**).

---

## 🏗️ System Architecture

```text
Customer Inquiries (55,552 Records)
         │
         ▼
┌────────────────────────────────┐
│  Phase 1: Audit & PII Filter   │
└────────┬───────────────────────┴───────────────────────┐
         │                                               │
         ▼ (60% Knowledge / Retrieval Split)             ▼ (20% Golden Candidate Split)
┌────────────────────────────────┐               ┌────────────────────────────────┐
│  Phase 4: RAG Vector Index     │               │  Phase 5: Golden Set (200 Recs)│
│  [31,989 Documents Indexed]    │               │  [0.0% Data Leakage Verified]  │
└────────┬───────────────────────┘               └───────────────┬────────────────┘
         │                                                       │
         └───────────────────────┬───────────────────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │ Phase 3 & 4: Ollama Engine    │
                 │ (qwen3.5:2b, 9b / gemma4:12b) │
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │ Phase 6 & 7: Multi-Metric Eval│
                 │ & LLM-as-Judge (Cohen's κ)    │
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │ Phase 8 & 9: Failure Taxonomy │
                 │ Classifier & Stress Testing   │
                 └───────────────┬───────────────┘
```

---

## 🧪 Evaluation Results

RAG substantially improved escalation performance on evaluated high-risk cases compared to the baseline LLM.

### Baseline vs. RAG Benchmark Comparison ($n = 200$ Golden Tickets)

| Metric | Baseline | RAG | Change |
| --- | --- | --- | --- |
| **Overall Accuracy** | **38.0%** | **43.0%** | **+5.0 pp** |
| **95% Wilson Confidence Interval** | `[31.56%, 44.89%]` | `[36.33%, 49.93%]` | Margin of error $\approx \pm 6.8\text{ pp}$ |
| **Average Token Overlap F1** | `0.142` | **`0.189`** | **+0.047** |
| **Escalation Precision** | `68.4%` | **`84.2%`** | **+15.8 pp** |
| **Escalation Recall** | `65.0%` | **`79.5%`** | **+14.5 pp** |
| **Escalation F1 Score** | `66.6%` | **`81.8%`** | **+15.2 pp** |

*Note: $\text{pp} = \text{percentage points}$. Confusion Matrix for RAG Escalation: $\text{TP}=66, \text{FP}=12, \text{FN}=17, \text{TN}=105$ ($n = 200$).*

---

## 🔬 Evaluation Methodology & Differentiators

Rather than relying on uncalibrated chatbot outputs, this system implements production evaluation techniques:

### 1. Isolated Golden Set & Zero-Leakage Audit (Phase 5)
- **200 stratified evaluation examples** sampled across difficulty levels (easy, medium, hard) and support categories.
- Verified **0.0% data leakage** (0 evaluation examples exist in the RAG retrieval index).

### 2. Multi-Dimensional Metrics & Wilson CIs (Phase 6)
- **Wilson Score 95% Confidence Intervals**:
  $$\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
- **Escalation Precision, Recall & F1**: Specifically measuring high-risk policy triggers.

### 3. LLM-as-Judge & Inter-Rater Agreement (Phase 7)
- Structured evaluation rubrics measuring Correctness, Relevance, Faithfulness, Completeness, and Escalation.
- Inter-rater agreement measured via **Cohen's Kappa ($\kappa$)**: $\kappa = 0.80$ (90.0% agreement rate on 50 human spot checks).

### 4. Systematic Failure Taxonomy (Phase 8)
Defines an 8-category failure taxonomy, with 4 observed categories in our evaluation ($n = 114 / 200$):
- **`wrong_interpretation` (LOW)**: 48 cases (42.1%)
- **`incomplete_answer` (MEDIUM)**: 42 cases (36.8%)
- **`incorrect_escalation` (HIGH)**: 16 cases (14.0%)
- **`unnecessary_escalation` (MEDIUM)**: 8 cases (7.0%)

---

## 🛠️ Full Pipeline Reproduction

Execute any phase of the pipeline via CLI:

```bash
# Data Audit & Split Partitioning (Phase 1)
python -m scripts.audit_dataset

# Build RAG Knowledge Index (Phase 4)
python -m scripts.build_index

# Generate Golden Set & Check Data Leakage (Phase 5)
python -m scripts.generate_golden_set

# Run Full Evaluation Suite (Phase 6)
python -m scripts.evaluate

# Run Empirical Judge Validation vs Human Annotations (Phase 7)
python -m scripts.validate_judge

# Run Failure Classifier & Stress Testing (Phases 8 & 9)
python -m src.analysis.failures
python -m src.analysis.stress_test

# Run Complete Test Suite (22 Unit Tests)
pytest
```

---

## 🛡️ Automated Test Suite Verification

Includes **22 automated unit tests** passing in `<9s`:

| Test Module | Coverage & Functionality Verified |
| --- | --- |
| `tests/test_data.py` | Data audit schema, PII pattern detection, split partitioning, 0% leakage |
| `tests/test_agent.py` | Input/output schemas, security escalation logic, caching, batch inference, CLI loader |
| `tests/test_retrieval.py` | Chunking, TF-IDF index building/loading, top-K search, citation propagation |
| `tests/test_evaluation.py` | Wilson CIs, token overlap F1, escalation precision/recall/F1 metrics |
| `tests/test_judge.py` | LLM judge scoring rubrics & Cohen's kappa ($\kappa$) agreement |
| `tests/test_failures.py` | Failure taxonomy classification logic |

---

## 📁 Repository Structure

```text
AI_Support_Agent/
├── pyproject.toml             # Dependencies & pytest config
├── configs/                   # System & dataset YAML configs
├── prompts/                   # Versioned system & judge prompts
├── src/
│   ├── agent/                 # Ollama LLM client, baseline & RAG agents
│   ├── data/                  # Audit, cleaning & leakage isolation
│   ├── retrieval/             # Knowledge chunking & TF-IDF retriever
│   ├── evaluation/            # Wilson CIs, F1 & escalation metrics
│   ├── judging/               # LLM-as-Judge & Cohen's kappa (κ)
│   └── analysis/              # Failure taxonomy & stress testing
├── scripts/
│   ├── cli.py                 # Terminal CLI with live model switcher
│   ├── audit_dataset.py       # Data audit script
│   ├── build_index.py         # RAG index builder
│   ├── generate_golden_set.py # Golden set generator
│   ├── evaluate.py            # Evaluation pipeline runner
│   ├── validate_judge.py      # LLM Judge vs Human validation runner
│   └── compare_models.py      # Multi-model benchmark runner
└── tests/                     # 22 unit tests across all components
```
