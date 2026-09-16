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

## 🎯 1. The Core Idea (The "Why")

Imagine a company claims: **"Our AI Customer Support Chatbot is 90% accurate!"**

That sounds great, but in the real world, **that headline number is often misleading**:

- **Why?** What if 85% of customer questions are super easy (like *"What are your store hours?"*), and the chatbot answers those easily...
- **BUT** on critical questions (like *"My account was hacked!"* or *"I was charged $4,000 wrongly"*), the chatbot gets **every single one wrong** or fails to get a human manager?

The company boasts "90% accuracy", but in reality, the bot causes real customer harm and operational risk.

This project takes the opposite approach: **it builds the agent, measures it, and then rigorously proves why its own headline score is misleading.**

---

## 🛠️ 2. What We Built (The System)

We built an end-to-end AI Support System on **55,552 real customer support tickets**:

1. **The AI Support Agent**: Connected to local models via **Ollama** (`qwen3.5:2b`, `qwen3.5:9b`, `gemma4:12b`).
2. **RAG Knowledge Base**: The agent looks up verified company support documents before answering questions to prevent fake claims and hallucinations.
3. **Escalation Engine**: The agent decides whether to answer automatically (`AUTO-RESOLVED`) OR send the ticket to a human manager (`ESCALATE TO HUMAN`) if it detects security hacks, refund overrides, or complex billing issues.

---

## 🏗️ System Architecture

```text
Customer Queries (55,552 Records)
         │
         ▼
┌────────────────────────────────┐
│  Phase 1: Audit & PII Filter   │
└────────┬───────────────────────┴───────────────────────┐
         │                                               │
         ▼ (60% Train Split)                             ▼ (20% Golden Candidate Split)
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

## 🧪 Evaluation Results & Key Findings

The surprising result wasn't just that RAG improved overall accuracy by 5%. It was that **RAG drastically improved safety and escalation precision** on high-risk tickets where the baseline model failed.

### Baseline vs. RAG Benchmark Comparison (200 Golden Tickets)

| Metric | Baseline LLM Agent | RAG Support Agent | Engineering Impact |
| --- | --- | --- | --- |
| **Overall Accuracy** | **38.0%** | **43.0%** | **+5.0%** overall gain |
| **95% Wilson Confidence Interval** | `[31.56%, 44.89%]` | `[36.33%, 49.93%]` | Quantified statistical bounds |
| **Average Token Overlap F1** | `0.142` | **`0.189`** | **+0.047** factual similarity |
| **Escalation Precision** | `68.4%` | **`84.2%`** | **+15.8%** reduction in false escalations |
| **Escalation Recall** | `65.0%` | **`79.5%`** | **+14.5%** reduction in missed security alerts |
| **Escalation F1 Score** | `66.6%` | **`81.8%`** | **+15.2%** overall safety improvement |

---

## 🔬 Evaluation Methodology & Differentiators

Rather than relying on basic string matching or uncalibrated chatbot outputs, this system implements production evaluation techniques:

### 1. Isolated Golden Set & Zero-Leakage Audit (Phase 5)
- **200 stratified evaluation examples** sampled across difficulty levels (easy, medium, hard) and support categories.
- Verified **0.0% data leakage** (0 evaluation examples exist in the RAG retrieval index).

### 2. Multi-Dimensional Metrics & Wilson CIs (Phase 6)
- **Wilson Score 95% Confidence Intervals**:
  $$\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
- **Escalation Precision, Recall & F1**: Specifically measuring high-risk policy triggers.

### 3. LLM-as-Judge & Inter-Rater Agreement (Phase 7)
- Structured evaluation rubrics measuring Correctness, Relevance, Faithfulness, Completeness, and Escalation.
- Inter-rater agreement measured via **Cohen's Kappa ($\kappa$)**:
  $$\kappa = \frac{p_o - p_e}{1 - p_e}$$

### 4. Systematic Failure Taxonomy (Phase 8)
Classifies all 114 evaluation failures into 8 actionable categories:
- **`incorrect_escalation` (HIGH)**: Missed mandatory security or compliance escalation.
- **`unnecessary_escalation` (MEDIUM)**: Over-escalated routine query.
- **`incomplete_answer` (MEDIUM)**: Answer missed critical resolution steps.
- **`wrong_interpretation` (LOW)**: Intent misinterpretation.

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
│   └── evaluate.py            # Evaluation pipeline runner
└── tests/                     # 22 unit tests across all components
```
