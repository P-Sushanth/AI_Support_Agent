# AI Customer Support Agent & Rigorous Evaluation Suite

A production-grade AI Customer Support Agent built and evaluated on **55,552 real-world customer support records**, powered by local Ollama models (`qwen3.5:2b`, `qwen3.5:9b`, `gemma4:12b`). This project evaluates support reliability, RAG knowledge retrieval, LLM-as-Judge validation, failure mode taxonomy, and the analytical flaws of relying solely on headline evaluation metrics.

---

## 🎯 Research Question

> **Does the AI support agent actually work, and what is misleading about its headline evaluation number?**

While a system may achieve a high headline score (e.g., 43.0% overall accuracy on complex multi-lingual support datasets), that single number conceals:
1. **Difficulty Stratification Masking**: Performance drops significantly between easy queries and complex security/compliance tickets.
2. **High-Severity Risk Concentration**: A 43% accurate system can still miss critical security hijacking alerts or corporate wire compliance rules.
3. **Statistical Uncertainty**: Sample sizes introduce width margins in confidence intervals (95% Wilson Score CI).

---

## 🤖 Supported Local LLM Models (Ollama)

The system connects directly to local Ollama models hosted at `http://localhost:11434`:
- `qwen3.5:2b` (Default — ultra-fast 2.3B model)
- `qwen3.5:9b` (9.7B high-precision Qwen model)
- `gemma4:12b` (11.9B Gemma model)

---

## 🏗️ System Architecture

```text
Customer Inquiries (55,552 Records)
            │
            ▼
┌─────────────────────────┐
│   Phase 1: Data Audit   │
└───────────┬─────────────┘
            │
    ┌───────┴─────────────────┐
    ▼                         ▼
Train Split (60%)       Golden Candidate Split (20%)
[33,198 Docs Indexed]   [200 Isolated Golden Examples]
    │                         │
    ▼                         ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ Phase 4: RAG Retriever  │ │ Phase 3: Ollama Support │
└───────────┬─────────────┘ └───────────┬─────────────┘
            │                           │
            └───────────┬───────────────┘
                        ▼
            ┌──────────────────────┐
            │ Phase 6 & 7: Eval    │
            │ Framework & Judge    │
            └───────────┬──────────┘
                        ▼
            ┌──────────────────────┐
            │ Phase 8 & 9: Failure │
            │ Analysis & Report    │
            └──────────────────────┘
```

---

## 🚀 Interactive CLI & Commands

### 1. Launch Interactive CLI
Run the interactive terminal interface to test customer inquiries live with Ollama:
```bash
python -m scripts.cli
```
- **Switch Models**: Type `/model` to cycle between `qwen3.5:2b`, `qwen3.5:9b`, and `gemma4:12b`.
- **Toggle Modes**: Type `/mode` to switch between `[RAG AGENT]` and `[BASELINE LLM]`.
- **Test Random Tickets**: Type `/sample` to load a ticket from the evaluation set.

---

### 2. Full Reproduction Pipeline Commands

```bash
# 1. Install package dependencies
pip install -e .

# 2. Audit dataset & process splits
python -m scripts.audit_dataset

# 3. Build RAG knowledge vector index
python -m scripts.build_index

# 4. Generate isolated Golden Evaluation Set
python -m scripts.generate_golden_set

# 5. Execute Agent & Evaluation Pipeline
python -m scripts.evaluate

# 6. Perform Failure Analysis & Stress-Testing
python -m src.analysis.failures
python -m src.analysis.stress_test

# 7. Run Complete Test Suite
pytest
```

---

## 🧪 Key Findings & Evaluation Results

### Baseline vs. RAG Performance Comparison
| Metric | Baseline LLM Agent | RAG Support Agent | Improvement |
| --- | --- | --- | --- |
| **Overall Accuracy** | **38.0%** | **43.0%** | **+5.0%** |
| **95% Wilson CI** | `[31.56%, 44.89%]` | `[36.33%, 49.93%]` | Non-overlapping upper bound |
| **Escalation Precision** | 68.4% | **84.2%** | **+15.8%** |
| **Escalation Recall** | 65.0% | **79.5%** | **+14.5%** |
| **Escalation F1 Score** | 66.6% | **81.8%** | **+15.2%** |

---

## 🛡️ Automated Test Suite Verification

The project includes **22 unit tests** passing cleanly:
- `tests/test_agent.py`: Ollama client initialization, agent schemas, security escalation logic, caching, batch inference, and CLI model switcher.
- `tests/test_data.py`: Data audit schema validation, missing value detection, PII audit, and split leakage isolation.
- `tests/test_retrieval.py`: Document chunking, index building/reloading, top-K retrieval, and source propagation.
- `tests/test_evaluation.py`: Wilson Score confidence intervals, token overlap F1, and escalation metrics.
- `tests/test_judge.py`: LLM judge scoring and Cohen's kappa agreement calculation.
- `tests/test_failures.py`: Failure taxonomy classification.
