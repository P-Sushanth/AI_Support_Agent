# Project Decision Log — Non-Obvious Architecture & Evaluation Choices

A record of **15 non-obvious engineering and research decisions** made during the development and evaluation of the AI Support Agent for `@AppleSupport`.

---

### 1. Target Brand Selection (`@AppleSupport`)
- **Decision**: Focused the agent specifically on `@AppleSupport` domain intents rather than a generic multi-brand chatbot.
- **Rationale**: Real-world brand support requires explicit domain constraints (e.g. Apple ID security, iOS updates, battery hardware exceptions) to evaluate escalation accuracy meaningfully.

### 2. Multi-Model Local Ollama Engine
- **Decision**: Integrated native Ollama API support for `qwen3.5:2b`, `qwen3.5:9b`, and `gemma4:12b` instead of cloud APIs alone.
- **Rationale**: Enables zero-cost, offline, reproducible batch evaluations while benchmarking precision vs latency trade-offs across model parameter scales.

### 3. Isolated Evaluation Split (0% Data Leakage Enforced)
- **Decision**: Partitioned the 55,552 records into a 60% Knowledge/Retrieval split and a 20% Golden Candidate split before building the vector index.
- **Rationale**: Prevents data leakage where the RAG vector index contains identical questions to the evaluation set.

### 4. 200-Ticket Stratified Golden Set Size
- **Decision**: Capped the Golden Evaluation Set at 200 hand-audited tickets sampled across easy, medium, and hard difficulty levels.
- **Rationale**: 200 tickets provides a tight Wilson 95% Confidence Interval ($\pm 6.8\text{ pp}$) while keeping multi-model evaluation fast and reproducible (<2 minutes per benchmark run).

### 5. Inclusion of Two Distinct Baselines
- **Decision**: Evaluated RAG against **Two Baselines**: a Trivial Heuristic Baseline (most frequent class / keyword rules) and a Simple Baseline (Zero-Shot Direct LLM).
- **Rationale**: Proves whether RAG retrieval provides a genuine safety gain over simple LLM prompting.

### 6. Escalation Precision & Recall over Simple Accuracy
- **Decision**: Primary safety metric focused on Escalation Precision, Recall, and F1 rather than overall accuracy alone.
- **Rationale**: A missed security hack escalation (False Negative) is far more dangerous to a brand than an auto-resolved routine query.

### 7. Wilson Score 95% Confidence Intervals over Point Estimates
- **Decision**: Reported all accuracy metrics with 95% Wilson Score Confidence Intervals.
- **Rationale**: Prevents false precision and quantifies statistical uncertainty bounds ($n = 200$).

### 8. Use of Percentage Points (`pp`) for Delta Metrics
- **Decision**: Reported metric improvements using percentage points (`pp`) rather than relative percentage gains.
- **Rationale**: An increase from 68.4% to 84.2% is mathematically $+15.8\text{ pp}$, preventing misleading relative percentage inflation.

### 9. Empirical LLM-as-Judge Inter-Rater Agreement ($\kappa = 0.80$)
- **Decision**: Validated the LLM Judge against 50 human spot-check annotations using Cohen's kappa coefficient ($\kappa$).
- **Rationale**: Proves empirically that the LLM Judge aligns with human judgment rather than blindly trusting judge scores.

### 10. TF-IDF Cosine Vector Search for Local RAG Knowledge
- **Decision**: Used TF-IDF vector search with n-grams (1, 2) over 31,989 indexed document chunks.
- **Rationale**: Delivers deterministic, zero-latency, sub-millisecond retrieval without GPU overhead or embedding drift.

### 11. 8-Category Theoretical Failure Taxonomy
- **Decision**: Built an 8-category failure classifier classifying errors into `incorrect_escalation`, `unnecessary_escalation`, `incomplete_answer`, `wrong_interpretation`, etc.
- **Rationale**: Enables actionable root-cause failure analysis beyond counting errors.

### 12. Structured JSON Output Enforcement
- **Decision**: Enforced Pydantic schema validation (`SupportAgentOutput`) with explicit `format="json"` in Ollama prompts.
- **Rationale**: Ensures 100% machine-readable outputs for automated batch evaluation pipelines.

### 13. Disk-Based SHA-256 Response Caching
- **Decision**: Implemented SHA-256 hash caching for LLM prompts in `results/baseline/cache/`.
- **Rationale**: Eliminates redundant LLM API calls during repeated metric evaluation runs.

### 14. Terminal CLI Model Switcher
- **Decision**: Built an interactive CLI (`scripts/cli.py`) with `/model` and `/mode` commands.
- **Rationale**: Allows instant manual spot-checking of live local Ollama models on real support tickets.

### 15. Clean Repository File Isolation
- **Decision**: Ignored internal markdown logs in `.gitignore` while tracking only `README.md` and `docs/` on GitHub.
- **Rationale**: Keeps the root repository clean, professional, and easy for evaluators to navigate.
