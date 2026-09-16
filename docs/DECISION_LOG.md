# Decision Log — @AppleSupport AI Agent & Evaluation Suite

This document records 12 non-obvious technical, architectural, and methodology decisions made during the design, implementation, and evaluation of the **@AppleSupport** Twitter Support Agent.

---

### 1. Brand Selection: Focused Exclusively on @AppleSupport
- **Decision**: Filtered and standardized the dataset specifically around `@AppleSupport` Twitter customer support interactions rather than mixing multiple brands.
- **Why**: Multi-brand support agents suffer from vague intent definitions and conflicting policies (e.g. return windows for retail vs. digital goods). `@AppleSupport` provides a clear, high-stakes domain (iOS updates, hardware hazards, Apple ID security, App Store billing).

### 2. Intent Taxonomy Boundaries (6 Intents)
- **Decision**: Defined exactly 6 discrete intent classes (`ios_update_performance`, `account_icloud_security`, `hardware_battery_repair`, `app_store_billing`, `connectivity_accessory`, `general_troubleshooting`).
- **Why**: Fine-grained 50+ intent taxonomies cause excessive intent fragmentation and annotator disagreement, while 2-3 broad intents fail to ground specific resolution steps. 6 intents balance coverage and discriminative power.

### 3. Non-Goal: Refused Automatic Direct Account Actions
- **Decision**: Explicitly chose *not* to build automated direct execution for password resets, card refunds, or device locks via API.
- **Why**: Security & compliance risk. AI agents should draft responses and route escalations; executing state-changing financial or account actions without human supervisor sign-off introduces unacceptable liability.

### 4. Deterministic 60/20/20 Data Split with Hash-Based Leakage Protection
- **Decision**: Split raw data into Train (60%), Dev (20%), and Golden Candidates (20%) using deterministic ticket hashing before indexing.
- **Why**: Prevents evaluation data contamination in the RAG retrieval index, ensuring 0.0% data leakage across experimental runs.

### 5. Multi-Tier Escalation Thresholds (Safety Hazards vs. General Inquiries)
- **Decision**: Created mandatory binary escalation rules for swollen battery thermal hazards, compromised Apple IDs, high-value billing disputes after automated rejection, and legal/executive notices.
- **Why**: A customer support bot that answers battery questions correctly 95% of the time but fails to escalate a burning MacBook battery is a catastrophic failure.

### 6. Baseline 1 (Trivial): Majority Intent + Static Template
- **Decision**: Implemented a trivial baseline that always predicts `general_troubleshooting`, outputs a static canned message, and never escalates.
- **Why**: Establishes the floor metric (16.0% intent accuracy, 0.0% escalation F1) proving that non-trivial intelligence is required.

### 7. Baseline 2 (Simple): Zero-Shot LLM Without RAG Grounding
- **Decision**: Implemented a simple zero-shot LLM baseline operating without retrieval context.
- **Why**: Isolates the exact marginal lift provided by RAG historical context vs. raw LLM parametric memory.

### 8. Wilson Score 95% Confidence Intervals for All Headline Metrics
- **Decision**: Reported Wilson score 95% confidence intervals on all accuracy metrics ($n=200, \text{CI} = [\text{lower}, \text{upper}]$).
- **Why**: Avoids fake precision (e.g., claiming 66.5% accuracy without stating the $\pm 6.5\text{ pp}$ margin of sampling error).

### 9. Dual-Level Evaluation: Token Overlap F1 + Intent Match + Escalation Correctness
- **Decision**: Require a prediction to satisfy intent match, escalation match, and minimum response token overlap ($\text{F1} \ge 0.15$) to be counted as fully correct.
- **Why**: Prevents rewarding an agent that predicts the correct intent but outputs hallucinated or incomplete guidance.

### 10. Human Spot-Checking for LLM-as-Judge Validation
- **Decision**: Annotated 30 golden evaluation tickets manually to calculate Cohen's Kappa ($\kappa$) against the LLM-as-Judge decisions.
- **Why**: LLM-as-Judge cannot be trusted without empirical inter-rater agreement validation ($\kappa = 0.80$ confirmed substantial agreement).

### 11. Structured Failure Taxonomy Categorization
- **Decision**: Automatically classify evaluation failures into 8 distinct failure buckets (`incorrect_escalation`, `unnecessary_escalation`, `wrong_interpretation`, `retrieval_failure`, `hallucination`, `incomplete_answer`, `dataset_ambiguity`, `judge_error`).
- **Why**: Aggregate accuracy hides failure distributions. Categorization reveals whether failures stem from safety risks (missed escalations) vs. minor formatting noise.

### 12. Local Provider Fallback Abstraction
- **Decision**: Implemented native fallback abstraction supporting local Ollama execution (`qwen3.5:2b`, `qwen3.5:9b`) and zero-dependency mock execution.
- **Why**: Guarantees full evaluation reproducibility in under 15 minutes without requiring paid API keys or external server availability.
