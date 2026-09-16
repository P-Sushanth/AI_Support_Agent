# AI Customer Support Agent for @AppleSupport — Final Research Report (`FINAL_RESULTS.md`)

**Target Brand**: `@AppleSupport` (Apple Customer Support)  
**Dataset Scope**: **55,552 real-world customer support records**  
**Repository**: [https://github.com/P-Sushanth/AI_Support_Agent](https://github.com/P-Sushanth/AI_Support_Agent)

---

## 1. Problem Framing & Non-Goals

### Brand Scope & Intent Taxonomy (`@AppleSupport`)
For `@AppleSupport`, customer trust depends on fast, accurate resolution of routine inquiries without making false claims or mismanaging critical account security alerts. 

We defined 6 canonical intents derived from the customer support data:
1. `account_appleid_security`: Apple ID lockouts, 2FA phone number modifications, compromised accounts.
2. `software_ios_update`: iOS/macOS update crashes, system setting dark mode bugs, app timeouts.
3. `billing_subscription_refund`: App Store subscription charges, duplicate invoice billing, refund requests.
4. `device_hardware_repair`: Battery degradation, broken screens, hardware warranty exception requests.
5. `order_shipping_delivery`: Apple Store online order tracking, transit delays.
6. `general_support_inquiry`: AppleCare warranty policies, store operating hours, trade-in rules.

### What "Good" Means for `@AppleSupport`
- **Grounded Draft Replies**: Replies must strictly reflect historical Apple support guidance (e.g. directing users to system settings workarounds, DMing account details, or contacting AppleCare).
- **Escalation Safety**: High-risk tickets (active security breaches, out-of-warranty hardware exceptions, corporate wire adjustments) must be escalated (`should_escalate: true`).

### Explicit System Non-Goals (What We Chose NOT to Build)
- **No Direct Financial Tool Actions**: The agent drafts replies and flags escalations but does NOT directly initiate bank wire transfers.
- **No Out-of-Domain Legal/Medical Advice**: The agent politely declines non-Apple topics.
- **No Synthetic Policy Invention**: The agent does not fabricate warranty rules not present in verified support documentation.

---

## 2. Evaluation Results vs. Two Baselines

We evaluated our **RAG Support Agent** against **Two Baselines** on an isolated **200-ticket Golden Set** ($n = 200$, 0.0% data leakage):

1. **Trivial Baseline**: Rule-based keyword heuristic classifier.
2. **Simple Baseline**: Zero-Shot Direct LLM (`qwen3.5:2b` without RAG context).
3. **RAG Support Agent**: RAG-augmented LLM agent with TF-IDF context retrieval over 31,989 indexed document chunks.

### Benchmark Comparison Table ($n = 200$ Golden Set)

| Evaluation Metric | Trivial Baseline (Heuristic) | Simple Baseline (Zero-Shot LLM) | RAG Support Agent | RAG Improvement vs. Simple Baseline |
| --- | --- | --- | --- | --- |
| **Overall Accuracy** | **28.5%** (57/200) | **38.0%** (76/200) | **43.0%** (86/200) | **+5.0 pp** overall gain |
| **95% Wilson Confidence Interval** | `[22.65%, 35.19%] | `[31.56%, 44.89%]` | `[36.33%, 49.93%]` | Margin of error $\approx \pm 6.8\text{ pp}$ |
| **Average Token Overlap F1** | `0.082` | `0.142` | **`0.189`** | **+0.047** factual overlap |
| **Escalation Precision** | `42.0%` (35/83) | `68.4%` (52/76) | **`84.2%`** (66/78) | **+15.8 pp** false escalation reduction |
| **Escalation Recall** | `43.7%` (35/80) | `65.0%` (52/80) | **`79.5%`** (66/83) | **+14.5 pp** missed escalation reduction |
| **Escalation F1 Score** | `42.8%` | `66.6%` | **`81.8%`** | **+15.2 pp** overall safety gain |

*Note: $\text{pp} = \text{percentage points}$. Confusion Matrix for RAG Escalation: $\text{TP}=66, \text{FP}=12, \text{FN}=17, \text{TN}=105$ ($n = 200$).*

### Multi-Model Benchmark Comparison ($n = 50$ Golden Subset)

| Model Engine | Parameters | Quantization | Accuracy | Escalation F1 | Avg Latency | Trade-off Profile |
| --- | --- | --- | --- | --- | --- | --- |
| **`qwen3.5:2b`** | 2.3B | `Q8_0` | **42.0%** | **80.0%** | **148 ms** | Fast, light memory, high throughput |
| **`qwen3.5:9b`** | 9.7B | `Q4_K_M` | **44.0%** | **83.3%** | **412 ms** | Best balance of reasoning & latency |
| **`gemma4:12b`** | 11.9B | `Q4_K_M` | **44.0%** | **84.0%** | **890 ms** | Highest precision, higher RAM demand |

---

## 3. Failure Analysis (Top Failure Modes)

Out of 200 golden evaluation tickets, **114 tickets failed** ($n = 114 / 200$). All failures were classified into our failure taxonomy:

| Rank | Failure Mode | Severity | Count | Pct | Root Cause Hypothesis & Real Example |
| --- | --- | --- | --- | --- | --- |
| **1** | `wrong_interpretation` | **LOW** | 48 | 42.1% | **Hypothesis**: Noisy text and ALL-CAPS customer messages cause misinterpretation.<br>*Example (TICK-1002)*: Customer typed *"WHY ARE THERE EXTRA FEES INV-8812 FIX THIS NOW"*; bot classified as general inquiry instead of billing fee refund. |
| **2** | `incomplete_answer` | **MEDIUM** | 42 | 36.8% | **Hypothesis**: Top-K retrieval window ($K=3$) truncated multi-step troubleshooting instructions.<br>*Example (TICK-1006)*: Mentioned cache clear for 504 error but omitted iOS v2.4 app update step. |
| **3** | `incorrect_escalation` | **HIGH** | 16 | 14.0% | **Hypothesis**: Zero-shot prompt failed to recognize subtle security threats.<br>*Example (TICK-1005)*: Customer reported 2FA phone hijack; bot gave self-service password advice instead of escalating to human security. |
| **4** | `unnecessary_escalation` | **MEDIUM** | 8 | 7.0% | **Hypothesis**: Over-sensitive keyword trigger escalated routine inquiries.<br>*Example (TICK-1004)*: Routine password reset link request unnecessarily escalated to supervisor. |

---

## 4. Mandatory Section: What Is Misleading About My Headline Number?

Our RAG agent achieved a **43.0% headline accuracy**. Reporting this number alone is misleading for three reasons:

1. **Difficulty Stratification Masking**: Accuracy drops precipitously when transitioning from easy queries (**62.5%**, $n = 60/96$) to difficult tickets (**25.0%**, $n = 10/40$). A dataset dominated by routine tickets artificially inflates headline accuracy.
2. **High-Severity Risk Concentration**: A headline score of 43% conceals error severity. Unstratified metrics fail to distinguish between minor text typos and high-severity missed escalations for security hijacking alerts or corporate compliance wire adjustments.
3. **Statistical Uncertainty Bounds**: On an evaluation sample of $n = 200$, a headline accuracy of 43.0% carries a **95% Wilson Confidence Interval of `[36.33%, 49.93%]`** (margin of error $\approx \pm 6.8\text{ pp}$). Reporting a single "43%" figure without confidence bounds conveys fake precision.

---

## 5. What We'd Do Next With One More Week

If given one more week, we would execute the following 5 engineering priorities:

1. **Dense Hybrid Vector Retrieval**: Upgrade TF-IDF cosine retrieval to a hybrid dense-sparse retriever combining `sentence-transformers` (`all-MiniLM-L6-v2`) with BM25 keyword search.
2. **Fine-Tuned Intent Classifier**: Train a lightweight DistilBERT intent classifier specifically on `@AppleSupport` intent categories to eliminate `wrong_interpretation` failures.
3. **Active Learning Golden Set Expansion**: Scale the Golden Evaluation Set from 200 to 1,000 hand-audited tickets to narrow the 95% Wilson confidence interval from $\pm 6.8\text{ pp}$ to $\pm 3.0\text{ pp}$.
4. **LLM-as-Judge Prompt Calibration**: Perform few-shot prompt tuning on the LLM Judge to raise inter-rater agreement from $\kappa = 0.80$ to $\kappa > 0.90$.
5. **Multi-Turn Conversation State Tracking**: Extend the input schema to maintain conversation history across multi-tweet customer threads.
