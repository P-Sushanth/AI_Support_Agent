import os
import json
import pandas as pd
from typing import List, Dict, Any

TAXONOMY_DESCRIPTIONS = {
    "incorrect_escalation": "Agent failed to escalate a security compromise, swollen battery hazard, or legal notice ticket.",
    "unnecessary_escalation": "Agent unnecessarily escalated a routine iOS settings or troubleshooting query.",
    "retrieval_failure": "Relevant Apple resolution context existed in train corpus but was not retrieved.",
    "hallucination": "Agent fabricated an unverified Apple warranty or refund policy.",
    "incomplete_answer": "Agent response missed critical required resolution steps.",
    "wrong_interpretation": "Agent misinterpreted customer intent.",
    "dataset_ambiguity": "Customer tweet in dataset was too noisy, ambiguous, or incomplete.",
    "judge_error": "LLM judge incorrectly scored a valid agent response."
}

def categorize_failure(agent_res: Dict[str, Any], golden_rec: Dict[str, Any]) -> Dict[str, Any]:
    output = agent_res["output"]
    pred_esc = output.get("should_escalate", False)
    true_esc = golden_rec.get("should_escalate", False)
    
    pred_intent = output.get("intent", "").lower()
    true_intent = golden_rec.get("ground_truth_intent", "").lower()
    
    if true_esc and not pred_esc:
        failure_type = "incorrect_escalation"
        severity = "high"
        root_cause = "Agent missed mandatory escalation trigger (security hack, swollen battery, or legal notice)."
    elif not true_esc and pred_esc:
        failure_type = "unnecessary_escalation"
        severity = "medium"
        root_cause = "Agent over-escalated a simple self-service troubleshooting query."
    elif pred_intent and true_intent and pred_intent != true_intent:
        failure_type = "wrong_interpretation"
        severity = "medium"
        root_cause = "Agent misclassified incoming customer tweet intent."
    elif len(output.get("response", "").strip()) < 15:
        failure_type = "incomplete_answer"
        severity = "medium"
        root_cause = "Agent produced an overly brief or truncated answer."
    else:
        failure_type = "retrieval_failure"
        severity = "low"
        root_cause = "Agent response failed to match ground truth resolution steps."

    return {
        "ticket_id": golden_rec.get("ticket_id") or golden_rec.get("id", "UNKNOWN"),
        "category": golden_rec.get("category", "GENERAL"),
        "difficulty": golden_rec.get("difficulty", "medium"),
        "failure_type": failure_type,
        "severity": severity,
        "agent_response": output.get("response"),
        "expected_behavior": golden_rec.get("ground_truth_response") or golden_rec.get("reference_answer"),
        "root_cause": root_cause,
        "taxonomy_description": TAXONOMY_DESCRIPTIONS.get(failure_type, "")
    }

def run_failure_analysis():
    os.makedirs("results/failures", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    golden_records = []
    with open("data/golden/golden_set.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_records.append(json.loads(line))
                
    golden_map = {g["ticket_id"]: g for g in golden_records}
    
    rag_results = []
    if os.path.exists("results/rag/golden_eval_results.jsonl"):
        with open("results/rag/golden_eval_results.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rag_results.append(json.loads(line))
                    
    failures = []
    for res in rag_results:
        g = golden_map.get(res["ticket_id"])
        if not g:
            continue
            
        output = res["output"]
        pred_esc = output.get("should_escalate", False)
        true_esc = g.get("should_escalate", False)
        pred_intent = output.get("intent", "").lower()
        true_intent = g.get("ground_truth_intent", "").lower()
        
        if pred_esc != true_esc or pred_intent != true_intent or len(output.get("response", "").strip()) < 15:
            fail_rec = categorize_failure(res, g)
            failures.append(fail_rec)
            
    with open("results/failures/failures.jsonl", "w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(fail, ensure_ascii=False) + "\n")
            
    if failures:
        df_fail = pd.DataFrame(failures)
        df_fail.to_csv("results/failures/failure_summary.csv", index=False)
        
    print(f"Failure Analysis Complete! Categorized {len(failures)} failures out of {len(rag_results)} evaluated items.")
    return failures

if __name__ == "__main__":
    run_failure_analysis()
