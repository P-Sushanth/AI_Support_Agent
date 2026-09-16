import os
import json
import pandas as pd
from typing import List, Dict, Any

TAXONOMY_DESCRIPTIONS = {
    "incorrect_escalation": "Agent failed to escalate a security, corporate tax, or policy exception ticket.",
    "unnecessary_escalation": "Agent unnecessarily escalated a routine self-service support query.",
    "retrieval_failure": "Relevant support context existed in train corpus but was not retrieved.",
    "hallucination": "Agent fabricated an unverified policy or refund claim.",
    "incomplete_answer": "Agent response missed critical required resolution steps.",
    "wrong_interpretation": "Agent misinterpreted customer intent.",
    "dataset_ambiguity": "Customer query in dataset was too noisy, ambiguous, or incomplete.",
    "judge_error": "LLM judge incorrectly scored a valid agent response."
}

def categorize_failure(agent_res: Dict[str, Any], golden_rec: Dict[str, Any]) -> Dict[str, Any]:
    output = agent_res["output"]
    pred_esc = output.get("should_escalate", False)
    true_esc = golden_rec.get("should_escalate", False)
    
    if true_esc and not pred_esc:
        failure_type = "incorrect_escalation"
        severity = "high"
        root_cause = "Agent missed mandatory escalation trigger (security alert, corporate wire, or out-of-policy exception)."
    elif not true_esc and pred_esc:
        failure_type = "unnecessary_escalation"
        severity = "medium"
        root_cause = "Agent over-escalated a simple self-service query."
    elif len(output.get("response", "").strip()) < 15:
        failure_type = "incomplete_answer"
        severity = "medium"
        root_cause = "Agent produced an overly brief or truncated answer."
    else:
        failure_type = "wrong_interpretation"
        severity = "low"
        root_cause = "Agent response failed to match expected resolution semantics."

    return {
        "example_id": golden_rec["id"],
        "ticket_id": golden_rec["original_ticket_id"],
        "category": golden_rec.get("category", "GENERAL"),
        "difficulty": golden_rec.get("difficulty", "medium"),
        "failure_type": failure_type,
        "severity": severity,
        "agent_response": output.get("response"),
        "expected_behavior": golden_rec.get("reference_answer"),
        "root_cause": root_cause,
        "taxonomy_description": TAXONOMY_DESCRIPTIONS.get(failure_type, "")
    }

def run_failure_analysis(eval_summary_path: str = "results/metrics/evaluation_summary.json"):
    os.makedirs("results/failures", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    golden_records = []
    with open("data/golden/golden_set.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_records.append(json.loads(line))
                
    golden_map = {g["id"]: g for g in golden_records}
    
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
        
        # Check if failed
        if pred_esc != true_esc or len(output.get("response", "").strip()) < 15:
            fail_rec = categorize_failure(res, g)
            failures.append(fail_rec)
            
    # Save failures.jsonl
    with open("results/failures/failures.jsonl", "w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(fail, ensure_ascii=False) + "\n")
            
    # Save failure summary CSV
    if failures:
        df_fail = pd.DataFrame(failures)
        df_fail.to_csv("results/failures/failure_summary.csv", index=False)
        
    # Generate failure taxonomy markdown
    taxonomy_doc = "# Failure Taxonomy & Analysis\n\n"
    taxonomy_doc += f"Total Identified Failures: **{len(failures)}** out of {len(rag_results)} evaluated tickets.\n\n"
    taxonomy_doc += "## Failure Types Breakdown\n\n"
    
    type_counts = {}
    for f in failures:
        ft = f["failure_type"]
        type_counts[ft] = type_counts.get(ft, 0) + 1
        
    taxonomy_doc += "| Failure Type | Count | Severity | Description |\n| --- | --- | --- | --- |\n"
    for ft, cnt in type_counts.items():
        sev = "high" if "incorrect_escalation" in ft else ("medium" if "unnecessary" in ft else "low")
        desc = TAXONOMY_DESCRIPTIONS.get(ft, "")
        taxonomy_doc += f"| `{ft}` | {cnt} | `{sev}` | {desc} |\n"
        
    with open("results/failures/failure_taxonomy.md", "w", encoding="utf-8") as f:
        f.write(taxonomy_doc)
        
    # Document docs/failure_analysis.md
    with open("docs/failure_analysis.md", "w", encoding="utf-8") as f:
        f.write(taxonomy_doc)
        
    print(f"Failure analysis completed! Categorized {len(failures)} failures.")
    return failures

if __name__ == "__main__":
    run_failure_analysis()
