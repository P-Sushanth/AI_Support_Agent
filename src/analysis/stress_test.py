import os
import json
import pandas as pd
from typing import Dict, Any

def run_stress_testing_experiments(eval_summary_path: str = "results/metrics/evaluation_summary.json") -> Dict[str, Any]:
    os.makedirs("results", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    with open(eval_summary_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    baseline = eval_data["baseline"]
    rag = eval_data["rag"]
    
    # 1. Headline Metric vs Confidence Interval
    headline_acc = rag["accuracy"] * 100
    ci_low, ci_high = [round(x * 100, 2) for x in rag["accuracy_ci_95"]]
    
    # 2. Difficulty Stratification
    diff_strat = rag["difficulty_stratification"]
    easy_acc = diff_strat.get("easy", {}).get("accuracy", 0.0) * 100
    hard_acc = diff_strat.get("hard", {}).get("accuracy", 0.0) * 100
    difficulty_gap = round(easy_acc - hard_acc, 2)
    
    # 3. Category Stratification
    cat_strat = rag["category_stratification"]
    cat_rows = []
    for cat, data in cat_strat.items():
        cat_rows.append({
            "category": cat,
            "total_tickets": data["total"],
            "correct_tickets": data["correct"],
            "accuracy_pct": round(data["accuracy"] * 100, 2)
        })
    df_cat = pd.DataFrame(cat_rows)
    df_cat.to_csv("results/final_comparison.csv", index=False)
    
    # 4. Severe Escalation Failure Risk Concentration
    esc_metrics = rag["escalation_metrics"]
    missed_escalations = esc_metrics["fn"]
    false_escalations = esc_metrics["fp"]
    
    experiments_report = {
        "experiment_1_headline_metric": {
            "headline_accuracy_pct": headline_acc,
            "sample_size": rag["sample_size"],
            "confidence_interval_95_pct": [ci_low, ci_high],
            "interpretation": f"The headline accuracy of {headline_acc}% carries a 95% confidence interval of [{ci_low}%, {ci_high}%], demonstrating ~{round(ci_high - ci_low, 1)}% margin of uncertainty."
        },
        "experiment_2_difficulty_stratification": {
            "easy_accuracy_pct": easy_acc,
            "hard_accuracy_pct": hard_acc,
            "difficulty_gap_pct": difficulty_gap,
            "interpretation": f"Performance drops by {difficulty_gap}% when moving from easy queries ({easy_acc}%) to hard queries ({hard_acc}%)."
        },
        "experiment_3_category_stratification": {
            "categories_evaluated": len(cat_rows),
            "lowest_performing_category": min(cat_rows, key=lambda x: x["accuracy_pct"]) if cat_rows else {},
            "highest_performing_category": max(cat_rows, key=lambda x: x["accuracy_pct"]) if cat_rows else {}
        },
        "experiment_6_baseline_vs_rag": {
            "baseline_accuracy_pct": round(baseline["accuracy"] * 100, 2),
            "rag_accuracy_pct": round(rag["accuracy"] * 100, 2),
            "accuracy_gain_pct": round((rag["accuracy"] - baseline["accuracy"]) * 100, 2),
            "interpretation": "RAG grounding improves support accuracy by providing verified context."
        },
        "experiment_7_high_severity_failure_concentration": {
            "missed_security_policy_escalations": missed_escalations,
            "unnecessary_escalations": false_escalations,
            "interpretation": f"While overall score is {headline_acc}%, the agent missed {missed_escalations} mandatory human escalations."
        }
    }
    
    with open("results/final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(experiments_report, f, indent=2)
        
    return experiments_report

if __name__ == "__main__":
    run_stress_testing_experiments()
