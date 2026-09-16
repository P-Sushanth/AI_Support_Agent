import os
import json
from typing import List, Dict, Any
from src.evaluation.metrics import wilson_score_interval, calculate_token_overlap_f1, calculate_escalation_metrics

class AgentEvaluator:
    def __init__(self, golden_set_path: str = "data/golden/golden_set.jsonl"):
        self.golden_set_path = golden_set_path
        self.golden_records = self._load_golden_set()

    def _load_golden_set(self) -> List[Dict[str, Any]]:
        records = []
        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def evaluate_predictions(self, agent_name: str, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        golden_map = {g["ticket_id"]: g for g in self.golden_records}
        
        intent_correct = 0
        esc_correct = 0
        total_eval = 0
        f1_scores = []
        
        esc_preds = []
        esc_truths = []
        
        by_category = {}
        by_difficulty = {}
        
        for res in batch_results:
            t_id = res["ticket_id"]
            golden = golden_map.get(t_id)
            if not golden:
                continue
                
            total_eval += 1
            output = res["output"]
            
            pred_intent = output.get("intent", "").lower()
            true_intent = golden.get("ground_truth_intent", "").lower()
            
            pred_resp = output.get("response", "")
            ref_resp = golden.get("ground_truth_response", "")
            
            token_scores = calculate_token_overlap_f1(pred_resp, ref_resp)
            f1_scores.append(token_scores["f1"])
            
            pred_esc = output.get("should_escalate", False)
            true_esc = golden.get("should_escalate", False)
            
            esc_preds.append(pred_esc)
            esc_truths.append(true_esc)
            
            # Intent correctness
            is_intent_match = (pred_intent == true_intent)
            if is_intent_match:
                intent_correct += 1
                
            # Escalation correctness
            is_esc_match = (pred_esc == true_esc)
            if is_esc_match:
                esc_correct += 1
                
            is_overall_correct = is_intent_match and is_esc_match and (token_scores["f1"] >= 0.15 or len(pred_resp.strip()) >= 20)
            
            # Difficulty breakdown
            diff = golden.get("difficulty", "medium")
            by_difficulty.setdefault(diff, {"correct": 0, "total": 0})
            by_difficulty[diff]["total"] += 1
            if is_overall_correct:
                by_difficulty[diff]["correct"] += 1

        acc_mean, acc_low, acc_high = wilson_score_interval(intent_correct, total_eval)
        avg_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
        escalation_metrics = calculate_escalation_metrics(esc_preds, esc_truths)
        
        diff_results = {}
        for d_name, d_data in by_difficulty.items():
            d_mean, d_low, d_high = wilson_score_interval(d_data["correct"], d_data["total"])
            diff_results[d_name] = {
                "correct": d_data["correct"],
                "total": d_data["total"],
                "accuracy": d_mean,
                "ci_95": [d_low, d_high]
            }

        return {
            "agent_name": agent_name,
            "total_evaluated": total_eval,
            "intent_accuracy": acc_mean,
            "intent_accuracy_ci_95": [acc_low, acc_high],
            "average_token_f1": avg_f1,
            "escalation_metrics": escalation_metrics,
            "difficulty_breakdown": diff_results
        }
