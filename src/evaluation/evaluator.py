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
        golden_map = {g["id"]: g for g in self.golden_records}
        golden_by_ticket = {g["original_ticket_id"]: g for g in self.golden_records}
        
        correct_count = 0
        total_eval = 0
        f1_scores = []
        
        esc_preds = []
        esc_truths = []
        
        by_category = {}
        by_difficulty = {}
        
        for res in batch_results:
            t_id = res["ticket_id"]
            golden = golden_map.get(t_id) or golden_by_ticket.get(t_id)
            if not golden:
                continue
                
            total_eval += 1
            output = res["output"]
            pred_resp = output.get("response", "")
            ref_resp = golden.get("reference_answer", "")
            
            token_scores = calculate_token_overlap_f1(pred_resp, ref_resp)
            f1_scores.append(token_scores["f1"])
            
            pred_esc = output.get("should_escalate", False)
            true_esc = golden.get("should_escalate", False)
            
            esc_preds.append(pred_esc)
            esc_truths.append(true_esc)
            
            # Correctness: escalation match AND minimum answer quality / similarity
            escalation_correct = (pred_esc == true_esc)
            answer_quality = (token_scores["f1"] >= 0.10) or (len(pred_resp.strip()) >= 20)
            
            is_correct = escalation_correct and answer_quality
            if is_correct:
                correct_count += 1
                
            # Category breakdown
            cat = golden.get("category", "GENERAL")
            by_category.setdefault(cat, {"correct": 0, "total": 0})
            by_category[cat]["total"] += 1
            if is_correct:
                by_category[cat]["correct"] += 1
                
            # Difficulty breakdown
            diff = golden.get("difficulty", "medium")
            by_difficulty.setdefault(diff, {"correct": 0, "total": 0})
            by_difficulty[diff]["total"] += 1
            if is_correct:
                by_difficulty[diff]["correct"] += 1
                
        acc_mean, acc_low, acc_high = wilson_score_interval(correct_count, total_eval)
        avg_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
        escalation_metrics = calculate_escalation_metrics(esc_preds, esc_truths)
        
        cat_results = {}
        for c_name, c_data in by_category.items():
            c_mean, c_low, c_high = wilson_score_interval(c_data["correct"], c_data["total"])
            cat_results[c_name] = {
                "correct": c_data["correct"],
                "total": c_data["total"],
                "accuracy": c_mean,
                "ci_95": [c_low, c_high]
            }
            
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
            "sample_size": total_eval,
            "correct_count": correct_count,
            "accuracy": acc_mean,
            "accuracy_ci_95": [acc_low, acc_high],
            "average_f1_token_overlap": avg_f1,
            "escalation_metrics": escalation_metrics,
            "category_stratification": cat_results,
            "difficulty_stratification": diff_results
        }
