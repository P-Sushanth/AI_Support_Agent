import math
from typing import Dict, Any, List, Tuple

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate Wilson Score 95% Confidence Interval for binomial proportions.
    Returns: (mean, ci_lower, ci_upper)
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = 1.95996  # 95% confidence z-score
    
    denominator = 1 + (z**2 / total)
    center = (p + (z**2 / (2 * total))) / denominator
    spread = (z * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))) / denominator
    
    ci_lower = max(0.0, round(center - spread, 4))
    ci_upper = min(1.0, round(center + spread, 4))
    mean = round(p, 4)
    return mean, ci_lower, ci_upper

def calculate_token_overlap_f1(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    overlap = pred_tokens.intersection(ref_tokens)
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def calculate_escalation_metrics(predictions: List[bool], ground_truths: List[bool]) -> Dict[str, Any]:
    tp = sum(1 for p, g in zip(predictions, ground_truths) if p is True and g is True)
    fp = sum(1 for p, g in zip(predictions, ground_truths) if p is True and g is False)
    fn = sum(1 for p, g in zip(predictions, ground_truths) if p is False and g is True)
    tn = sum(1 for p, g in zip(predictions, ground_truths) if p is False and g is False)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0
    
    acc_mean, acc_low, acc_high = wilson_score_interval(tp + tn, len(predictions))
    
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": acc_mean,
        "accuracy_ci_95": [acc_low, acc_high]
    }
