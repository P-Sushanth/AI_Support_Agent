from typing import List, Dict, Any

def compute_cohens_kappa(rater1: List[int], rater2: List[int]) -> Dict[str, float]:
    """
    Calculate Cohen's kappa coefficient for inter-rater agreement.
    kappa = (Po - Pe) / (1 - Pe)
    """
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return {"agreement_rate": 0.0, "cohens_kappa": 0.0}
        
    n = len(rater1)
    agreements = sum(1 for a, b in zip(rater1, rater2) if a == b)
    p_o = agreements / n
    
    # Expected agreement by chance
    cats = set(rater1).union(set(rater2))
    p_e = 0.0
    for c in cats:
        cnt1 = sum(1 for x in rater1 if x == c)
        cnt2 = sum(1 for x in rater2 if x == c)
        p_e += (cnt1 / n) * (cnt2 / n)
        
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0
    
    return {
        "sample_size": n,
        "agreement_rate": round(p_o, 4),
        "cohens_kappa": round(kappa, 4)
    }
