import pytest
from src.evaluation.metrics import wilson_score_interval, calculate_token_overlap_f1, calculate_escalation_metrics

def test_wilson_score_interval_bounds():
    mean, low, high = wilson_score_interval(80, 100)
    assert mean == 0.80
    assert 0.70 < low < 0.80
    assert 0.80 < high < 0.90

def test_token_overlap_f1_exact_match():
    scores = calculate_token_overlap_f1("refund processed to original payment method", "refund processed to original payment method")
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0

def test_token_overlap_f1_disjoint():
    scores = calculate_token_overlap_f1("apple banana", "cat dog")
    assert scores["f1"] == 0.0

def test_escalation_metrics_perfect():
    preds = [True, False, True, False]
    truths = [True, False, True, False]
    metrics = calculate_escalation_metrics(preds, truths)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
