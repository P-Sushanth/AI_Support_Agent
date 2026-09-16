import pytest
from src.analysis.failures import categorize_failure, TAXONOMY_DESCRIPTIONS

def test_categorize_failure_missed_escalation():
    agent_res = {
        "output": {"response": "We will help with your issue.", "should_escalate": False}
    }
    golden_rec = {
        "id": "GOLD-1",
        "original_ticket_id": "T-1",
        "category": "ACCOUNT",
        "difficulty": "hard",
        "should_escalate": True,
        "reference_answer": "Escalated to security team."
    }
    fail = categorize_failure(agent_res, golden_rec)
    assert fail["failure_type"] == "incorrect_escalation"
    assert fail["severity"] == "high"

def test_categorize_failure_unnecessary_escalation():
    agent_res = {
        "output": {"response": "Escalating simple question.", "should_escalate": True}
    }
    golden_rec = {
        "id": "GOLD-2",
        "original_ticket_id": "T-2",
        "category": "ORDER",
        "difficulty": "easy",
        "should_escalate": False,
        "reference_answer": "You can cancel under My Orders."
    }
    fail = categorize_failure(agent_res, golden_rec)
    assert fail["failure_type"] == "unnecessary_escalation"
    assert fail["severity"] == "medium"
