import pytest
from src.judging.schemas import JudgeEvaluationOutput
from src.judging.judge import LLMJudge
from src.judging.agreement import compute_cohens_kappa

def test_judge_schema_validation():
    eval_out = JudgeEvaluationOutput(
        correctness=2,
        relevance=2,
        faithfulness=2,
        completeness=2,
        escalation_correct=1,
        reason="Fully correct answer"
    )
    assert eval_out.correctness == 2
    assert eval_out.escalation_correct == 1

def test_llm_judge_evaluation():
    judge = LLMJudge()
    res = judge.evaluate_sample(
        question="Cancel my order",
        reference_answer="Go to My Orders to cancel.",
        agent_response="You can cancel your order under My Orders.",
        should_escalate_gt=False,
        agent_should_escalate=False
    )
    assert res.correctness >= 1
    assert res.escalation_correct == 1

def test_cohens_kappa_calculation():
    rater1 = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    rater2 = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 9/10 agreement
    
    metrics = compute_cohens_kappa(rater1, rater2)
    assert metrics["agreement_rate"] == 0.9
    assert metrics["cohens_kappa"] > 0.70
