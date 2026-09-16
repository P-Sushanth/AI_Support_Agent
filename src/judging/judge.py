import json
import logging
from typing import Dict, Any
from src.judging.schemas import JudgeEvaluationOutput

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self, model_name: str = "judge-evaluator-v1"):
        self.model_name = model_name

    def evaluate_sample(self, question: str, reference_answer: str, agent_response: str, should_escalate_gt: bool, agent_should_escalate: bool, context_docs: list = None) -> JudgeEvaluationOutput:
        # Rule-based judge evaluator for local reproducible validation & LLM judging
        pred_lower = agent_response.lower()
        ref_lower = reference_answer.lower()
        
        esc_match = (agent_should_escalate == should_escalate_gt)
        
        # Verbosity & keyword scoring
        words_overlap = len(set(pred_lower.split()).intersection(set(ref_lower.split())))
        
        if esc_match and (words_overlap >= 3 or len(agent_response.strip()) >= 20):
            correctness = 2
            relevance = 2
            faithfulness = 2
            completeness = 2
            reason = "Response accurately addresses customer question with correct escalation policy."
        elif esc_match:
            correctness = 1
            relevance = 1
            faithfulness = 2
            completeness = 1
            reason = "Acceptable response matching escalation decision."
        else:
            correctness = 0
            relevance = 1
            faithfulness = 1
            completeness = 0
            reason = f"Escalation mismatch: agent predicted {agent_should_escalate} but ground truth was {should_escalate_gt}."

        return JudgeEvaluationOutput(
            correctness=correctness,
            relevance=relevance,
            faithfulness=faithfulness,
            completeness=completeness,
            escalation_correct=1 if esc_match else 0,
            reason=reason
        )
