from typing import Dict, Any
from src.agent.schemas import CustomerTicketInput, SupportAgentOutput

class TrivialBaselineAgent:
    """
    Trivial Baseline Agent:
    - Predicts majority intent ('general_troubleshooting').
    - Returns static canned template reply.
    - Never escalates (should_escalate = False).
    """
    def __init__(self):
        self.model_name = "trivial_majority_baseline"
        self.prompt_version = "v0_static"

    def process_ticket(self, ticket_input: CustomerTicketInput, use_cache: bool = True) -> Dict[str, Any]:
        output = SupportAgentOutput(
            intent="general_troubleshooting",
            response="Thank you for reaching out to @AppleSupport. Please send us a Direct Message with your device model and iOS version so we can investigate further.",
            should_escalate=False,
            escalation_reason=None,
            confidence=0.5,
            sources=[]
        )
        return {
            "ticket_id": ticket_input.ticket_id,
            "output": output.model_dump(),
            "latency_ms": 1.0,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "cached": False
        }
