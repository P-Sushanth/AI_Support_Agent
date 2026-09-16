import logging
from typing import Dict, Any, Optional
from src.agent.schemas import CustomerTicketInput, SupportAgentOutput
from src.agent.client import LLMClient
from src.agent.prompts import PromptManager

logger = logging.getLogger(__name__)

class BaselineSupportAgent:
    def __init__(self, client: Optional[LLMClient] = None, prompt_version: str = "system_v1"):
        self.client = client or LLMClient(provider="mock", model_name="simple-baseline-v1")
        self.prompt_manager = PromptManager()
        self.prompt_version = prompt_version
        self.system_prompt = self.prompt_manager.load_prompt(prompt_version)

    def process_ticket(self, ticket_input: CustomerTicketInput, use_cache: bool = True) -> Dict[str, Any]:
        user_msg = self.prompt_manager.format_user_message(
            customer_message=ticket_input.customer_message,
            category=ticket_input.category or ""
        )
        
        try:
            llm_result = self.client.generate(self.system_prompt, user_msg, use_cache=use_cache)
            raw_response = llm_result["response"]
            
            structured_output = SupportAgentOutput(
                intent=raw_response.get("intent", "general_troubleshooting"),
                response=raw_response.get("response", "Thank you for reaching out to @AppleSupport."),
                should_escalate=raw_response.get("should_escalate", False),
                escalation_reason=raw_response.get("escalation_reason"),
                confidence=raw_response.get("confidence", 1.0),
                sources=raw_response.get("sources", [])
            )
            
            return {
                "ticket_id": ticket_input.ticket_id,
                "output": structured_output.model_dump(),
                "latency_ms": llm_result["latency_ms"],
                "model_name": llm_result["model_name"],
                "prompt_version": self.prompt_version,
                "cached": llm_result.get("cached", False)
            }
        except Exception as e:
            logger.error(f"Error processing ticket {ticket_input.ticket_id}: {e}")
            fallback_output = SupportAgentOutput(
                intent="general_troubleshooting",
                response="We encountered an issue processing your request. Escalating to support.",
                should_escalate=True,
                escalation_reason=f"Processing exception: {str(e)}",
                confidence=0.0,
                sources=[]
            )
            return {
                "ticket_id": ticket_input.ticket_id,
                "output": fallback_output.model_dump(),
                "latency_ms": 0.0,
                "model_name": self.client.model_name,
                "prompt_version": self.prompt_version,
                "error": str(e)
            }
