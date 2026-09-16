import logging
from typing import Dict, Any, Optional
from src.agent.schemas import CustomerTicketInput, SupportAgentOutput
from src.agent.client import LLMClient
from src.agent.prompts import PromptManager
from src.retrieval.retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)

class RAGSupportAgent:
    def __init__(self, client: Optional[LLMClient] = None, retriever: Optional[KnowledgeRetriever] = None, top_k: int = 3):
        self.client = client or LLMClient(model_name="rag-support-agent-v1")
        self.prompt_manager = PromptManager()
        self.system_prompt = self.prompt_manager.load_prompt("system_v1")
        self.top_k = top_k
        
        if retriever is None:
            self.retriever = KnowledgeRetriever(top_k=top_k)
            self.retriever.load_index("results/rag/index")
        else:
            self.retriever = retriever

    def process_ticket(self, ticket_input: CustomerTicketInput, use_cache: bool = True) -> Dict[str, Any]:
        # 1. Perform Top-K Retrieval from indexed knowledge base
        retrieved_docs = self.retriever.retrieve(
            query=ticket_input.customer_message,
            top_k=self.top_k
        )
        
        context_texts = [d["text"] for d in retrieved_docs]
        retrieved_source_ids = [d["doc_id"] for d in retrieved_docs]
        
        # 2. Format RAG prompt incorporating knowledge context
        formatted_user_msg = self.prompt_manager.format_user_message(
            customer_message=ticket_input.customer_message,
            category=ticket_input.category or "",
            context_docs=context_texts
        )
        
        try:
            llm_result = self.client.generate(self.system_prompt, formatted_user_msg, use_cache=use_cache)
            raw = llm_result["response"]
            
            structured_output = SupportAgentOutput(
                intent=raw.get("intent", "general_support"),
                response=raw.get("response", "Thank you for reaching out."),
                should_escalate=raw.get("should_escalate", False),
                escalation_reason=raw.get("escalation_reason"),
                confidence=raw.get("confidence", 0.95),
                sources=retrieved_source_ids
            )
            
            return {
                "ticket_id": ticket_input.ticket_id,
                "output": structured_output.model_dump(),
                "retrieved_documents": retrieved_docs,
                "latency_ms": llm_result["latency_ms"],
                "model_name": llm_result["model_name"],
                "cached": llm_result.get("cached", False)
            }
        except Exception as e:
            logger.error(f"Error processing RAG ticket {ticket_input.ticket_id}: {e}")
            fallback_output = SupportAgentOutput(
                intent="error_fallback",
                response="Error processing inquiry. Escalating to supervisor.",
                should_escalate=True,
                escalation_reason=str(e),
                confidence=0.0,
                sources=retrieved_source_ids
            )
            return {
                "ticket_id": ticket_input.ticket_id,
                "output": fallback_output.model_dump(),
                "retrieved_documents": retrieved_docs,
                "latency_ms": 0.0,
                "model_name": self.client.model_name,
                "error": str(e)
            }
