from typing import List, Optional
from pydantic import BaseModel, Field

class CustomerTicketInput(BaseModel):
    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_id: Optional[str] = Field(None, description="Customer account ID")
    category: Optional[str] = Field(None, description="Reported or inferred ticket category")
    customer_message: str = Field(..., description="Raw text of customer support query")
    metadata: Optional[dict] = Field(default_factory=dict, description="Optional ticket context metadata")

class SupportAgentOutput(BaseModel):
    intent: str = Field(..., description="Inferred support intent or topic")
    response: str = Field(..., description="Final natural language response provided to customer")
    should_escalate: bool = Field(..., description="Whether the issue must be escalated to human support")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation if should_escalate is True")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model self-reported confidence score (0.0 to 1.0)")
    sources: List[str] = Field(default_factory=list, description="IDs of knowledge base documents used as reference context")

class BatchInferenceResult(BaseModel):
    ticket_id: str
    input: CustomerTicketInput
    output: SupportAgentOutput
    latency_ms: float
    model_name: str
    prompt_version: str
    timestamp: str
