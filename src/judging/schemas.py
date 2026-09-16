from pydantic import BaseModel, Field

class JudgeEvaluationOutput(BaseModel):
    correctness: int = Field(..., ge=0, le=2, description="0=Incorrect, 1=Acceptable/Partial, 2=Strong/Fully Correct")
    relevance: int = Field(..., ge=0, le=2, description="0=Irrelevant, 1=Relevant, 2=Direct & Focused")
    faithfulness: int = Field(..., ge=0, le=2, description="0=Hallucinated/Unsupported, 1=Mostly Grounded, 2=Fully Grounded")
    completeness: int = Field(..., ge=0, le=2, description="0=Incomplete, 1=Adequate, 2=Comprehensive")
    escalation_correct: int = Field(..., ge=0, le=1, description="0=Wrong escalation decision, 1=Correct escalation decision")
    reason: str = Field(..., description="Detailed explanation of judge score rationale")
