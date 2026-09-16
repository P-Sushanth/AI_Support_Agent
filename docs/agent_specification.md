# Phase 2 — Agent Specification

## Objective
Define explicit input/output contracts, core agent responsibilities, non-goals, versioned prompts, and escalation rules for the AI Support Agent before implementation.

---

## 1. Input & Output Schemas

### Canonical Input Structure (`CustomerTicketInput`)
```json
{
  "ticket_id": "TICK-1001",
  "customer_id": "CUST-8832",
  "category": "billing_inquiry",
  "customer_message": "Hi, I was charged twice for my subscription this month ($29.99 x 2). Can you check invoice #INV-9041?",
  "metadata": {}
}
```

### Canonical Output Structure (`SupportAgentOutput`)
```json
{
  "intent": "duplicate_billing_refund",
  "response": "I apologize for the double charge. I have verified invoice #INV-9041 and processed a refund of $29.99 back to your original payment method within 3-5 business days.",
  "should_escalate": false,
  "escalation_reason": null,
  "confidence": 0.95,
  "sources": []
}
```

---

## 2. Core Agent Responsibilities
1. **Intent Understanding**: Parse and categorize customer queries accurately.
2. **Factual Grounding**: Answer inquiries using official support rules and context.
3. **Escalation Detection**: Identify tickets that require human agent intervention based on policy rules or security risks.
4. **Structured Communication**: Output well-formed JSON conforming strictly to `SupportAgentOutput`.

---

## 3. Explicit Escalation Policy

The agent **MUST** trigger escalation (`should_escalate: true`) under the following triggers:
- **Security & Account Hijacking**: Active hacker reports, unauthorized 2FA phone number modifications, or account takeover alerts.
- **Complex Corporate Finance**: Requests for retroactive corporate tax exemption wire credits or complex multi-invoice tax recalculations.
- **Out-of-Policy Exceptions**: Return or refund requests beyond the standard 30-day window demanding exception overrides.
- **Explicit Human Demands**: Customer explicitly requests a human representative, manager, or supervisor.
- **Context Ambiguity / Insufficient Policy**: High uncertainty where company policy is unknown or missing.

---

## 4. System Non-Goals (Constraints)
- **No Policy Invention**: The agent must NEVER fabricate returns, warranties, or operational policies not stated in official context.
- **No Unauthorized Action Claims**: The agent must NEVER claim to have modified a customer's live credit card or database unless executed via a tool call.
- **No Out-of-Domain Answers**: The agent must politely decline medical, legal, or non-platform queries.

---

## 5. Success Criteria & Metrics Mapping
- **Correctness**: Answer accurately resolves the customer's query.
- **Faithfulness**: Claims in response are supported by retrieved knowledge.
- **Escalation Precision & Recall**: Agent correctly escalates mandatory tickets without over-escalating simple self-service tickets.
- **Schema Compliance**: 100% of batch inference outputs parse into `SupportAgentOutput`.
