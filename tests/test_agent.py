import os
import json
import pytest
from src.agent.schemas import CustomerTicketInput, SupportAgentOutput
from src.agent.client import LLMClient
from src.agent.agent import BaselineSupportAgent
from scripts.run_agent import run_batch_inference

def test_customer_ticket_input_schema():
    inp = CustomerTicketInput(
        ticket_id="APPL-101",
        customer_message="How do I record screen on iPad?"
    )
    assert inp.ticket_id == "APPL-101"
    assert inp.customer_message == "How do I record screen on iPad?"
    assert inp.metadata == {}

def test_support_agent_output_schema():
    out = SupportAgentOutput(
        intent="general_troubleshooting",
        response="Settings > Control Center",
        should_escalate=False,
        confidence=0.9,
        sources=["doc_1"]
    )
    assert out.intent == "general_troubleshooting"
    assert out.should_escalate is False
    assert out.confidence == 0.9

def test_mock_client_generation():
    client = LLMClient(provider="mock", model_name="apple-agent-v1")
    assert client.provider == "mock"
    assert client.model_name == "apple-agent-v1"
    
    res = client.generate("System prompt", "How do I reset Apple ID password?", use_cache=False)
    assert "response" in res
    assert "intent" in res["response"]

def test_baseline_agent_escalation_security():
    mock_client = LLMClient(provider="mock")
    agent = BaselineSupportAgent(client=mock_client)
    inp = CustomerTicketInput(
        ticket_id="APPL-SEC-01",
        customer_message="My Apple ID password was changed by a hacker and 2FA was bypassed!"
    )
    result = agent.process_ticket(inp, use_cache=False)
    output = result["output"]
    assert output["should_escalate"] is True
    assert "account_icloud_security" in output["intent"].lower() or "Compromised" in output["escalation_reason"]

def test_baseline_agent_caching(tmp_path):
    cache_dir = str(tmp_path / "cache")
    client = LLMClient(provider="mock", cache_dir=cache_dir)
    agent = BaselineSupportAgent(client=client)
    
    inp = CustomerTicketInput(ticket_id="APPL-C-1", customer_message="How to reset password?")
    
    res1 = agent.process_ticket(inp, use_cache=True)
    assert res1["cached"] is False
    
    res2 = agent.process_ticket(inp, use_cache=True)
    assert res2["cached"] is True

def test_batch_inference(tmp_path):
    dev_split = tmp_path / "dev.jsonl"
    out_file = tmp_path / "results.jsonl"
    
    records = [
        {"ticket_id": "APPL-T1", "category": "iOS", "customer_message": "Battery drain after iOS update", "should_escalate": False},
        {"ticket_id": "APPL-T2", "category": "Security", "customer_message": "Swollen battery heating up!", "should_escalate": True}
    ]
    with open(dev_split, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    res = run_batch_inference(str(dev_split), str(out_file))
    assert len(res) == 2
    assert os.path.exists(out_file)
