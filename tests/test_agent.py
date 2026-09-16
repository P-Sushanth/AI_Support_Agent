import os
import json
import pytest
from src.agent.schemas import CustomerTicketInput, SupportAgentOutput
from src.agent.client import LLMClient
from src.agent.agent import BaselineSupportAgent
from scripts.run_agent import run_batch_inference
from scripts.cli import load_sample_tickets

def test_customer_ticket_input_schema():
    inp = CustomerTicketInput(
        ticket_id="TICK-101",
        customer_message="Where is my order?"
    )
    assert inp.ticket_id == "TICK-101"
    assert inp.customer_message == "Where is my order?"
    assert inp.metadata == {}

def test_support_agent_output_schema():
    out = SupportAgentOutput(
        intent="order_status",
        response="Your order is in transit.",
        should_escalate=False,
        confidence=0.9,
        sources=["doc_1"]
    )
    assert out.intent == "order_status"
    assert out.should_escalate is False
    assert out.confidence == 0.9

def test_ollama_client_initialization():
    client = LLMClient(provider="ollama", model_name="qwen3.5:2b")
    assert client.provider == "ollama"
    assert client.model_name == "qwen3.5:2b"
    
    # Test generation (falls back to mock if server unaccessible, returns valid output)
    res = client.generate("System prompt", "Cancel my order", use_cache=False)
    assert "response" in res
    assert "intent" in res["response"]

def test_baseline_agent_escalation_security():
    agent = BaselineSupportAgent()
    inp = CustomerTicketInput(
        ticket_id="TICK-SEC-01",
        customer_message="My 2FA phone was hacked and I am locked out!"
    )
    result = agent.process_ticket(inp, use_cache=False)
    output = result["output"]
    assert output["should_escalate"] is True
    assert "security" in output["intent"].lower() or "Security" in output["escalation_reason"] or "hack" in output["intent"].lower()

def test_baseline_agent_caching(tmp_path):
    cache_dir = str(tmp_path / "cache")
    client = LLMClient(cache_dir=cache_dir)
    agent = BaselineSupportAgent(client=client)
    
    inp = CustomerTicketInput(ticket_id="TICK-C-1", customer_message="How to reset password?")
    
    res1 = agent.process_ticket(inp, use_cache=True)
    assert res1["cached"] is False
    
    res2 = agent.process_ticket(inp, use_cache=True)
    assert res2["cached"] is True

def test_batch_inference(tmp_path):
    dev_split = tmp_path / "dev.jsonl"
    out_file = tmp_path / "results.jsonl"
    
    records = [
        {"ticket_id": "T1", "category": "ORDER", "customer_message": "Cancel order #123", "should_escalate": False},
        {"ticket_id": "T2", "category": "BILLING", "customer_message": "Vat refund wire transfer $4,250", "should_escalate": True}
    ]
    with open(dev_split, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    res = run_batch_inference(str(dev_split), str(out_file))
    assert len(res) == 2
    assert os.path.exists(out_file)

def test_cli_load_sample_tickets():
    samples = load_sample_tickets()
    assert isinstance(samples, list)
    assert len(samples) > 0
