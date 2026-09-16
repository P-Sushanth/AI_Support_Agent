import os
import json
from src.agent.client import LLMClient
from src.agent.agent import BaselineSupportAgent
from src.agent.rag_agent import RAGSupportAgent
from src.agent.schemas import CustomerTicketInput
from src.evaluation.evaluator import AgentEvaluator

MODELS = ["qwen3.5:2b", "qwen3.5:9b", "gemma4:12b"]

def run_model_comparison(limit: int = 50):
    os.makedirs("results/metrics", exist_ok=True)
    golden_path = "data/golden/golden_set.jsonl"
    
    golden_records = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_records.append(json.loads(line))
                
    golden_subset = golden_records[:limit]
    evaluator = AgentEvaluator(golden_set_path=golden_path)
    
    comparison_results = []
    
    for m in MODELS:
        print(f"Running evaluation benchmark for model: {m}...")
        client = LLMClient(provider="ollama", model_name=m)
        agent = RAGSupportAgent(client=client)
        
        batch_res = []
        for g in golden_subset:
            inp = CustomerTicketInput(ticket_id=g["id"], category=g.get("category",""), customer_message=g["question"])
            res = agent.process_ticket(inp, use_cache=True)
            batch_res.append(res)
            
        eval_metrics = evaluator.evaluate_predictions(m, batch_res)
        
        latencies = [r["latency_ms"] for r in batch_res if "latency_ms" in r]
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        
        comparison_results.append({
            "model_name": m,
            "accuracy_pct": round(eval_metrics["accuracy"] * 100, 1),
            "escalation_f1_pct": round(eval_metrics["escalation_metrics"]["f1"] * 100, 1),
            "escalation_precision_pct": round(eval_metrics["escalation_metrics"]["precision"] * 100, 1),
            "escalation_recall_pct": round(eval_metrics["escalation_metrics"]["recall"] * 100, 1),
            "avg_latency_ms": avg_latency
        })
        
    output_path = "results/metrics/model_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
        
    print(f"Model comparison completed! Summary saved to {output_path}")
    return comparison_results

if __name__ == "__main__":
    run_model_comparison()
