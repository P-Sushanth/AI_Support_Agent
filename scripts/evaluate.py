import os
import json
from src.agent.agent import BaselineSupportAgent
from src.agent.rag_agent import RAGSupportAgent
from src.agent.schemas import CustomerTicketInput
from src.evaluation.evaluator import AgentEvaluator

def run_full_evaluation():
    os.makedirs("results/metrics", exist_ok=True)
    golden_path = "data/golden/golden_set.jsonl"
    
    # Load golden records
    golden_records = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_records.append(json.loads(line))
                
    print(f"Running evaluation on {len(golden_records)} golden set examples...")
    
    # 1. Run Baseline Agent
    baseline_agent = BaselineSupportAgent()
    baseline_results = []
    for g in golden_records:
        inp = CustomerTicketInput(
            ticket_id=g["id"],
            category=g.get("category", ""),
            customer_message=g["question"]
        )
        res = baseline_agent.process_ticket(inp)
        baseline_results.append(res)
        
    # Save baseline predictions
    with open("results/baseline/golden_eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in baseline_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 2. Run RAG Agent
    rag_agent = RAGSupportAgent()
    rag_results = []
    for g in golden_records:
        inp = CustomerTicketInput(
            ticket_id=g["id"],
            category=g.get("category", ""),
            customer_message=g["question"]
        )
        res = rag_agent.process_ticket(inp)
        rag_results.append(res)
        
    # Save RAG predictions
    with open("results/rag/golden_eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in rag_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 3. Evaluate Both
    evaluator = AgentEvaluator(golden_set_path=golden_path)
    baseline_eval = evaluator.evaluate_predictions("Baseline LLM", baseline_results)
    rag_eval = evaluator.evaluate_predictions("RAG Support Agent", rag_results)
    
    summary = {
        "golden_set_size": len(golden_records),
        "baseline": baseline_eval,
        "rag": rag_eval
    }
    
    output_path = "results/metrics/evaluation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Evaluation completed successfully! Results written to {output_path}")
    print(f"Baseline Accuracy: {baseline_eval['accuracy']} (CI: {baseline_eval['accuracy_ci_95']})")
    print(f"RAG Accuracy:      {rag_eval['accuracy']} (CI: {rag_eval['accuracy_ci_95']})")
    return summary

if __name__ == "__main__":
    run_full_evaluation()
