import os
import json
from src.agent.trivial_agent import TrivialBaselineAgent
from src.agent.agent import BaselineSupportAgent
from src.agent.rag_agent import RAGSupportAgent
from src.agent.schemas import CustomerTicketInput
from src.evaluation.evaluator import AgentEvaluator

def run_full_evaluation():
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/baseline", exist_ok=True)
    os.makedirs("results/rag", exist_ok=True)
    golden_path = "data/golden/golden_set.jsonl"
    
    golden_records = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_records.append(json.loads(line))
                
    print(f"Running evaluation on {len(golden_records)} golden set examples...")
    
    # 1. Run Trivial Baseline Agent
    trivial_agent = TrivialBaselineAgent()
    trivial_results = []
    for g in golden_records:
        inp = CustomerTicketInput(
            ticket_id=g["ticket_id"],
            category=g.get("category", ""),
            customer_message=g["customer_message"]
        )
        res = trivial_agent.process_ticket(inp)
        trivial_results.append(res)
        
    # 2. Run Simple Baseline Agent (Zero-Shot)
    simple_agent = BaselineSupportAgent()
    simple_results = []
    for g in golden_records:
        inp = CustomerTicketInput(
            ticket_id=g["ticket_id"],
            category=g.get("category", ""),
            customer_message=g["customer_message"]
        )
        res = simple_agent.process_ticket(inp)
        simple_results.append(res)
        
    # 3. Run Proposed RAG Agent
    rag_agent = RAGSupportAgent()
    rag_results = []
    for g in golden_records:
        inp = CustomerTicketInput(
            ticket_id=g["ticket_id"],
            category=g.get("category", ""),
            customer_message=g["customer_message"]
        )
        res = rag_agent.process_ticket(inp)
        rag_results.append(res)
        
    # Save RAG results to disk for failure analysis
    with open("results/rag/golden_eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in rag_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open("results/baseline/golden_eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in simple_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Evaluate All Three
    evaluator = AgentEvaluator(golden_set_path=golden_path)
    trivial_eval = evaluator.evaluate_predictions("Trivial Baseline (Majority/Canned)", trivial_results)
    simple_eval = evaluator.evaluate_predictions("Simple Baseline (Zero-Shot)", simple_results)
    rag_eval = evaluator.evaluate_predictions("Proposed RAG Support Agent", rag_results)
    
    summary = {
        "golden_set_size": len(golden_records),
        "trivial_baseline": trivial_eval,
        "simple_baseline": simple_eval,
        "rag_agent": rag_eval
    }
    
    output_path = "results/metrics/evaluation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n================ FULL EVALUATION RESULTS ================")
    print(f"Golden Evaluation Set: {len(golden_records)} tickets\n")
    print(f"1. Trivial Baseline (Majority/Canned):")
    print(f"   - Intent Accuracy: {trivial_eval['intent_accuracy']*100:.1f}% (CI: {trivial_eval['intent_accuracy_ci_95']})")
    print(f"   - Escalation F1:   {trivial_eval['escalation_metrics']['f1']*100:.1f}%\n")
    print(f"2. Simple Baseline (Zero-Shot):")
    print(f"   - Intent Accuracy: {simple_eval['intent_accuracy']*100:.1f}% (CI: {simple_eval['intent_accuracy_ci_95']})")
    print(f"   - Escalation F1:   {simple_eval['escalation_metrics']['f1']*100:.1f}%\n")
    print(f"3. Proposed RAG Agent:")
    print(f"   - Intent Accuracy: {rag_eval['intent_accuracy']*100:.1f}% (CI: {rag_eval['intent_accuracy_ci_95']})")
    print(f"   - Escalation F1:   {rag_eval['escalation_metrics']['f1']*100:.1f}%")
    print(f"========================================================\n")
    
    return summary

if __name__ == "__main__":
    run_full_evaluation()
