import os
import json
import argparse
from src.agent.rag_agent import RAGSupportAgent
from src.agent.schemas import CustomerTicketInput

def run_rag_batch_inference(input_file: str, output_file: str, limit: int = None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    agent = RAGSupportAgent()
    
    results = []
    processed_count = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            
            inp = CustomerTicketInput(
                ticket_id=record["ticket_id"],
                category=record.get("category", ""),
                customer_message=record.get("customer_message", "")
            )
            
            out = agent.process_ticket(inp)
            out["ground_truth"] = {
                "category": record.get("category"),
                "intent": record.get("intent"),
                "should_escalate": record.get("should_escalate"),
                "reference_response": record.get("agent_response"),
                "required_facts": record.get("required_facts", []),
                "difficulty": record.get("difficulty", "medium")
            }
            results.append(out)
            processed_count += 1
            
            if limit and processed_count >= limit:
                break
                
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"Completed RAG batch inference for {processed_count} tickets. Results saved to {output_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Agent batch inference")
    parser.add_argument("--input", type=str, default="data/processed/dev.jsonl", help="Input JSONL split")
    parser.add_argument("--output", type=str, default="results/rag/batch_results.jsonl", help="Output JSONL results file")
    parser.add_argument("--limit", type=int, default=500, help="Max tickets to process")
    args = parser.parse_args()
    
    run_rag_batch_inference(args.input, args.output, limit=args.limit)
