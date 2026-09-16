import os
import json
from src.data.audit import run_dataset_audit
from src.retrieval.documents import build_knowledge_corpus_from_train
from src.retrieval.retriever import KnowledgeRetriever
from scripts.generate_golden_set import generate_golden_set_from_real_data
from src.agent.trivial_agent import TrivialBaselineAgent
from src.agent.agent import BaselineSupportAgent
from src.agent.rag_agent import RAGSupportAgent
from src.agent.schemas import CustomerTicketInput
from src.evaluation.evaluator import AgentEvaluator
from scripts.validate_judge import run_judge_validation
from src.analysis.failures import run_failure_analysis
from src.analysis.stress_test import run_stress_testing_experiments

def run_full_pipeline():
    print("========================================================")
    print("STEP 1: Auditing and Partitioning Dataset Splits...")
    print("========================================================")
    run_dataset_audit()

    print("\n========================================================")
    print("STEP 2: Building RAG Knowledge Base Index...")
    print("========================================================")
    chunks = build_knowledge_corpus_from_train("data/processed/train.jsonl")
    retriever = KnowledgeRetriever()
    retriever.build_index(chunks)
    retriever.save_index("results/rag/index")

    print("\n========================================================")
    print("STEP 3: Generating Golden Set & Human Spot-Checks...")
    print("========================================================")
    generate_golden_set_from_real_data()

    print("\n========================================================")
    print("STEP 4: Evaluating Baselines & RAG Support Agent...")
    print("========================================================")
    golden_path = "data/golden/golden_set.jsonl"
    golden_records = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_records.append(json.loads(line))

    # 1. Trivial Baseline
    trivial_agent = TrivialBaselineAgent()
    trivial_results = [trivial_agent.process_ticket(CustomerTicketInput(ticket_id=g["ticket_id"], customer_message=g["customer_message"])) for g in golden_records]

    # 2. Simple Baseline (Zero-Shot)
    simple_agent = BaselineSupportAgent()
    simple_results = [simple_agent.process_ticket(CustomerTicketInput(ticket_id=g["ticket_id"], customer_message=g["customer_message"])) for g in golden_records]

    # 3. RAG Support Agent
    rag_agent = RAGSupportAgent()
    rag_results = [rag_agent.process_ticket(CustomerTicketInput(ticket_id=g["ticket_id"], customer_message=g["customer_message"])) for g in golden_records]

    # Save prediction logs
    os.makedirs("results/rag", exist_ok=True)
    os.makedirs("results/baseline", exist_ok=True)
    with open("results/rag/golden_eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in rag_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open("results/baseline/golden_eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in simple_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n================ FULL EVALUATION BENCHMARK ================")
    print(f"Golden Set: {len(golden_records)} real @AppleSupport tickets")
    print(f"1. Trivial Baseline (Majority/Canned):  Acc = {trivial_eval['intent_accuracy']*100:.1f}% | Esc F1 = {trivial_eval['escalation_metrics']['f1']*100:.1f}%")
    print(f"2. Simple Baseline (Zero-Shot):         Acc = {simple_eval['intent_accuracy']*100:.1f}% | Esc F1 = {simple_eval['escalation_metrics']['f1']*100:.1f}%")
    print(f"3. Proposed RAG Agent:                 Acc = {rag_eval['intent_accuracy']*100:.1f}% | Esc F1 = {rag_eval['escalation_metrics']['f1']*100:.1f}%")
    print(f"===========================================================\n")

    print("========================================================")
    print("STEP 5: Validating LLM-as-Judge vs Human Spot Checks...")
    print("========================================================")
    run_judge_validation()

    print("\n========================================================")
    print("STEP 6: Categorizing Failure Taxonomy...")
    print("========================================================")
    run_failure_analysis()

    print("\n========================================================")
    print("STEP 7: Stress-Testing Headline Metric Bias...")
    print("========================================================")
    run_stress_testing_experiments()

    print("\nPIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    return summary

if __name__ == "__main__":
    run_full_pipeline()
