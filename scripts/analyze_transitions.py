import json
from src.evaluation.evaluator import AgentEvaluator

def run_transition_analysis():
    golden_records = []
    with open("data/golden/golden_set.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_records.append(json.loads(line))
                
    golden_map = {g["ticket_id"]: g for g in golden_records}
    
    simple_results = []
    with open("results/baseline/golden_eval_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                simple_results.append(json.loads(line))
                
    rag_results = []
    with open("results/rag/golden_eval_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rag_results.append(json.loads(line))
                
    simple_map = {r["ticket_id"]: r["output"] for r in simple_results}
    rag_map = {r["ticket_id"]: r["output"] for r in rag_results}
    
    c_to_c = []
    c_to_w = []
    w_to_c = []
    w_to_w = []
    
    for t_id, golden in golden_map.items():
        gt_intent = golden["ground_truth_intent"].lower()
        
        s_output = simple_map.get(t_id, {})
        r_output = rag_map.get(t_id, {})
        
        s_correct = (s_output.get("intent", "").lower() == gt_intent)
        r_correct = (r_output.get("intent", "").lower() == gt_intent)
        
        entry = {
            "ticket_id": t_id,
            "message": golden["customer_message"],
            "gt_intent": gt_intent,
            "simple_intent": s_output.get("intent", ""),
            "rag_intent": r_output.get("intent", "")
        }
        
        if s_correct and r_correct:
            c_to_c.append(entry)
        elif s_correct and not r_correct:
            c_to_w.append(entry)
        elif not s_correct and r_correct:
            w_to_c.append(entry)
        else:
            w_to_w.append(entry)
            
    total = len(golden_map)
    print("================ SIMPLE -> RAG INTENT TRANSITION MATRIX ================")
    print(f"Total Golden Set Tickets: {total}\n")
    print(f"1. Correct -> Correct: {len(c_to_c):3d} ({len(c_to_c)/total*100:.1f}%) -- RAG preserved correct classification")
    print(f"2. Correct -> Wrong:   {len(c_to_w):3d} ({len(c_to_w)/total*100:.1f}%) -- RAG context introduced distractor noise")
    print(f"3. Wrong   -> Correct: {len(w_to_c):3d} ({len(w_to_c)/total*100:.1f}%) -- RAG context fixed simple baseline error")
    print(f"4. Wrong   -> Wrong:   {len(w_to_w):3d} ({len(w_to_w)/total*100:.1f}%) -- Both models failed on complex query\n")
    print("========================================================================\n")
    
    print("SAMPLE Correct -> Wrong (RAG Distractor Noise):")
    for x in c_to_w[:3]:
        print(f"ID: {x['ticket_id']} | GT: {x['gt_intent']} | Simple: {x['simple_intent']} | RAG: {x['rag_intent']}")
        print(f"   Msg: {x['message']}\n")

    return {
        "correct_to_correct": len(c_to_c),
        "correct_to_wrong": len(c_to_w),
        "wrong_to_correct": len(w_to_c),
        "wrong_to_wrong": len(w_to_w),
        "c_to_w_examples": c_to_w[:5]
    }

if __name__ == "__main__":
    run_transition_analysis()
