import os
import sys
import json
import random
from src.agent.agent import BaselineSupportAgent
from src.agent.rag_agent import RAGSupportAgent
from src.agent.schemas import CustomerTicketInput

# ANSI Colors for terminal UI
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header():
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}          AI CUSTOMER SUPPORT AGENT — INTERACTIVE TERMINAL CLI         {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"Commands: {YELLOW}/mode{RESET} (toggle Baseline/RAG) | {YELLOW}/sample{RESET} (test random ticket) | {YELLOW}/quit{RESET} (exit)")
    print("------------------------------------------------------------------------\n")

def load_sample_tickets():
    samples = []
    golden_path = "data/golden/golden_set.jsonl"
    if os.path.exists(golden_path):
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    return samples

def run_cli():
    print_header()
    
    baseline_agent = BaselineSupportAgent()
    rag_agent = RAGSupportAgent()
    current_mode = "RAG" # Default to RAG Agent
    sample_tickets = load_sample_tickets()
    
    ticket_counter = 100
    
    while True:
        mode_badge = f"{GREEN}[RAG AGENT]{RESET}" if current_mode == "RAG" else f"{YELLOW}[BASELINE LLM]{RESET}"
        try:
            user_input = input(f"{BOLD}Customer Query {mode_badge} > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Exiting AI Support Agent CLI. Goodbye!{RESET}")
            break
            
        if not user_input:
            continue
            
        cmd = user_input.lower()
        if cmd in ["/quit", "/exit", "quit", "exit"]:
            print(f"{YELLOW}Exiting AI Support Agent CLI. Goodbye!{RESET}")
            break
        elif cmd == "/mode":
            current_mode = "Baseline" if current_mode == "RAG" else "RAG"
            print(f"{BOLD}Switched mode to: {current_mode}{RESET}\n")
            continue
        elif cmd == "/help":
            print_header()
            continue
        elif cmd == "/sample":
            if not sample_tickets:
                print(f"{RED}No sample tickets found in golden set.{RESET}\n")
                continue
            sample = random.choice(sample_tickets)
            print(f"\n{BOLD}{YELLOW}--- Random Sample Ticket ({sample['category']}) ---{RESET}")
            print(f"ID: {sample['id']} | Expected Escalation: {sample['should_escalate']}")
            print(f"Customer Message: {sample['question']}\n")
            user_input = sample['question']

        ticket_counter += 1
        t_id = f"CLI-{ticket_counter}"
        inp = CustomerTicketInput(ticket_id=t_id, customer_message=user_input)
        
        # Process query
        if current_mode == "RAG":
            res = rag_agent.process_ticket(inp, use_cache=False)
        else:
            res = baseline_agent.process_ticket(inp, use_cache=False)
            
        output = res["output"]
        
        # Format response banner
        print(f"\n{BOLD}{CYAN}------------------- AGENT RESPONSE -------------------{RESET}")
        print(f"{BOLD}Intent:{RESET}       {output['intent']}")
        print(f"{BOLD}Confidence:{RESET}   {output['confidence']*100:.1f}%")
        
        esc = output["should_escalate"]
        if esc:
            esc_badge = f"{BOLD}{RED}[ESCALATE TO HUMAN]{RESET}"
            reason_str = f" ({output.get('escalation_reason', 'Policy trigger')})"
        else:
            esc_badge = f"{BOLD}{GREEN}[AUTO-RESOLVED]{RESET}"
            reason_str = ""
            
        print(f"{BOLD}Action:{RESET}       {esc_badge}{reason_str}")
        
        if current_mode == "RAG" and res.get("retrieved_documents"):
            print(f"\n{BOLD}Retrieved Knowledge Sources:{RESET}")
            for doc in res["retrieved_documents"]:
                print(f"  • [{doc['doc_id']}] (Score: {doc['score']:.3f}) {doc['text'][:80]}...")
                
        print(f"\n{BOLD}Response Message:{RESET}\n{output['response']}")
        print(f"{BOLD}{CYAN}------------------------------------------------------{RESET}\n")

if __name__ == "__main__":
    run_cli()
