import os
import re
import json
from datasets import load_dataset

RAW_DATA_PATH = "data/raw/customer_support_raw.jsonl"

def classify_intent_and_escalation(customer_msg: str):
    msg_lower = customer_msg.lower()
    
    # 1. Escalation Triggers
    should_escalate = False
    escalation_reason = None
    
    if any(w in msg_lower for w in ["hack", "stolen", "breach", "compromised", "bypassed"]):
        should_escalate = True
        escalation_reason = "Compromised Apple ID or security threat reported."
    elif any(w in msg_lower for w in ["swollen", "smoke", "fire", "hazard", "explosion"]):
        should_escalate = True
        escalation_reason = "Hazardous hardware / thermal battery safety alert."
    elif any(w in msg_lower for w in ["lawyer", "legal", "sue", "manager", "supervisor", "executive"]):
        should_escalate = True
        escalation_reason = "Explicit demand for executive management or legal notice."
    elif any(w in msg_lower for w in ["unauthorized", "stolen card", "fraud", "$1,200", "$850"]):
        should_escalate = True
        escalation_reason = "High-value disputed billing transaction."

    # 2. Intent Classification
    if any(w in msg_lower for w in ["update", "ios", "drain", "slow", "lag", "freeze", "crash"]):
        intent = "ios_update_performance"
        category = "iOS Performance & Updates"
    elif any(w in msg_lower for w in ["password", "icloud", "apple id", "lock", "2fa", "sign-in", "security"]):
        intent = "account_icloud_security"
        category = "Apple ID & Security"
    elif any(w in msg_lower for w in ["repair", "screen", "cracked", "battery", "applecare", "hardware", "swollen"]):
        intent = "hardware_battery_repair"
        category = "Hardware & Battery Repair"
    elif any(w in msg_lower for w in ["charge", "refund", "subscription", "purchase", "billing", "app store", "money"]):
        intent = "app_store_billing"
        category = "App Store & Subscriptions"
    elif any(w in msg_lower for w in ["airpods", "bluetooth", "watch", "wi-fi", "wifi", "connect", "pair"]):
        intent = "connectivity_accessory"
        category = "Bluetooth & Accessories"
    else:
        intent = "general_troubleshooting"
        category = "General iOS & Mac Help"

    difficulty = "hard" if should_escalate else ("medium" if len(customer_msg) > 120 else "easy")
    return intent, category, should_escalate, escalation_reason, difficulty

def fetch_and_process_twitter_dataset():
    os.makedirs("data/raw", exist_ok=True)
    
    print("Loading Customer Support on Twitter dataset (TNE-AI/customer-support-on-twitter-conversation)...")
    ds = load_dataset('TNE-AI/customer-support-on-twitter-conversation', split='train')
    
    apple_rows = [r for r in ds if r['company'] == 'AppleSupport']
    print(f"Extracted {len(apple_rows)} raw @AppleSupport Twitter conversation threads.")
    
    records = []
    ticket_counter = 1000
    
    for row in apple_rows:
        text = row['conversation']
        c_match = re.search(r'Customer:\s*(.*?)(?=\nSupport:|\nCustomer:|$)', text, re.DOTALL)
        s_match = re.search(r'Support:\s*(.*?)(?=\nSupport:|\nCustomer:|$)', text, re.DOTALL)
        
        if c_match and s_match:
            c_msg = c_match.group(1).strip()
            s_resp = s_match.group(1).strip()
            
            # Clean URLs and twitter tags slightly for readability while preserving real customer wording
            if len(c_msg) >= 15 and len(s_resp) >= 15:
                ticket_counter += 1
                ticket_id = f"APPL-{ticket_counter}"
                
                intent, category, should_esc, esc_reason, diff = classify_intent_and_escalation(c_msg)
                
                rec = {
                    "ticket_id": ticket_id,
                    "customer_id": f"tw_user_{ticket_counter}",
                    "brand": "@AppleSupport",
                    "category": category,
                    "intent": intent,
                    "difficulty": diff,
                    "customer_message": c_msg,
                    "agent_response": s_resp,
                    "should_escalate": should_esc,
                    "escalation_reason": esc_reason,
                    "required_facts": [s_resp[:100]]
                }
                records.append(rec)
                
                if len(records) >= 1000:
                    break

    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(records)} authentic @AppleSupport Twitter conversation records in {RAW_DATA_PATH}")
    return records

if __name__ == "__main__":
    fetch_and_process_twitter_dataset()
