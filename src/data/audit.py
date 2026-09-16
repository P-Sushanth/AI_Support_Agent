import os
import json
import re
import yaml
import pandas as pd

BITEXT_PATH = "data/raw/bitext_customer_support.csv"
MULTILANG_PATH = "data/raw/multilang_tickets.csv"
TWITTER_PATH = "data/raw/twitter_support_sample.csv"

PROCESSED_TRAIN = "data/processed/train.jsonl"
PROCESSED_DEV = "data/processed/dev.jsonl"
PROCESSED_GOLDEN_CANDIDATES = "data/processed/golden_candidate.jsonl"
CONFIG_PATH = "configs/data.yaml"
PROFILE_PATH = "results/dataset_profile.json"
REPORT_PATH = "docs/dataset_report.md"

TARGET_BRAND = "@AppleSupport"

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
}

ESCALATION_INTENTS = {
    "security", "security_incident", "account_appleid_security", "data_breach", "account_hack",
    "refund_override", "device_hardware_repair", "legal_threat"
}

def load_and_standardize_raw_datasets():
    canonical_records = []
    
    # 1. Bitext Dataset (Mapped to @AppleSupport Domain Intents)
    if os.path.exists(BITEXT_PATH):
        df_bitext = pd.read_csv(BITEXT_PATH).fillna("")
        for idx, row in df_bitext.iterrows():
            intent = str(row.get("intent", "")).strip().lower()
            category = str(row.get("category", "")).strip().upper()
            msg = str(row.get("instruction", "")).strip()
            resp = str(row.get("response", "")).strip()
            
            if not msg:
                continue
                
            # Map raw intent to @AppleSupport brand intent taxonomy
            if "cancel" in intent or "order" in category:
                brand_intent = "order_shipping_delivery"
            elif "refund" in intent or "invoice" in intent:
                brand_intent = "billing_subscription_refund"
            elif "password" in intent or "account" in category:
                brand_intent = "account_appleid_security"
            else:
                brand_intent = "software_ios_update"
                
            should_escalate = any(k in intent for k in ESCALATION_INTENTS)
            difficulty = "hard" if should_escalate else ("medium" if len(msg) > 100 else "easy")
            
            canonical_records.append({
                "ticket_id": f"APPLE-BIT-{idx+1:05d}",
                "brand": TARGET_BRAND,
                "source_dataset": "bitext_customer_support",
                "category": category or "APPLE_SUPPORT",
                "intent": brand_intent,
                "language": "en",
                "priority": "high" if should_escalate else "normal",
                "customer_message": msg,
                "agent_response": resp,
                "should_escalate": should_escalate,
                "difficulty": difficulty,
                "required_facts": [resp[:100]] if resp else []
            })
            
    # 2. Multilingual Support Tickets Dataset
    if os.path.exists(MULTILANG_PATH):
        df_multi = pd.read_csv(MULTILANG_PATH).fillna("")
        for idx, row in df_multi.iterrows():
            subject = str(row.get("subject", "")).strip()
            body = str(row.get("body", "")).strip()
            resp = str(row.get("answer", "")).strip()
            queue = str(row.get("queue", "")).strip()
            priority = str(row.get("priority", "")).strip().lower()
            lang = str(row.get("language", "en")).strip()
            t_type = str(row.get("type", "")).strip()
            tag1 = str(row.get("tag_1", "")).strip().lower()
            
            msg = f"{subject}\n{body}".strip() if subject else body
            if not msg:
                continue
                
            should_escalate = (
                priority in ["high", "urgent"] or
                t_type == "Incident" or
                tag1 in ["security", "outage", "disruption", "data breach"]
            )
            difficulty = "hard" if priority in ["high", "urgent"] else ("medium" if len(msg) > 150 else "easy")
            
            canonical_records.append({
                "ticket_id": f"APPLE-MULTI-{idx+1:05d}",
                "brand": TARGET_BRAND,
                "source_dataset": "multilang_tickets",
                "category": queue or "APPLE_TECHNICAL",
                "intent": tag1 or "software_ios_update",
                "language": lang,
                "priority": priority or "normal",
                "customer_message": msg,
                "agent_response": resp,
                "should_escalate": should_escalate,
                "difficulty": difficulty,
                "required_facts": [resp[:100]] if resp else []
            })
            
    # 3. Twitter Support Dataset (@AppleSupport Handles)
    if os.path.exists(TWITTER_PATH):
        df_tw = pd.read_csv(TWITTER_PATH).fillna("")
        for idx, row in df_tw.iterrows():
            msg = str(row.get("text", "")).strip()
            if not msg:
                continue
            canonical_records.append({
                "ticket_id": f"APPLE-TW-{idx+1:04d}",
                "brand": TARGET_BRAND,
                "source_dataset": "twitter_support",
                "category": "APPLE_SOCIAL",
                "intent": "general_support_inquiry",
                "language": "en",
                "priority": "normal",
                "customer_message": msg,
                "agent_response": "We are here to help. Please DM us your device model and iOS version so we can assist.",
                "should_escalate": False,
                "difficulty": "easy",
                "required_facts": ["DM device details to @AppleSupport"]
            })
            
    return canonical_records

def audit_dataset(records):
    total_rows = len(records)
    if total_rows == 0:
        return {}
    
    source_counts = {}
    category_counts = {}
    difficulty_counts = {}
    language_counts = {}
    escalation_counts = {True: 0, False: 0}
    pii_counts = {k: 0 for k in PII_PATTERNS}
    duplicate_count = 0
    seen = set()
    
    for rec in records:
        src = rec["source_dataset"]
        source_counts[src] = source_counts.get(src, 0) + 1
        
        cat = rec["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
        diff = rec["difficulty"]
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        lang = rec["language"]
        language_counts[lang] = language_counts.get(lang, 0) + 1
        
        esc = rec["should_escalate"]
        escalation_counts[esc] = escalation_counts.get(esc, 0) + 1
        
        msg = rec["customer_message"]
        if msg in seen:
            duplicate_count += 1
        else:
            seen.add(msg)
            
        for pii_key, pattern in PII_PATTERNS.items():
            if re.search(pattern, msg):
                pii_counts[pii_key] += 1
                
    return {
        "target_brand": TARGET_BRAND,
        "total_records": total_rows,
        "source_distribution": source_counts,
        "category_distribution": category_counts,
        "difficulty_distribution": difficulty_counts,
        "language_distribution": language_counts,
        "escalation_distribution": escalation_counts,
        "duplicate_count": duplicate_count,
        "pii_counts": pii_counts
    }

def partition_and_save(records):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    clean_records = []
    seen = set()
    for rec in records:
        msg = rec["customer_message"].strip()
        if not msg or msg in seen:
            continue
        seen.add(msg)
        clean_records.append(rec)
        
    clean_records.sort(key=lambda x: x["ticket_id"])
    
    n = len(clean_records)
    n_train = int(n * 0.6)
    n_dev = int(n * 0.2)
    
    train_records = clean_records[:n_train]
    dev_records = clean_records[n_train:n_train + n_dev]
    golden_candidates = clean_records[n_train + n_dev:]
    
    with open(PROCESSED_TRAIN, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(PROCESSED_DEV, "w", encoding="utf-8") as f:
        for r in dev_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(PROCESSED_GOLDEN_CANDIDATES, "w", encoding="utf-8") as f:
        for r in golden_candidates:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    data_config = {
        "target_brand": TARGET_BRAND,
        "raw_datasets": {
            "bitext": BITEXT_PATH,
            "multilang": MULTILANG_PATH,
            "twitter": TWITTER_PATH
        },
        "splits": {
            "train": PROCESSED_TRAIN,
            "dev": PROCESSED_DEV,
            "golden_candidates": PROCESSED_GOLDEN_CANDIDATES
        },
        "counts": {
            "total_clean": len(clean_records),
            "train": len(train_records),
            "dev": len(dev_records),
            "golden_candidates": len(golden_candidates)
        }
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False)
        
    return data_config, train_records, dev_records, golden_candidates

def run_full_audit():
    records = load_and_standardize_raw_datasets()
    profile = audit_dataset(records)
    data_config, train, dev, golden = partition_and_save(records)
    print(f"Phase 1 Audit completed for {TARGET_BRAND}! Processed {len(records)} support tickets.")
    return profile, data_config

if __name__ == "__main__":
    run_full_audit()
