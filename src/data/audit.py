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

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
}

ESCALATION_INTENTS = {
    "security", "security_incident", "data_breach", "account_hack",
    "refund_override", "fraud", "cancel_account", "legal_threat"
}

def load_and_standardize_raw_datasets():
    canonical_records = []
    
    # 1. Bitext Dataset
    if os.path.exists(BITEXT_PATH):
        df_bitext = pd.read_csv(BITEXT_PATH).fillna("")
        for idx, row in df_bitext.iterrows():
            intent = str(row.get("intent", "")).strip().lower()
            category = str(row.get("category", "")).strip().upper()
            msg = str(row.get("instruction", "")).strip()
            resp = str(row.get("response", "")).strip()
            
            if not msg:
                continue
                
            should_escalate = any(k in intent for k in ESCALATION_INTENTS)
            difficulty = "hard" if should_escalate else ("medium" if len(msg) > 100 else "easy")
            
            canonical_records.append({
                "ticket_id": f"BITEXT-{idx+1:05d}",
                "source_dataset": "bitext_customer_support",
                "category": category or "GENERAL",
                "intent": intent or "general_query",
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
                "ticket_id": f"MULTI-{idx+1:05d}",
                "source_dataset": "multilang_tickets",
                "category": queue or "TECHNICAL",
                "intent": tag1 or t_type.lower() or "support_issue",
                "language": lang,
                "priority": priority or "normal",
                "customer_message": msg,
                "agent_response": resp,
                "should_escalate": should_escalate,
                "difficulty": difficulty,
                "required_facts": [resp[:100]] if resp else []
            })
            
    # 3. Twitter Support Sample
    if os.path.exists(TWITTER_PATH):
        df_tw = pd.read_csv(TWITTER_PATH).fillna("")
        for idx, row in df_tw.iterrows():
            msg = str(row.get("text", "")).strip()
            if not msg:
                continue
            canonical_records.append({
                "ticket_id": f"TWITTER-{idx+1:04d}",
                "source_dataset": "twitter_support",
                "category": "SOCIAL_SUPPORT",
                "intent": "twitter_query",
                "language": "en",
                "priority": "normal",
                "customer_message": msg,
                "agent_response": "Thank you for reaching out. Please DM us your account details to assist.",
                "should_escalate": False,
                "difficulty": "easy",
                "required_facts": ["DM account details for support"]
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
    
    # Clean exact duplicates
    clean_records = []
    seen = set()
    for rec in records:
        msg = rec["customer_message"].strip()
        if not msg or msg in seen:
            continue
        seen.add(msg)
        clean_records.append(rec)
        
    # Sort deterministically
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

def generate_report(profile, data_config):
    report_content = f"""# Comprehensive Real Dataset Audit Report

## Executive Summary
This project incorporates three real-world customer support datasets totaling **{profile['total_records']:,}** records across **{len(profile['category_distribution'])}** support categories.

- **Clean Processed Records**: {data_config['counts']['total_clean']:,}
- **Duplicates Excluded**: {profile['duplicate_count']:,}
- **Train Split (60%)**: {data_config['counts']['train']:,} records
- **Dev Split (20%)**: {data_config['counts']['dev']:,} records
- **Golden Candidate Split (20%)**: {data_config['counts']['golden_candidates']:,} records

---

## 1. Source Breakdown
| Dataset Source | Records | Description |
| --- | --- | --- |
| `bitext_customer_support` | {profile['source_distribution'].get('bitext_customer_support', 0):,} | 27K intent-focused customer queries & responses |
| `multilang_tickets` | {profile['source_distribution'].get('multilang_tickets', 0):,} | Enterprise ticket logs with priority, queues & languages |
| `twitter_support` | {profile['source_distribution'].get('twitter_support', 0):,} | Social media customer support interactions |

---

## 2. Difficulty & Language Stratification

### Difficulty Distribution
| Difficulty | Count | Percentage |
| --- | --- | --- |
| `easy` | {profile['difficulty_distribution'].get('easy', 0):,} | {profile['difficulty_distribution'].get('easy', 0)/profile['total_records']*100:.1f}% |
| `medium` | {profile['difficulty_distribution'].get('medium', 0):,} | {profile['difficulty_distribution'].get('medium', 0)/profile['total_records']*100:.1f}% |
| `hard` | {profile['difficulty_distribution'].get('hard', 0):,} | {profile['difficulty_distribution'].get('hard', 0)/profile['total_records']*100:.1f}% |

### Primary Languages
| Language | Ticket Count |
| --- | --- |
"""
    for lang, cnt in sorted(profile['language_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
        report_content += f"| `{lang}` | {cnt:,} |\n"

    report_content += f"""
---

## 3. Escalation Requirements (`should_escalate`)
| Escalation Required | Record Count | Percentage |
| --- | --- | --- |
| `True` (Requires Human Support / Security Escalation) | {profile['escalation_distribution'].get(True, 0):,} | {profile['escalation_distribution'].get(True, 0)/profile['total_records']*100:.1f}% |
| `False` (Automated Resolution Eligible) | {profile['escalation_distribution'].get(False, 0):,} | {profile['escalation_distribution'].get(False, 0)/profile['total_records']*100:.1f}% |

---

## 4. PII Audit Findings
Matches detected across customer message content:
- Email patterns: {profile['pii_counts']['email']:,}
- Phone patterns: {profile['pii_counts']['phone']:,}
- IP address patterns: {profile['pii_counts']['ip_address']:,}

---

## 5. Leakage Prevention Strategy
All evaluation subsets (`golden_candidate.jsonl`) are strictly partitioned deterministically based on ticket IDs and stored in `data/processed/golden_candidate.jsonl`. No records in `golden_candidate.jsonl` will be indexed into the RAG vector store.
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

def run_full_audit():
    records = load_and_standardize_raw_datasets()
    profile = audit_dataset(records)
    data_config, train, dev, golden = partition_and_save(records)
    generate_report(profile, data_config)
    print(f"Phase 1 Audit completed! Processed {len(records)} real support tickets.")
    return profile, data_config

if __name__ == "__main__":
    run_full_audit()
