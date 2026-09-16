import os
import json
import pytest
from src.data.audit import load_and_standardize_raw_datasets, audit_and_redact_pii, run_dataset_audit

def test_load_apple_support_dataset():
    records = load_and_standardize_raw_datasets()
    assert len(records) > 0
    assert "ticket_id" in records[0]
    assert "customer_message" in records[0]
    assert records[0]["brand"] == "@AppleSupport"

def test_audit_pii_redaction():
    sample_records = [
        {"customer_message": "Contact me at user@example.com or 555-0192"}
    ]
    clean_records, pii_counts = audit_and_redact_pii(sample_records)
    assert pii_counts["email"] == 1
    assert pii_counts["phone"] == 1
    assert "[EMAIL_REDACTED]" in clean_records[0]["customer_message"]

def test_dataset_audit_pipeline():
    profile = run_dataset_audit()
    assert profile["total_records"] > 0
    assert profile["data_leakage_overlap"] == 0
    assert os.path.exists("data/processed/train.jsonl")
    assert os.path.exists("data/processed/dev.jsonl")
    assert os.path.exists("data/processed/golden_candidate.jsonl")
