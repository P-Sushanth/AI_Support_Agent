import os
import json
import pytest
from src.data.audit import load_and_standardize_raw_datasets, audit_dataset, partition_and_save

def test_load_real_datasets():
    records = load_and_standardize_raw_datasets()
    assert len(records) > 50000
    assert "ticket_id" in records[0]
    assert "customer_message" in records[0]
    assert "source_dataset" in records[0]

def test_audit_real_datasets_schema():
    records = load_and_standardize_raw_datasets()
    profile = audit_dataset(records)
    
    assert "total_records" in profile
    assert profile["total_records"] == len(records)
    assert "source_distribution" in profile
    assert "category_distribution" in profile
    assert "difficulty_distribution" in profile
    assert "pii_counts" in profile
    assert profile["total_records"] > 50000

def test_partition_and_save_real_datasets():
    records = load_and_standardize_raw_datasets()
    data_config, train, dev, golden = partition_and_save(records)
    
    assert os.path.exists(data_config["splits"]["train"])
    assert os.path.exists(data_config["splits"]["dev"])
    assert os.path.exists(data_config["splits"]["golden_candidates"])
    
    assert len(train) > 0
    assert len(dev) > 0
    assert len(golden) > 0
    
    # Verify no overlap between train and golden candidate ticket IDs
    train_ids = {r["ticket_id"] for r in train}
    golden_ids = {r["ticket_id"] for r in golden}
    assert len(train_ids.intersection(golden_ids)) == 0
