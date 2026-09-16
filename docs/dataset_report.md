# Comprehensive Real Dataset Audit Report

## Executive Summary
This project incorporates three real-world customer support datasets totaling **55,552** records across **22** support categories.

- **Clean Processed Records**: 53,315
- **Duplicates Excluded**: 2,237
- **Train Split (60%)**: 31,989 records
- **Dev Split (20%)**: 10,663 records
- **Golden Candidate Split (20%)**: 10,663 records

---

## 1. Source Breakdown
| Dataset Source | Records | Description |
| --- | --- | --- |
| `bitext_customer_support` | 26,872 | 27K intent-focused customer queries & responses |
| `multilang_tickets` | 28,587 | Enterprise ticket logs with priority, queues & languages |
| `twitter_support` | 93 | Social media customer support interactions |

---

## 2. Difficulty & Language Stratification

### Difficulty Distribution
| Difficulty | Count | Percentage |
| --- | --- | --- |
| `easy` | 28,778 | 51.8% |
| `medium` | 15,596 | 28.1% |
| `hard` | 11,178 | 20.1% |

### Primary Languages
| Language | Ticket Count |
| --- | --- |
| `en` | 43,303 |
| `de` | 12,249 |

---

## 3. Escalation Requirements (`should_escalate`)
| Escalation Required | Record Count | Percentage |
| --- | --- | --- |
| `True` (Requires Human Support / Security Escalation) | 20,015 | 36.0% |
| `False` (Automated Resolution Eligible) | 35,537 | 64.0% |

---

## 4. PII Audit Findings
Matches detected across customer message content:
- Email patterns: 0
- Phone patterns: 1,241
- IP address patterns: 2

---

## 5. Leakage Prevention Strategy
All evaluation subsets (`golden_candidate.jsonl`) are strictly partitioned deterministically based on ticket IDs and stored in `data/processed/golden_candidate.jsonl`. No records in `golden_candidate.jsonl` will be indexed into the RAG vector store.
