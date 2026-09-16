# Golden Set Methodology

## Objective
Construct a high-quality, stratified evaluation golden set representing real customer inquiries while ensuring strict zero data leakage into the retrieval index.

## Target Size & Composition
- **Total Golden Examples**: 200
- **Data Leakage Check**: 0 overlapping records found in retrieval index (0.0% leakage rate).

## Stratification Strategy
Samples were drawn deterministically from `data/processed/golden_candidate.jsonl` using stratified sampling across categories (ORDER, BILLING, TECHNICAL, ACCOUNT, GENERAL, SOCIAL_SUPPORT) and difficulty levels (easy, medium, hard).

## Human Annotation & Reference Verification
Each reference answer represents verified support guidance. Acceptable variations, required facts, and mandatory escalation requirements are explicitly specified per record schema.
