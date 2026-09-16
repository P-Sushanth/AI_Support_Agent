# Phase 1: @AppleSupport Dataset Audit & Isolation Report

## Dataset Summary
- **Target Brand**: `@AppleSupport`
- **Total Standardized Records**: `1011`
- **Train Split (Knowledge Base)**: `606` records (60%)
- **Dev Split (Local Tuning)**: `202` records (20%)
- **Golden Candidate Split**: `203` records (20%)
- **Data Leakage Overlap**: `0` records (0.0% Leakage ✅)

## PII Audit
- **Email Redactions**: 0
- **Phone Redactions**: 328
- **IP Address Redactions**: 0

## Defined Apple Support Intents
{
  "ios_update_performance": 604,
  "general_troubleshooting": 284,
  "connectivity_accessory": 42,
  "hardware_battery_repair": 38,
  "account_icloud_security": 24,
  "app_store_billing": 19
}
