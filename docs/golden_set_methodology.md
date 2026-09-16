# Golden Evaluation Set Methodology

## Dataset Source
Built exclusively from real customer-agent conversations in **Customer Support on Twitter** (Kaggle dataset `thoughtvector/customer-support-on-twitter` / `TNE-AI/customer-support-on-twitter-conversation`), targeting **@AppleSupport**.

## Intent Taxonomy (6 Intents)
1. `ios_update_performance`: iOS update installation issues, post-update battery drain, device slowdowns.
2. `account_icloud_security`: Apple ID password resets, 2FA lockouts, compromised Apple ID alerts.
3. `hardware_battery_repair`: AppleCare+ coverage, screen damage, swollen battery safety hazard.
4. `app_store_billing`: Unexpected charges, Report A Problem refunds, unauthorized child purchases.
5. `connectivity_accessory`: AirPods setup/reset, Apple Watch Wi-Fi disconnections, Bluetooth troubleshooting.
6. `general_troubleshooting`: Screen recording, Night Shift, general settings guidance.

## Stratification & Leakage Protection
- **Size**: 200 real @AppleSupport evaluation tickets.
- **Leakage Protection**: Sampled from isolated `golden_candidate.jsonl` split (0.0% overlap with the 606 indexed RAG train chunks).
- **Human Spot Checks**: 30 examples annotated in `data/human_eval/human_spot_checks.jsonl` for inter-rater agreement validation.
