# Commercialization Notes for YDChat

This project is designed for commercial product development if you follow licensing and compliance constraints.

## What you can sell

- Hosted inference APIs using your own YDChat checkpoints
- On-prem deployments built from this codebase
- Fine-tuned domain models trained from your own licensed data
- Products that embed YDChat generation as a feature

## What to avoid

- Shipping checkpoints trained on unlicensed or scraped copyrighted data without rights
- Mixing private customer data into training without consent and processing agreements
- Claiming model provenance you cannot prove
- Releasing third-party pretrained weights under this brand

## Compliance checklist

- Maintain dataset inventory with source, license, and collection date
- Keep records of filtering, deduplication, and PII handling
- Define retention and deletion rules for logs and training artifacts
- Add security controls for model endpoints (auth, rate limits, abuse monitoring)
- Run legal review for regional requirements (privacy, AI transparency, sector rules)

## Data governance

- Prefer explicit contracts or public datasets with commercial terms
- Track user opt-in/opt-out where applicable
- Avoid storing raw sensitive data in checkpoints and logs

## Shipping guidance

- Tag every release with config + data snapshot IDs
- Keep reproducible training metadata (seed, hyperparams, commit hash)
- Version checkpoints with clear provenance

This document is operational guidance and not legal advice.
