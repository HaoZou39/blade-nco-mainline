# P6-A Baseline Reproduction

The frozen cache and exact solver reproduce all 10,000 survivor hashes in identical order.

| Metric | Frozen reference | Reproduced |
|---|---:|---:|
| Candidate count | 119,530 | 119,530 |
| Survivor count | 10,000 | 10,000 |
| Random mean assignment cost | 25.8676 | 25.867617 |
| Guided mean assignment cost | 15.5665 | 15.566525 |
| Random 5D SWD | 1.7076 | 1.707605 |
| Guided 5D SWD | 1.2706 | 1.270578 |
| Nearest-target distance_min contribution | 83.7% | 83.7276% |

Frozen matching config SHA256: `77e2f8374f2547760f165cdc952436c55fc0e0eb498b3e7752db1a66e1394666`. Candidate representation SHA256: `7670dca68aa869a0ff63ee5b381932cd5af6e1744959859f01445928046bd546`. Saved survivor manifest SHA256: `e3a56c7926ff08db48b4e54df1a83f66955f8eb0bd5e0d26973f9f81f733e2e2`. Baseline solver status is exact/integral with the same float32 cost matrix and 208/209 quotas. Historical matching wall was 66.240s; this audit's independent rerun is recorded in `logs/runtime_events.csv`.
