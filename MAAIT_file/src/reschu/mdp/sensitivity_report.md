# Engagement MDP — reward sensitivity

Each key reward is halved (×0.5) and doubled (×2), the MDP re-solved, and the new policy run on the same fresh engagements. Score = mean reward under the DEFAULT rewards (one yardstick). Rates: misfire = share of friendlies/civilians declared enemy; let through = share of enemies declared friendly/civilian.

## Small map

| Variant | Score | Correct | Misfire | Let through | USV lost | Unresolved | Time |
|---|---|---|---|---|---|---|---|
| default | 4.04 ± 0.20 | 94.8% | 3.3% | 1.7% | 0.2% | 2.6% | 38 s |
| time cost (step) ×0.5 | 4.01 ± 0.19 | 95.5% | 3.3% | 0.9% | 0.1% | 2.5% | 44 s |
| time cost (step) ×2 | 4.22 ± 0.19 | 95.9% | 3.0% | 1.3% | 0.3% | 1.8% | 34 s |
| USV lost ×0.5 | 4.23 ± 0.18 | 96.0% | 2.5% | 1.6% | 0.2% | 1.8% | 36 s |
| USV lost ×2 | 4.06 ± 0.19 | 94.8% | 3.2% | 1.8% | 0.1% | 2.7% | 38 s |
| firing on friendly/civilian ×0.5 | 3.64 ± 0.25 | 93.2% | 7.9% | 1.9% | 0.2% | 2.1% | 34 s |
| firing on friendly/civilian ×2 | 4.23 ± 0.18 | 96.0% | 2.5% | 1.7% | 0.1% | 1.8% | 37 s |
| letting an enemy through ×0.5 | 4.02 ± 0.20 | 94.8% | 3.3% | 1.8% | 0.2% | 2.6% | 38 s |
| letting an enemy through ×2 | 4.03 ± 0.20 | 95.5% | 3.5% | 0.6% | 0.2% | 2.5% | 43 s |
| leaving a contact unresolved ×0.5 | 4.06 ± 0.20 | 95.0% | 3.3% | 1.7% | 0.2% | 2.5% | 37 s |
| leaving a contact unresolved ×2 | 4.25 ± 0.18 | 96.2% | 2.2% | 1.6% | 0.2% | 1.8% | 36 s |

## Medium map

| Variant | Score | Correct | Misfire | Let through | USV lost | Unresolved | Time |
|---|---|---|---|---|---|---|---|
| default | 4.37 ± 0.15 | 98.8% | 1.4% | 0.3% | 0.1% | 0.4% | 47 s |
| time cost (step) ×0.5 | 4.21 ± 0.15 | 98.7% | 1.8% | 0.1% | 0.1% | 0.4% | 55 s |
| time cost (step) ×2 | 4.23 ± 0.17 | 97.5% | 1.3% | 1.9% | 0.4% | 0.4% | 42 s |
| USV lost ×0.5 | 4.32 ± 0.16 | 98.6% | 1.4% | 0.3% | 0.3% | 0.4% | 47 s |
| USV lost ×2 | 4.38 ± 0.15 | 98.8% | 1.4% | 0.3% | 0.1% | 0.4% | 47 s |
| firing on friendly/civilian ×0.5 | 4.37 ± 0.15 | 98.8% | 1.4% | 0.3% | 0.1% | 0.4% | 47 s |
| firing on friendly/civilian ×2 | 4.37 ± 0.15 | 98.8% | 1.4% | 0.3% | 0.1% | 0.4% | 47 s |
| letting an enemy through ×0.5 | 4.37 ± 0.15 | 98.8% | 1.4% | 0.3% | 0.1% | 0.4% | 47 s |
| letting an enemy through ×2 | 4.33 ± 0.15 | 98.9% | 1.4% | 0.1% | 0.1% | 0.4% | 50 s |
| leaving a contact unresolved ×0.5 | 4.37 ± 0.15 | 98.8% | 1.4% | 0.3% | 0.1% | 0.4% | 47 s |
| leaving a contact unresolved ×2 | 4.38 ± 0.15 | 98.8% | 1.4% | 0.3% | 0.1% | 0.4% | 47 s |

## Large map

| Variant | Score | Correct | Misfire | Let through | USV lost | Unresolved | Time |
|---|---|---|---|---|---|---|---|
| default | 4.10 ± 0.17 | 98.4% | 1.5% | 1.0% | 0.3% | 0.1% | 57 s |
| time cost (step) ×0.5 | 4.12 ± 0.15 | 98.9% | 1.7% | 0.3% | 0.0% | 0.2% | 63 s |
| time cost (step) ×2 | 4.08 ± 0.18 | 97.7% | 1.8% | 1.8% | 0.4% | 0.1% | 52 s |
| USV lost ×0.5 | 3.91 ± 0.19 | 97.9% | 1.2% | 0.9% | 0.9% | 0.1% | 57 s |
| USV lost ×2 | 4.18 ± 0.16 | 98.7% | 1.6% | 0.9% | 0.1% | 0.1% | 58 s |
| firing on friendly/civilian ×0.5 | 4.04 ± 0.18 | 98.1% | 2.4% | 0.9% | 0.2% | 0.1% | 57 s |
| firing on friendly/civilian ×2 | 4.10 ± 0.17 | 98.4% | 1.5% | 1.0% | 0.3% | 0.1% | 57 s |
| letting an enemy through ×0.5 | 4.10 ± 0.17 | 98.3% | 1.5% | 1.1% | 0.3% | 0.1% | 56 s |
| letting an enemy through ×2 | 4.09 ± 0.17 | 98.8% | 1.6% | 0.3% | 0.3% | 0.1% | 60 s |
| leaving a contact unresolved ×0.5 | 4.10 ± 0.17 | 98.4% | 1.5% | 1.0% | 0.3% | 0.1% | 57 s |
| leaving a contact unresolved ×2 | 4.09 ± 0.17 | 98.4% | 1.6% | 1.0% | 0.3% | 0.1% | 57 s |

