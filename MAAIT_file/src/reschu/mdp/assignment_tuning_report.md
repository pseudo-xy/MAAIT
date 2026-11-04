# MDP team assignment — tuning and confirmation

Tuning on 200 separate missions (gain per minute rule):

| Switching bonus | Time floor (min) | Mission score |
|---|---|---|
| 1 | 0.5 | 46.13 ± 2.63 |
| 0 | 0.5 | 44.62 ± 2.68 |
| 4 | 1 | 43.62 ± 2.64 |
| 2 | 2 | 43.59 ± 2.60 |
| 4 | 0.5 | 43.44 ± 2.51 |
| 4 | 2 | 43.29 ± 2.57 |
| 0 | 2 | 43.02 ± 2.60 |
| 2 | 0.5 | 42.78 ± 2.73 |
| 2 | 1 | 42.30 ± 2.60 |
| 2 | 0.25 | 42.24 ± 2.57 |
| 1 | 2 | 42.15 ± 2.68 |
| 1 | 0.25 | 41.91 ± 2.74 |
| 4 | 0.25 | 41.87 ± 2.81 |
| 0 | 0.25 | 41.51 ± 3.00 |
| 1 | 1 | 41.20 ± 2.54 |
| 0 | 1 | 41.06 ± 2.83 |

Confirmation on 1500 FRESH missions, tuned settings {'SWITCH_MARGIN': 1.0, 'RATE_FLOOR_MIN': 0.5}:

| Rule | Mission score | Correct | Unresolved | Friendly/civilian fired on | Aggressors neutralised | USVs lost |
|---|---|---|---|---|---|---|
| rate | 42.10 ± 1.07 | 84.7% | 11.6% | 0.33 | 77.7% | 0.00 |
| nearest | 39.99 ± 1.11 | 81.9% | 14.8% | 0.26 | 73.7% | 0.00 |

- Mission score, rate − nearest: +2.115 ± 1.404 (z = 3.0)
- Correct share, rate − nearest: +0.028 ± 0.009 (z = 6.4)
- Unresolved share, rate − nearest: -0.032 ± 0.008 (z = -7.5)
- Fired on per mission, rate − nearest: +0.072 ± 0.035 (z = 4.1)
