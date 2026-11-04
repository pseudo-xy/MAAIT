# Engagement MDP — build report

Built 2026-10-25T11:22:44.

Rewards: usv_lost = -25, step = -0.2

## Small map

12000 exploration + 8000 refinement engagements, 190346 transitions; value iteration converged in 150 sweeps.

| Policy | Score (95% CI) | Correct | Declared correctly when declaring | USV lost | Unresolved | Mean time |
|---|---|---|---|---|---|---|
| mdp | 3.99 ± 0.15 | 94.8% | 97.2% | 0.4% | 2.0% | 38 s |
| rush | -0.23 ± 0.33 | 75.9% | 76.4% | 0.4% | 0.4% | 16 s |
| threshold | 3.68 ± 0.19 | 94.2% | 95.9% | 0.5% | 1.1% | 35 s |

By contact type (score / correct / lost / unresolved / time):

| Type | mdp | rush | threshold |
|---|---|---|---|
| civilian | 1.6 / 97% / 0% / 0% / 40 s | -1.1 / 83% / 0% / 0% / 13 s | -0.4 / 90% / 0% / 0% / 56 s |
| standoff_low | 3.4 / 95% / 0% / 0% / 44 s | 2.0 / 82% / 0% / 0% / 15 s | 4.2 / 99% / 0% / 0% / 34 s |
| friendly | 2.4 / 86% / 0% / 10% / 35 s | -13.8 / 25% / 0% / 2% / 16 s | 1.9 / 86% / 0% / 6% / 31 s |
| aggressive | 8.5 / 98% / 2% / 0% / 33 s | 8.3 / 96% / 2% / 0% / 19 s | 8.4 / 97% / 3% / 0% / 26 s |
| standoff_med | 4.1 / 99% / 0% / 0% / 36 s | 3.8 / 94% / 0% / 0% / 18 s | 4.4 / 99% / 0% / 0% / 27 s |

Policy by distance band (share of visited states choosing each action):

| Band | approach | hold | back_off | declare_enemy | declare_friendly | declare_civilian | abandon |
|---|---|---|---|---|---|---|---|
| danger | 1% | – | 7% | 88% | 2% | 2% | 1% |
| id | 5% | 10% | 25% | 29% | 8% | 22% | 0% |
| cam_near | 38% | 22% | 30% | 3% | 2% | 5% | – |
| cam_far | 30% | 49% | 4% | 16% | – | 1% | – |
| mid | 48% | 22% | 0% | 25% | 3% | 2% | 0% |
| far | 99% | 0% | – | – | 1% | – | 0% |

Most-visited states and the chosen action:

- `far | P(enemy) 0.75-0.90 (most likely) | P(aggressor) 0.40-1.00` → **approach** (visits 7445, value 5.43)
- `far | P(enemy) 0.60-0.75 (most likely) | P(aggressor) 0.10-0.40` → **approach** (visits 6539, value 4.57)
- `cam_near | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 6449, value 4.72)
- `id | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **declare_enemy** (visits 6389, value 4.83)
- `far | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.40-1.00` → **approach** (visits 5126, value 6.77)
- `id | P(civilian) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **declare_civilian** (visits 4980, value 2.99)
- `far | P(enemy) 0.60-0.75 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 4235, value 3.33)
- `far | P(enemy) 0.00-0.60 (most likely) | P(aggressor) 0.10-0.40` → **approach** (visits 3856, value 4.00)
- `mid | P(enemy) 0.60-0.75 (most likely) | P(aggressor) 0.10-0.40` → **hold** (visits 3784, value 4.89)
- `far | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 3583, value 3.97)
- `mid | P(enemy) 0.75-0.90 (most likely) | P(aggressor) 0.40-1.00` → **hold** (visits 3448, value 5.96)
- `far | P(enemy) 0.00-0.60 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 3242, value 3.06)

## Medium map

12000 exploration + 8000 refinement engagements, 198730 transitions; value iteration converged in 119 sweeps.

| Policy | Score (95% CI) | Correct | Declared correctly when declaring | USV lost | Unresolved | Mean time |
|---|---|---|---|---|---|---|
| mdp | 4.37 ± 0.11 | 98.7% | 99.2% | 0.3% | 0.3% | 46 s |
| rush | 0.16 ± 0.31 | 78.4% | 78.6% | 0.0% | 0.2% | 30 s |
| threshold | 4.02 ± 0.14 | 97.1% | 97.7% | 0.1% | 0.4% | 49 s |

By contact type (score / correct / lost / unresolved / time):

| Type | mdp | rush | threshold |
|---|---|---|---|
| standoff_med | 4.1 / 100% / 0% / 0% / 44 s | 3.5 / 94% / 0% / 0% / 32 s | 4.1 / 99% / 0% / 0% / 42 s |
| civilian | 1.8 / 99% / 0% / 0% / 51 s | -2.2 / 80% / 0% / 0% / 24 s | 0.7 / 95% / 0% / 0% / 64 s |
| standoff_low | 3.8 / 99% / 0% / 0% / 50 s | 1.5 / 80% / 0% / 0% / 27 s | 3.7 / 98% / 0% / 0% / 48 s |
| aggressive | 8.6 / 98% / 2% / 0% / 44 s | 8.7 / 99% / 0% / 0% / 37 s | 8.9 / 99% / 1% / 0% / 44 s |
| friendly | 3.7 / 97% / 0% / 1% / 43 s | -10.6 / 39% / 0% / 1% / 31 s | 2.9 / 94% / 0% / 2% / 48 s |

Policy by distance band (share of visited states choosing each action):

| Band | approach | hold | back_off | declare_enemy | declare_friendly | declare_civilian | abandon |
|---|---|---|---|---|---|---|---|
| danger | 4% | – | 2% | 88% | 2% | 0% | 3% |
| id | 22% | 1% | 11% | 34% | 11% | 21% | 0% |
| cam_near | 62% | 13% | 10% | 7% | 8% | – | 0% |
| cam_far | 54% | 14% | 1% | 29% | – | 2% | 0% |
| mid | 81% | 3% | – | 11% | 3% | 2% | 0% |
| far | 100% | 0% | – | – | – | – | 0% |

Most-visited states and the chosen action:

- `id | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **declare_enemy** (visits 8102, value 4.85)
- `far | P(enemy) 0.60-0.75 (most likely) | P(aggressor) 0.10-0.40` → **approach** (visits 7387, value 4.60)
- `far | P(enemy) 0.75-0.90 (most likely) | P(aggressor) 0.40-1.00` → **approach** (visits 7291, value 5.70)
- `cam_near | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 5273, value 4.72)
- `far | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.40-1.00` → **approach** (visits 5262, value 6.97)
- `id | P(civilian) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **declare_civilian** (visits 5212, value 2.97)
- `far | P(enemy) 0.60-0.75 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 4826, value 3.08)
- `far | P(enemy) 0.00-0.60 (most likely) | P(aggressor) 0.10-0.40` → **approach** (visits 4466, value 4.08)
- `far | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 4400, value 3.66)
- `mid | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.40-1.00` → **declare_enemy** (visits 4060, value 7.92)
- `far | P(enemy) 0.00-0.60 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 3743, value 2.86)
- `far | P(civilian) 0.60-0.75 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 3526, value 2.14)

## Large map

12000 exploration + 8000 refinement engagements, 200090 transitions; value iteration converged in 105 sweeps.

| Policy | Score (95% CI) | Correct | Declared correctly when declaring | USV lost | Unresolved | Mean time |
|---|---|---|---|---|---|---|
| mdp | 4.13 ± 0.11 | 98.5% | 99.0% | 0.3% | 0.2% | 58 s |
| rush | 0.31 ± 0.30 | 80.5% | 80.7% | 0.0% | 0.1% | 43 s |
| threshold | 3.58 ± 0.16 | 96.3% | 96.4% | 0.0% | 0.1% | 61 s |

By contact type (score / correct / lost / unresolved / time):

| Type | mdp | rush | threshold |
|---|---|---|---|
| aggressive | 8.2 / 98% / 1% / 0% / 62 s | 8.2 / 98% / 0% / 0% / 56 s | 8.7 / 100% / 0% / 0% / 62 s |
| standoff_low | 3.5 / 98% / 0% / 0% / 58 s | 1.8 / 84% / 0% / 0% / 35 s | 3.4 / 97% / 0% / 0% / 55 s |
| civilian | 1.6 / 99% / 0% / 0% / 57 s | -1.9 / 82% / 0% / 0% / 34 s | 0.3 / 94% / 0% / 0% / 72 s |
| friendly | 3.6 / 98% / 0% / 1% / 52 s | -10.0 / 43% / 0% / 1% / 42 s | 1.7 / 91% / 0% / 1% / 61 s |
| standoff_med | 3.7 / 100% / 0% / 0% / 59 s | 3.4 / 96% / 0% / 0% / 48 s | 3.8 / 99% / 0% / 0% / 57 s |

Policy by distance band (share of visited states choosing each action):

| Band | approach | hold | back_off | declare_enemy | declare_friendly | declare_civilian | abandon |
|---|---|---|---|---|---|---|---|
| danger | – | – | 7% | 82% | 5% | – | 5% |
| id | 25% | – | 9% | 37% | 13% | 17% | 0% |
| cam_near | 54% | 25% | 3% | 8% | 10% | – | 0% |
| cam_far | 71% | 4% | 2% | 9% | 9% | 4% | 0% |
| mid | 87% | – | – | 12% | 1% | – | 0% |
| far | 100% | – | – | – | – | – | 0% |

Most-visited states and the chosen action:

- `id | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **declare_enemy** (visits 9092, value 4.91)
- `far | P(enemy) 0.60-0.75 (most likely) | P(aggressor) 0.10-0.40` → **approach** (visits 8777, value 4.08)
- `far | P(enemy) 0.75-0.90 (most likely) | P(aggressor) 0.40-1.00` → **approach** (visits 7364, value 5.12)
- `far | P(enemy) 0.60-0.75 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 5640, value 2.88)
- `far | P(enemy) 0.00-0.60 (most likely) | P(aggressor) 0.10-0.40` → **approach** (visits 4860, value 3.66)
- `far | P(enemy) 0.00-0.60 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 4598, value 2.68)
- `id | P(civilian) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **declare_civilian** (visits 4429, value 2.97)
- `far | P(enemy) 0.75-0.90 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 4293, value 3.13)
- `far | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.40-1.00` → **approach** (visits 4223, value 6.35)
- `far | P(enemy) 0.75-0.90 (most likely) | P(aggressor) 0.10-0.40` → **approach** (visits 4163, value 4.20)
- `far | P(civilian) 0.60-0.75 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 3985, value 1.95)
- `far | P(enemy) 0.97-1.00 (most likely) | P(aggressor) 0.00-0.10` → **approach** (visits 3809, value 3.51)

