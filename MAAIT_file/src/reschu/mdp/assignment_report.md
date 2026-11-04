# MDP team assignment — mission evaluation

600 random missions (15 min, 3 MDP USVs, 6-9 contacts of random types, random map size), each run under every assignment rule on the same seed. The same engagement MDP drives every USV; only the choice of contact differs.

| Assignment | Mission score (95% CI) | Score per contact | Correct | Wrong | Unresolved | Friendly/civilian fired on | Aggressors neutralised | USVs lost | Mean time to ID |
|---|---|---|---|---|---|---|---|---|---|
| value (MDP team, total gain) | 40.39 ± 1.59 | 3.74 | 82.8% | 3.4% | 13.8% | 0.34 / mission | 79.9% | 0.00 / mission | 4.9 min |
| rate (MDP team, gain per minute) | 43.26 ± 1.62 | 3.89 | 85.9% | 3.6% | 10.5% | 0.33 / mission | 78.6% | 0.00 / mission | 4.6 min |
| nearest (engine, closest boat) | 41.12 ± 1.73 | 3.67 | 82.8% | 3.2% | 14.0% | 0.26 / mission | 76.4% | 0.00 / mission | 4.3 min |

Paired differences against the engine's nearest-boat rule (same seeds):

- value − nearest: -0.73 ± 2.15 per mission (better in 40%, worse in 40%)
- rate − nearest: +2.14 ± 2.14 per mission (better in 44%, worse in 36%)

By map size (mean mission score, value / rate / nearest):

- small: 36.25 / 41.82 / 39.50 (224 missions)
- medium: 45.40 / 47.72 / 44.96 (188 missions)
- large: 40.29 / 40.52 / 39.20 (188 missions)
