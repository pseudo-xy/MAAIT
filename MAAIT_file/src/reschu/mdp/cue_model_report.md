# MDP cue model — fit report

Fitted 2026-10-25T10:33:42. Window = 10 ticks (10 s). Speed variation: spread 0.15, jitter 0.05. Sensor noise: 1.0. Camera looks in training: 35481.
Training windows: 71567 · tempering: tau_speed = 1.0, tau_beh = 0.7, tau_cam = 0.85 (validation log-loss 0.6847).

## Held-out accuracy as evidence accumulates

| Observed for | Contacts | 5-type accuracy | Declaration accuracy (enemy/friendly/civilian) | Calibration error |
|---|---|---|---|---|
| 10 s | 796 | 62.6% | 62.7% | 0.089 |
| 20 s | 781 | 67.1% | 69.7% | 0.053 |
| 30 s | 766 | 70.0% | 74.2% | 0.049 |
| 60 s | 731 | 80.6% | 85.5% | 0.013 |
| 120 s | 637 | 88.5% | 93.9% | 0.014 |
| 180 s | 563 | 94.3% | 97.2% | 0.009 |

## Declaration confusion after 60 s (rows = true type)

| True type | → enemy | → friendly | → civilian |
|---|---|---|---|
| civilian | 16% | 0% | 84% |
| friendly | 20% | 79% | 1% |
| standoff_low | 87% | 1% | 13% |
| standoff_med | 90% | 10% | 0% |
| aggressive | 95% | 4% | 1% |
