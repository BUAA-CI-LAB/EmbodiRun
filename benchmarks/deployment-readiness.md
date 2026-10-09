# Deployment readiness evidence

This template records the deployment and operations comparison required by the V1 open-source plan. Fill it with one measured run per row; leave `TBD` when the measurement is unavailable.

## Test identity

| Field | Value |
|---|---|
| Date | TBD |
| EmbodiRun revision | TBD |
| EmbodiInfer revision | TBD |
| Recipe / hardware | TBD |
| Model and checkpoint hash | TBD |
| Host OS, Python, uv/Docker | TBD |
| GPU, driver, CUDA | TBD |

## With and without EmbodiRun

| Measure | Manual deployment | EmbodiRun | Evidence |
|---|---:|---:|---|
| First successful deployment time | TBD | TBD | command log |
| Manual setup steps | TBD | TBD | checklist |
| Backend replacement time | TBD | TBD | before/after config |
| Adding one device time | TBD | TBD | deployment diff |
| Update and restart steps | TBD | TBD | command log |
| Fault localization time | TBD | TBD | incident record |
| CPU / RAM / VRAM | TBD | TBD | profiler output |

The comparison must keep model, checkpoint, hardware, and task fixed. Do not mark a row as measured when it is inferred from code inspection.

## Multi-device scaling

Record one row for each device count and retain the raw output and plotting command.

| Device count | Backend layout | Request rate | p50 / p95 inference | Control-loop p50 / p95 | GPU memory | Failure rate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD |

## Ownership

The person adding a result must include the command, configuration, raw output path, and hardware identity in the pull request. Until that evidence exists, the support matrix and README should keep the `🟨 待补充` / `🟨 Pending` marker.
