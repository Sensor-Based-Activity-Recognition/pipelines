# RUN: 2023-04-21 23-44-30
## Hyperparameters
| Path        | Param                            | HEAD^   | workspace   |
|-------------|----------------------------------|---------|-------------|
| params.yaml | dvclive.model_hparams.num_epochs | 2       | 3           |

## Metrics
| Path                 | Metric           | HEAD^   | workspace   | Change   |
|----------------------|------------------|---------|-------------|----------|
| dvclive/metrics.json | epoch            | 2       | 3           | 1        |
| dvclive/metrics.json | step             | 354     | 531         | 177      |
| dvclive/metrics.json | test.epoch.acc   | 0.3491  | 0.37706     | 0.02796  |
| dvclive/metrics.json | test.epoch.f1    | 0.25647 | 0.29407     | 0.03759  |
| dvclive/metrics.json | test.step.acc    | 0.3491  | 0.37706     | 0.02796  |
| dvclive/metrics.json | test.step.f1     | 0.25647 | 0.29407     | 0.03759  |
| dvclive/metrics.json | train.epoch.acc  | 0.81162 | 0.89319     | 0.08157  |
| dvclive/metrics.json | train.epoch.f1   | 0.76306 | 0.85782     | 0.09476  |
| dvclive/metrics.json | train.epoch.loss | 0.53597 | 0.33408     | -0.20189 |
| dvclive/metrics.json | train.step.acc   | 0.9375  | 0.96875     | 0.03125  |
| dvclive/metrics.json | train.step.f1    | 0.93225 | 0.95789     | 0.02564  |
| dvclive/metrics.json | train.step.loss  | 0.30755 | 0.10856     | -0.19899 |
| dvclive/metrics.json | val.epoch.acc    | 0.85795 | 0.89823     | 0.04028  |
| dvclive/metrics.json | val.epoch.f1     | 0.83348 | 0.87489     | 0.04141  |
| dvclive/metrics.json | val.epoch.loss   | 0.43421 | 0.32432     | -0.10989 |
| dvclive/metrics.json | val.step.f1      | 0.64286 | 0.66667     | 0.02381  |
| dvclive/metrics.json | val.step.loss    | 0.23401 | 0.78875     | 0.55474  |

# Plots
TODO
