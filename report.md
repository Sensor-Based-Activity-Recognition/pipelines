# RUN: 2023-04-21 23-36-50
## Hyperparameters
| Path        | Param                               | HEAD^   | workspace   |
|-------------|-------------------------------------|---------|-------------|
| params.yaml | dvclive.model_hparams.learning_rate | 0.03    | 0.1         |
| params.yaml | dvclive.model_hparams.weight_decay  | 0.001   | 0.0005      |

## Metrics
| Path                 | Metric           | HEAD^   | workspace   | Change   |
|----------------------|------------------|---------|-------------|----------|
| dvclive/metrics.json | test.epoch.acc   | 0.44875 | 0.3491      | -0.09964 |
| dvclive/metrics.json | test.epoch.f1    | 0.36818 | 0.25647     | -0.1117  |
| dvclive/metrics.json | test.step.acc    | 0.44875 | 0.3491      | -0.09964 |
| dvclive/metrics.json | test.step.f1     | 0.36818 | 0.25647     | -0.1117  |
| dvclive/metrics.json | train.epoch.acc  | 0.73694 | 0.81162     | 0.07468  |
| dvclive/metrics.json | train.epoch.f1   | 0.65581 | 0.76306     | 0.10725  |
| dvclive/metrics.json | train.epoch.loss | 0.74477 | 0.53597     | -0.2088  |
| dvclive/metrics.json | train.step.f1    | 0.89444 | 0.93225     | 0.03781  |
| dvclive/metrics.json | train.step.loss  | 0.29613 | 0.30755     | 0.01142  |
| dvclive/metrics.json | val.epoch.acc    | 0.82049 | 0.85795     | 0.03746  |
| dvclive/metrics.json | val.epoch.f1     | 0.79137 | 0.83348     | 0.04211  |
| dvclive/metrics.json | val.epoch.loss   | 0.55358 | 0.43421     | -0.11937 |
| dvclive/metrics.json | val.step.f1      | 0.66667 | 0.64286     | -0.02381 |
| dvclive/metrics.json | val.step.loss    | 0.95916 | 0.23401     | -0.72515 |

# Plots
TODO
