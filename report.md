# RUN: 2023-04-21 23-02-34
## Hyperparameters
| Path        | Param                               | HEAD~1   | workspace   |
|-------------|-------------------------------------|----------|-------------|
| params.yaml | dvclive.model_hparams.learning_rate | 0.1      | 0.03        |
| params.yaml | dvclive.model_hparams.weight_decay  | 0.0005   | 0.001       |

## Metrics
| Path                 | Metric           | HEAD~1   | workspace   | Change   |
|----------------------|------------------|----------|-------------|----------|
| dvclive/metrics.json | test.epoch.acc   | 0.36129  | 0.44875     | 0.08746  |
| dvclive/metrics.json | test.epoch.f1    | 0.26923  | 0.36818     | 0.09894  |
| dvclive/metrics.json | test.step.acc    | 0.36129  | 0.44875     | 0.08746  |
| dvclive/metrics.json | test.step.f1     | 0.26923  | 0.36818     | 0.09894  |
| dvclive/metrics.json | train.epoch.acc  | 0.82433  | 0.73694     | -0.08739 |
| dvclive/metrics.json | train.epoch.f1   | 0.76904  | 0.65581     | -0.11323 |
| dvclive/metrics.json | train.epoch.loss | 0.50873  | 0.74477     | 0.23604  |
| dvclive/metrics.json | train.step.acc   | 0.96875  | 0.9375      | -0.03125 |
| dvclive/metrics.json | train.step.f1    | 0.95789  | 0.89444     | -0.06345 |
| dvclive/metrics.json | train.step.loss  | 0.15193  | 0.29613     | 0.1442   |
| dvclive/metrics.json | val.epoch.acc    | 0.8311   | 0.82049     | -0.0106  |
| dvclive/metrics.json | val.epoch.f1     | 0.7995   | 0.79137     | -0.00813 |
| dvclive/metrics.json | val.epoch.loss   | 0.62941  | 0.55358     | -0.07583 |
| dvclive/metrics.json | val.step.acc     | 0.71429  | 0.85714     | 0.14286  |
| dvclive/metrics.json | val.step.f1      | 0.5      | 0.66667     | 0.16667  |
| dvclive/metrics.json | val.step.loss    | 0.98827  | 0.95916     | -0.02911 |

# Plots
TODO
