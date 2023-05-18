# Standard Libraries
import sys
import yaml
import pickle
from argparse import Namespace

# Internal Libraries
from models.MLP import MLP
from models.utils.DataLoaderTabular import DataModuleTabular
from models.utils.DataLoaderSklearn import (
    DataLoaderSklearn_Tabular,
    DataLoaderSklearn_Segments,
)

# 3rd Party Libraries
import torch
import pandas as pd
from pytorch_lightning import Trainer
from dvclive.lightning import DVCLiveLogger
from dvclive import Live
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


# Helper functions
def get_model(model_name, config):
    pytorch_models = {
        "MLP": MLP,
    }

    sklearn_models = {
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    }

    if model_name in pytorch_models:
        return "pytorch", pytorch_models[model_name](config)
    elif model_name in sklearn_models:
        return "sklearn", sklearn_models[model_name](**config.model_hparams)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


# get args
stagename = sys.argv[1]
input_filename_data = sys.argv[2]
input_filename_train_test_split = sys.argv[3]
output_model = sys.argv[4]
output_prediction = sys.argv[5]

config = Namespace(**yaml.safe_load(open("params.yaml"))[stagename])

# Define model
model_type, model = get_model(config.model, config)

# Define datamodule
if config.data["type"] == "Tabular":
    datamodule = DataModuleTabular(
        config, input_filename_data, input_filename_train_test_split
    )
elif config.data["type"] == "Sklearn_Tabular":
    datamodule = DataLoaderSklearn_Tabular(
        config.data.get("params"), input_filename_data, input_filename_train_test_split
    )
elif config.data["type"] == "Sklearn_Segments":
    datamodule = DataLoaderSklearn_Segments(
        config.data.get("params"), input_filename_data, input_filename_train_test_split
    )
else:
    raise NotImplementedError(f"Datamodule {config.type} not implemented")

if model_type == "pytorch":
    # Define trainer
    trainer = Trainer(
        accelerator="auto",
        logger=DVCLiveLogger(),
        max_epochs=config.model_hparams["num_epochs"],
        enable_progress_bar=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )

    # Train model
    trainer.fit(model, datamodule)

    # Test model
    trainer.test(model, datamodule)

    # Save predicted and true labels
    with torch.no_grad():
        model.eval()
        test_data = datamodule.test_data.data
        labels = datamodule.test_data.labels
        pred_labels = torch.argmax(model(test_data), dim=1)
        pd.DataFrame(
            {
                "pred_labels": pred_labels,
                "true_labels": labels,
            }
        ).to_csv(output_prediction, index=False)

    # Save model
    trainer.save_checkpoint(output_model)
elif model_type == "sklearn":
    # Train model
    X_train, y_train = datamodule.train_data, datamodule.train_labels
    X_test, y_test = datamodule.test_data, datamodule.test_labels

    model.fit(X_train, y_train)

    def calc_metrics(model, X, y):
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        return acc, f1

    # Model metrics on train & test set
    train_acc, train_f1 = calc_metrics(model, X_train, y_train)
    test_acc, test_f1 = calc_metrics(model, X_test, y_test)

    # Log metrics with DVC
    dvclive = Live()
    dvclive.log_metric("train.epoch.acc", train_acc)
    dvclive.log_metric("train.epoch.f1", train_f1)
    dvclive.log_metric("test.epoch.acc", test_acc)
    dvclive.log_metric("test.epoch.f1", test_f1)
    dvclive.end()

    # Predict on test set
    y_pred = model.predict(X_test)

    # Save predicted and true labels
    pd.DataFrame(
        {
            "pred_labels": y_pred,
            "true_labels": y_test,
        }
    ).to_csv(output_prediction, index=False)

    # Save model
    with open(output_model, "wb") as f:
        pickle.dump(model, f)
