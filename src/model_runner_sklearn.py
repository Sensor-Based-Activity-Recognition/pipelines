# Standard Libraries
import sys
import yaml
import pickle
from argparse import Namespace

# Internal Libraries
from models.utils.DataLoaderSklearn import DataLoaderSklearn

# 3rd Party Libraries
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from dvclive import Live

# get args
stagename = sys.argv[1]
input_filename_data = sys.argv[2]
input_filename_train_test_split = sys.argv[3]
output_model = sys.argv[4]
output_prediction = sys.argv[5]

hparams = Namespace(**yaml.safe_load(open("params.yaml"))[stagename])

# Define model
if hparams.model == "HistGradientBoostingClassifier":
    model = HistGradientBoostingClassifier(**hparams.model_hparams)
else:
    raise NotImplementedError(f"Model {hparams.model} not implemented")

# Define datamodule
if hparams.data["type"] == "Sklearn":
    datamodule = DataLoaderSklearn(
        hparams, input_filename_data, input_filename_train_test_split
    )
else:
    raise NotImplementedError(f"Datamodule {hparams.type} not implemented")

# Train model
X_train, y_train = datamodule.train_data, datamodule.train_labels
X_test, y_test = datamodule.test_data, datamodule.test_labels

model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Log metrics with DVC
dvclive = Live(dir="hist_gradient_boost")
dvclive.log_metric("accuracy", accuracy)
dvclive.next_step()

# Save predicted and true labels
pd.DataFrame(
    {
        "pred_labels": y_pred,
        "true_labels": y_test,
    }
).to_csv(output_prediction, index=False)

# Save model
with open(output_model, 'wb') as f:
    pickle.dump(model, f)
