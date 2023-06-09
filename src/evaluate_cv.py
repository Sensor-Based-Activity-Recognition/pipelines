import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sys
import sklearn.metrics as metrics

from models.utils.DataLoaderSklearn import onehotencode


# get args
stage_name = sys.argv[1]
input_filenames = sys.argv[2]
output_filename = sys.argv[3]

# onehotencode
reverse_onehotencode = {v: k for k, v in onehotencode.items()}

# get all the files which match the pattern
files_path = glob.glob(input_filenames + "*.csv")
files = [pd.read_csv(file) for file in files_path]

# calculate the accuracy and f1 score for each file
metrics_ = {}
for i, file in enumerate(files):
    name = f"split_{i}"

    # get the true and predicted labels
    y_true = file["true_labels"].values
    y_pred = file["pred_labels"].values

    # calculate the accuracy and f1 score
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average="macro")

    # save the results
    metrics_[name] = {"accuracy": accuracy, "f1_macro_score": f1_score}

# create a dataframe with the results
metrics_df = pd.DataFrame(metrics_).T
metrics_df["accuracy"] = metrics_df["accuracy"].astype(float)
metrics_df["f1_macro_score"] = metrics_df["f1_macro_score"].astype(float)

# plot the results as boxplot and display mean and std
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
sns.boxplot(
    data=metrics_df,
    ax=ax,
    palette="Set2",
    linewidth=1.5,
    width=0.5,
    fliersize=3,
    whis=1.5,
)
ax.set_title(f"Accuracy and F1 Macro Score Distribution for {len(files)}-fold CV")
ax.set_ylabel("Score distribution")
ax.set_xlabel("Metric")
ax.text(
    0.5,
    0.84,
    f"Mean Accuracy: {metrics_df['accuracy'].mean():.4f} +/- {metrics_df['accuracy'].std():.4f}",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax.transAxes,
    fontsize=8,
)
ax.text(
    0.5,
    0.8,
    f"Mean F1 Macro Score: {metrics_df['f1_macro_score'].mean():.4f} +/- {metrics_df['f1_macro_score'].std():.4f}",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax.transAxes,
    fontsize=8,
)
plt.tight_layout()
plt.ylim(0.8, 1)
plt.xticks(ticks=[0, 1], labels=["Accuracy", "F1 Macro Score"])
plt.savefig(output_filename)
