import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.utils.DataLoaderSklearn import onehotencode

import sys


# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]


reverse_onehotencode = {v: k for k, v in onehotencode.items()}

# Load the datasets
# contains the two columns: pred_labels,true_labels
data = pd.read_csv(input_filename)

# map the labels to the correct names
data["pred_labels"] = data["pred_labels"].map(reverse_onehotencode)
data["true_labels"] = data["true_labels"].map(reverse_onehotencode)

# get the confusion matrix
confusion_matrix = pd.crosstab(
    data["true_labels"], data["pred_labels"], rownames=["True"], colnames=["Pred"]
)


# plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", ax=ax)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.tight_layout()
plt.savefig(output_filename)
