import sys
import yaml
from dill import load
import pandas as pd
import json
import torch
from sklearn.model_selection import train_test_split, KFold

if __name__ == "__main__":
    # Retrieve arguments from command line
    stage_name = sys.argv[1]  # The stage name argument
    input_filename = sys.argv[2]  # The input filename argument
    output_filename = sys.argv[3]  # The output filename argument

    # Load parameters from a YAML file
    params = yaml.safe_load(open("params.yaml"))[stage_name]
    split_type = params["type"]  # The type of split
    random_seed = params["seed"]  # The seed for random number generator

    # Load data from a dill file
    print("Reading dill file...")
    with open(input_filename, "rb") as fr:
        data = []  # Initialize list to hold dictionaries
        # Iterate over each key-value pair in the loaded data
        for key, segments in load(fr).items():
            for segment in segments:
                # Append dictionary to the list
                data.append(
                    {
                        "activity": key[0],
                        "person": key[1],
                        "hash": key[2],
                        "segment_id": segment["segment_id"][0],
                    }
                )

    # Convert list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    # If split type is 'segment', perform train test split on the DataFrame
    if split_type == "segment":
        split_ratio = params["test_ratio"]  # The ratio for splitting data into test set
        stratified = params["stratified"]  # Whether the split should be stratified

        print(
            f"Splitting {input_filename} into train and test set with {split_type} split and {split_ratio} test ratio"
        )

        train, test = train_test_split(
            df,
            test_size=split_ratio,
            random_state=random_seed,
            stratify=df[["activity"]] if stratified else None,
        )
    elif split_type in ["measurement", "person"]:
        split_ratio = params["test_ratio"]  # The ratio for splitting data into test set
        stratified = params["stratified"]  # Whether the split should be stratified

        print(
            f"Splitting {input_filename} into train and test set with {split_type} split and {split_ratio} test ratio"
        )

        group_by_columns = [
            "activity",
            ("hash" if split_type == "measurement" else "person"),
        ]
        df_grouped = (
            df.groupby(group_by_columns).agg({"segment_id": list}).reset_index()
        )
        train, test = train_test_split(
            df_grouped,
            test_size=split_ratio,
            random_state=random_seed,
            stratify=df_grouped[["activity"]] if stratified else None,
        )
        train, test = train.explode("segment_id"), test.explode("segment_id")
    elif split_type == "cross_validation":
        splits = params["cv_splits"]

        kfolds = KFold(n_splits=splits, shuffle=True, random_state=random_seed).split(
            df
        )

        # create a train-test split for each split
        for i in range(splits):
            train, test = next(kfolds)
            train, test = df.iloc[train], df.iloc[test]

            # Dump the train and test data into a JSON file
            print(f"Dumping split {i}...")
            with open(f"{output_filename}_{i + 1}.json", "w") as fw:
                json.dump(
                    {
                        "train": train["segment_id"].values.tolist(),
                        "test": test["segment_id"].values.tolist(),
                    },
                    fw,
                )

        print("Cross validation splits done.")

        # exit after creating all splits
        sys.exit(0)

    else:
        raise ValueError(f"Unknown split type {split_type}")

    print(
        f"Train test split done: {len(train)} train, {len(test)} test, test-ratio {len(test) / (len(train) + len(test))}"
    )

    # Dump the train and test data into a JSON file
    print("Dumping...")
    with open(output_filename, "w") as fw:
        json.dump(
            {
                "train": train["segment_id"].values.tolist(),
                "test": test["segment_id"].values.tolist(),
            },
            fw,
        )
