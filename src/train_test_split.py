import sys
import yaml
from dill import load
import pandas as pd
import json
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # get args
    stage_name = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    # get params
    params = yaml.safe_load(open("params.yaml"))[stage_name]
    split_type = params["type"]
    split_ratio = params["test_ratio"]
    random_seed = params["seed"]
    stratified = params["stratified"]

    print(
        f"Splitting {input_filename} into train and test set with {split_type} split and {split_ratio} test ratio"
    )

    # read dill file
    print("Reading dill file...")
    with open(input_filename, "rb") as fr:
        data = []  # list with dicts { activity, person, hash, segment_id }
        for key, segments in load(fr).items():
            for segment in segments:
                data.append(
                    {
                        "activity": key[0],
                        "person": key[1],
                        "hash": key[2],
                        "segment_id": segment["segment_id"][0],
                    }
                )

    df = pd.DataFrame(data)

    # split nach segment
    if split_type == "segment":
        train, test = train_test_split(
            df,
            test_size=split_ratio,
            random_state=random_seed,
            stratify=df[["activity"]] if stratified else None,
        )
    # split nach person oder measurement
    elif split_type in ["measurement", "person"]:
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
    # unbekannter split type
    else:
        raise ValueError(f"Unknown split type {split_type}")

    print(
        f"Train test split done: {len(train)} train, {len(test)} test, test-ratio {len(test) / (len(train) + len(test))}"
    )

    # dump, json file with list for train and test segment ids
    print("Dumping...")
    with open(output_filename, "w") as fw:
        json.dump(
            {
                "train": train["segment_id"].values.tolist(),
                "test": test["segment_id"].values.tolist(),
            },
            fw,
        )
