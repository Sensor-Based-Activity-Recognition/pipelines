# random train test split on segment, measurement and person basis
# specify type, ratio and random seed as hyperparameter in params.yaml
import sys
import yaml
from dill import load
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # get args
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    # get params
    params = yaml.safe_load(open("params.yaml"))["train_test_split"]
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
        data = [] # list with dicts { activity, person, hash, segment_id }
        for key, segments in load(fr).items():
            for segment in segments:
                data.append({
                    "activity": key[0],
                    "person": key[1],
                    "hash": key[2],
                    "segment_id": segment["segment_id"][0]
                })
        
        df = pd.DataFrame(data)

        print(df.shape)
        print(df.sample(5))

        # split
        if split_type == "segment":
            # split on segment basis
            train, test = train_test_split(df,
                                           test_size=split_ratio,
                                           random_state=random_seed,
                                           stratify=df[["activity"]] if stratified else None)
        elif split_type == "measurement":
            # split on measurement/hash basis
            # group by activity and hash convert segment_id to list
            df_grouped_by_hash = df.groupby(["activity", "hash"]).agg({"segment_id": lambda x: list(x)})
            # reset index to get activity and hash as columns
            df_grouped_by_hash = df_grouped_by_hash.reset_index()
            print(df_grouped_by_hash.shape)
            print(df_grouped_by_hash.sample(5))
            train, test = train_test_split(df_grouped_by_hash,
                                           test_size=split_ratio,
                                           random_state=random_seed,
                                           stratify=df_grouped_by_hash[["activity"]] if stratified else None)
            # convert segment_id list to dataframe
            train = train.explode("segment_id")
            test = test.explode("segment_id")
        elif split_type == "person":
            # split on person basis
            df_grouped_by_person = df.groupby(["activity", "person"]).agg({"segment_id": lambda x: list(x)})
            df_grouped_by_person = df_grouped_by_person.reset_index()
            print(df_grouped_by_person.shape)
            print(df_grouped_by_person.sample(5))
            train, test = train_test_split(df_grouped_by_person,
                                             test_size=split_ratio,
                                                random_state=random_seed,
                                                stratify=df_grouped_by_person[["activity"]] if stratified else None)
            train = train.explode("segment_id")
            test = test.explode("segment_id")
        else:
            raise ValueError(f"Unknown split type {split_type}")
        
        print(train.shape)
        print(test.shape)

        print(train.sample(5))
        print(test.sample(5))

        # dump, json file with list for train and test segment ids
        print("Dumping...")
        with open(output_filename, "w") as fw:
            json.dump({
                "train": train["segment_id"].values.tolist(),
                "test": test["segment_id"].values.tolist()
            }, fw)