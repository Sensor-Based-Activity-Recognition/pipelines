import yaml
import sys
from dill import load, dump
import pandas as pd
from typing import List
from utils import transformers

# get args
stage_name = sys.argv[1]
input_filename_segments = sys.argv[2]
input_filename_traintest_split_file = sys.argv[3]
output_filename_transformed_segments = sys.argv[4]
output_filename_transformation = sys.argv[5]


def get_structure_dict(segments: dict):
    """Get structure of segments (segmentid)"""

    struct_dict = {}  # structure is stored here
    for key, segments in segments.items():
        segments: List[pd.DataFrame]

        segment_ids = []
        for segment in segments:
            segment_ids.append(segment.iloc[0, :]["segment_id"])

        struct_dict[key] = segment_ids

    return struct_dict


def combine_df(segments: dict, train_ids: list, test_ids: list):
    train_dfs_list = []
    test_dfs_list = []
    ids_passed = []
    for _, segments in segments.items():
        segments: List[pd.DataFrame]

        for segment in segments:
            segment_id = segment.iloc[0, :]["segment_id"]

            # sanity check (segment_id must be unique)
            if segment_id not in ids_passed:
                ids_passed.append(segment_id)

            else:
                raise Exception("duplicated segment_id not gud diese")

            # decide if segment is added to train or test dataframe
            if segment_id in train_ids:
                train_dfs_list.append(segment)

            elif segment_id in test_ids:
                test_dfs_list.append(segment)

            else:
                raise Exception("segment_id not found in train_ids or test_ids")

    train_df = pd.concat(train_dfs_list, axis=0, ignore_index=True)
    train_df = train_df.astype(
        train_dfs_list[0].dtypes.apply(lambda x: x.name).to_dict()
    )

    test_df = pd.concat(test_dfs_list, axis=0, ignore_index=True)
    test_df = test_df.astype(test_dfs_list[0].dtypes.apply(lambda x: x.name).to_dict())

    return train_df, test_df  # convert to dataframe and return


def reconstruct_segments_dict(
    dict_structure: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ids: list,
    test_ids: list,
):
    segments = {}
    for key, ids in dict_structure.items():
        df_list = []
        for id in ids:
            if id in train_ids:
                df_list.append(train_df[train_df.segment_id == id])  # append dataframe

            elif id in test_ids:
                df_list.append(test_df[test_df.segment_id == id])  # append dataframe

            else:
                raise Exception("id found which is not in train or test split")

        segments[key] = df_list

    return segments


def read_files():
    """loads necessary files"""

    with open(input_filename_segments, "rb") as fr:  # load data
        segments: dict = load(fr)

    df_train_test_split = pd.read_json(
        input_filename_traintest_split_file, typ="series"
    )

    return segments, df_train_test_split


def write_files(transformed_segments: dict, transformer: object):
    # dump transformed segments
    with open(output_filename_transformed_segments, "wb") as f:
        dump(transformed_segments, f)

    # dump transformer
    with open(output_filename_transformation, "wb") as f:
        dump(transformer, f)


# get params
params: dict = yaml.safe_load(open("params.yaml"))[stage_name]
transformation_type = params["transformation"]
transformation_params = params.get("transformation_params", None)

match transformation_type:
    case "min-max":
        feature_range_min = transformation_params["feature_range_min"]
        feature_range_max = transformation_params["feature_range_max"]

        print(
            f"Applying min-max scaler with feature_range_min:{feature_range_min} and feature_range_max:{feature_range_max}"
        )

        (
            segments,
            train_test_split,
        ) = read_files()  # read segment dictionary and train_test_split segment_ids

        structure = get_structure_dict(segments)  # get structure of segments dict

        train_df, test_df = combine_df(
            segments, train_test_split["train"], train_test_split["test"]
        )  # split segments into two separate dataframes containing train and test data

        scaler = transformers.Transformer_Min_Max(feature_range_min, feature_range_max)

        train_df_transformed = scaler.fit_transform(train_df)
        test_df_transformed = scaler.transform(test_df)

        segments_transformed = reconstruct_segments_dict(
            structure,
            train_df_transformed,
            test_df_transformed,
            train_test_split["train"],
            train_test_split["test"],
        )  # store transformed segments
        # write files
        write_files(segments_transformed, scaler)

    case "standardize":
        print("Applying standardization")
        # read segment dictionary and train_test_split segment_ids
        (
            segments,
            train_test_split,
        ) = read_files()
        # get structure of segments dict
        structure = get_structure_dict(segments)
        # split segments into two separate dataframes containing train and test data
        train_df, test_df = combine_df(
            segments, train_test_split["train"], train_test_split["test"]
        )

        scaler = transformers.Transformer_Standardize()

        train_df_transformed = scaler.fit_transform(train_df)
        test_df_transformed = scaler.transform(test_df)

        segments_transformed = reconstruct_segments_dict(
            structure,
            train_df_transformed,
            test_df_transformed,
            train_test_split["train"],
            train_test_split["test"],
        )  # store transformed segments

        write_files(segments_transformed, scaler)  # write files

    case "pca":
        n_components = params["transformation_params"]["n_components"]

        print(f"Applying pca with n_components:{n_components}")

        # read segment dictionary and train_test_split segment_ids
        segments, train_test_split = read_files()
        # get structure of segments dict
        structure = get_structure_dict(segments)
        # split segments into two separate dataframes containing train and test data
        train_df, test_df = combine_df(
            segments, train_test_split["train"], train_test_split["test"]
        )

        scaler = transformers.Transformer_PCA(n_components)

        train_df_transformed = scaler.fit_transform(train_df)
        test_df_transformed = scaler.transform(test_df)
        # store transformed segments
        segments_transformed = reconstruct_segments_dict(
            structure,
            train_df_transformed,
            test_df_transformed,
            train_test_split["train"],
            train_test_split["test"],
        )
        # write files
        write_files(segments_transformed, scaler)

    case _:
        raise Exception(f"unknown transformation: '{transformation_type}'")
