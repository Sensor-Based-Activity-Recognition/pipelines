import yaml
import sys
from dill import load, dump
import pandas as pd
from typing import List
from utils import transformers

# get command line arguments
stage_name = sys.argv[1]
input_filename_segments = sys.argv[2]
input_filename_traintest_split_file = sys.argv[3]
output_filename_transformed_segments = sys.argv[4]
output_filename_transformation = sys.argv[5]


def get_structure_dict(segments: dict):
    """
    Get structure of segments (segmentid)

    Args:
        segments (dict): A dictionary containing segments

    Returns:
        struct_dict (dict): A dictionary containing the structure of the segments
    """

    struct_dict = {}  # structure is stored here
    for key, segments in segments.items():
        segments: List[pd.DataFrame]

        segment_ids = []
        for segment in segments:
            segment_ids.append(segment.iloc[0, :]["segment_id"])

        struct_dict[key] = segment_ids

    return struct_dict


def combine_df(segments: dict, train_ids: list, test_ids: list):
    """
    Combine DataFrame.

    Args:
        segments (dict): A dictionary containing segments
        train_ids (list): List containing ids for training set
        test_ids (list): List containing ids for testing set

    Returns:
        tuple: train_df (DataFrame) and test_df (DataFrame)
    """

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

    train_df = pd.concat(train_dfs_list, axis=0, ignore_index=False)
    train_df = train_df.astype(
        train_dfs_list[0].dtypes.apply(lambda x: x.name).to_dict()
    )

    test_df = pd.concat(test_dfs_list, axis=0, ignore_index=False)
    test_df = test_df.astype(test_dfs_list[0].dtypes.apply(lambda x: x.name).to_dict())

    return train_df, test_df  # convert to dataframe and return


def reconstruct_segments_dict(
    dict_structure: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ids: list,
    test_ids: list,
):
    """
    Reconstruct segments dictionary.

    Args:
        dict_structure (dict): A dictionary containing structure of segments
        train_df (DataFrame): DataFrame containing training data
        test_df (DataFrame): DataFrame containing testing data
        train_ids (list): List containing ids for training set
        test_ids (list): List containing ids for testing set

    Returns:
        segments (dict): Reconstructed dictionary of segments
    """
    
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
    """
    Load necessary files

    Returns:
        tuple: segments (dict), df_train_test_split (DataFrame)
    """

    with open(input_filename_segments, "rb") as fr:  # load data
        segments: dict = load(fr)

    df_train_test_split = pd.read_json(
        input_filename_traintest_split_file, typ="series"
    )

    return segments, df_train_test_split


def write_files(transformed_segments: dict, transformer: object):
    """
    Write transformed segments and transformer to files.

    Args:
        transformed_segments (dict): A dictionary containing transformed segments
        transformer (object): An object of transformer
    """

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

# Perform the appropriate transformation based on the transformation_type
match transformation_type:
    case "min-max":
        # retrieve the feature range for the transformation
        feature_range_min = transformation_params["feature_range_min"]
        feature_range_max = transformation_params["feature_range_max"]

        print(
            f"Applying min-max scaler with feature_range_min:{feature_range_min} and feature_range_max:{feature_range_max}"
        )

        # read segment dictionary and train_test_split segment_ids
        segments, train_test_split = read_files()

        # get structure of segments dict
        structure = get_structure_dict(segments)

        # split segments into two separate dataframes containing train and test data
        train_df, test_df = combine_df(segments, train_test_split["train"], train_test_split["test"])

        # initialize the Min-Max scaler with the defined feature range
        scaler = transformers.Transformer_Min_Max(feature_range_min, feature_range_max)

        # fit and transform the train and test datasets
        train_df_transformed = scaler.fit_transform(train_df)
        test_df_transformed = scaler.transform(test_df)

        # reconstruct the segments dictionary
        segments_transformed = reconstruct_segments_dict(structure, train_df_transformed, test_df_transformed, train_test_split["train"], train_test_split["test"])

        # write the transformed segments and the scaler to files
        write_files(segments_transformed, scaler)

    case "standardize":
        print("Applying standardization")
        
        # read segment dictionary and train_test_split segment_ids
        segments, train_test_split = read_files()
        
        # get structure of segments dict
        structure = get_structure_dict(segments)
        
        # split segments into two separate dataframes containing train and test data
        train_df, test_df = combine_df(segments, train_test_split["train"], train_test_split["test"])

        # initialize the Standard Scaler
        scaler = transformers.Transformer_Standardize()

        # fit and transform the train and test datasets
        train_df_transformed = scaler.fit_transform(train_df)
        test_df_transformed = scaler.transform(test_df)

        # reconstruct the segments dictionary
        segments_transformed = reconstruct_segments_dict(structure, train_df_transformed, test_df_transformed, train_test_split["train"], train_test_split["test"])

        # write the transformed segments and the scaler to files
        write_files(segments_transformed, scaler)

    case "pca":
        # retrieve the number of components for the PCA transformation
        n_components = params["transformation_params"]["n_components"]

        print(f"Applying pca with n_components:{n_components}")

        # read segment dictionary and train_test_split segment_ids
        segments, train_test_split = read_files()
        
        # get structure of segments dict
        structure = get_structure_dict(segments)
        
        # split segments into two separate dataframes containing train and test data
        train_df, test_df = combine_df(segments, train_test_split["train"], train_test_split["test"])

        # initialize the PCA transformer with the defined number of components
        scaler = transformers.Transformer_PCA(n_components)

        # fit and transform the train and test datasets
        train_df_transformed = scaler.fit_transform(train_df)
        test_df_transformed = scaler.transform(test_df)

        # reconstruct the segments dictionary
        segments_transformed = reconstruct_segments_dict(structure, train_df_transformed, test_df_transformed, train_test_split["train"], train_test_split["test"])

        # write the transformed segments and the scaler to files
        write_files(segments_transformed, scaler)

    case _:
        raise Exception(f"unknown transformation: '{transformation_type}'")
