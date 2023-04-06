import pandas as pd
import sys
import yaml
import time
from dill import load, dump
from tqdm import tqdm


# helper function
def aggregate(df: pd.DataFrame, aggregation_functions: list) -> pd.DataFrame:
    """Aggregate data using aggregation functions

    Args:
        df (pd.DataFrame): Dataframe to aggregate
        aggregation_functions (list): List of aggregation functions passed to pandas.DataFrame.agg

    Returns:
        pd.DataFrame: Aggregated dataframe
    """
    if len(aggregation_functions) == 0:
        raise ValueError("No aggregation functions provided")
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df_agg = df[numeric_columns].agg(aggregation_functions)
    # join first values of non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns
    df_agg[non_numeric_columns] = df[non_numeric_columns].iloc[0]
    return df_agg


if __name__ == "__main__":
    # get args
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    # get config
    params = yaml.safe_load(open("params.yaml"))
    aggregation_functions = params["aggregate"]

    print(
        f"Aggregating data from {input_filename} to {output_filename} with aggregation functions {aggregation_functions}"
    )

    # execute aggregation
    with open(input_filename, "rb") as fr:
        data = {}
        for key, segments in tqdm(load(fr).items()):
            data[key] = [aggregate(segment, aggregation_functions) for segment in segments]

        with open(output_filename, "wb") as fw:
            dump(data, fw)
            
time.sleep(1)
