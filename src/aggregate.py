import pandas as pd
import sys
import yaml
import time
from dill import load, dump
from tqdm import tqdm


# Hilfsfunktion
def aggregate(df: pd.DataFrame, aggregation_functions: list) -> pd.DataFrame:
    """
    Aggregiert Daten mithilfe von Aggregationsfunktionen.

    Args:
        df (pd.DataFrame): DataFrame, das aggregiert werden soll.
        aggregation_functions (list): Liste von Aggregationsfunktionen, die an pandas.DataFrame.agg übergeben werden.

    Returns:
        pd.DataFrame: Aggregiertes DataFrame.
    """
    if len(aggregation_functions) == 0:
        raise ValueError("No aggregation functions provided")

    numeric_columns = df.select_dtypes(include=["number"]).columns  # Ermittelt die numerischen Spalten.
    df_agg = df[numeric_columns].agg(aggregation_functions)  # Führt die Aggregation auf den numerischen Spalten durch.

    # Fügt die ersten Werte der nicht numerischen Spalten hinzu.
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns
    df_agg[non_numeric_columns] = df[non_numeric_columns].iloc[0]

    return df_agg


if __name__ == "__main__":
    # Ermittelt die Argumente.
    stage_name = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    # Ermittelt die Parameter.
    aggregation_functions = yaml.safe_load(open("params.yaml"))[stage_name]

    print(
        f"Aggregating data from {input_filename} to {output_filename} with aggregation functions {aggregation_functions}"
    )

    # Führt die Aggregation durch.
    with open(input_filename, "rb") as fr:
        data = {}
        for key, segments in tqdm(load(fr).items()):
            data[key] = [
                aggregate(segment, aggregation_functions) for segment in segments
            ]

        with open(output_filename, "wb") as fw:
            dump(data, fw)

time.sleep(1)  # Pausiert das Skript für eine Sekunde.
