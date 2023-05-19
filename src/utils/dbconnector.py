""" QuestDB Connector """

# Base Modules
import io
import json
import requests
import yaml

# 3rd Party Modules
import polars as pl


class Database:
    """
    Class to connect to QuestDB and get data
    """

    def __init__(self, questdb_settings):
        self._questdb_settings = questdb_settings

    def get_data(self, query):
        """
        Gets data from a query

        Args:
            query (str): Query to run (must contain timestamp and hash columns)

        Returns:
            data (polars.DataFrame): Dataframe with the data
        """
        # prepare url with db config and query
        url = f"http://{self._questdb_settings['host']}:{self._questdb_settings['port']}/exp?query={query}"

        # get data
        request = requests.get(url, timeout=600)

        # read data into polars dataframe
        data = pl.read_csv(io.StringIO(request.text))

        # convert timestamp to datetime
        data = data.with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime, fmt="%Y-%m-%dT%H:%M:%S.%6fZ")
            .alias("timestamp")
        )

        # change columns with Float64 to Float32
        for col in data.columns:
            if data[col].dtype == pl.Float64:
                data = data.with_columns(pl.col(col).cast(pl.Float32).alias(col))

        # sort by insert hash and time
        data = data.sort(["hash", "timestamp"])

        # return data
        return data
