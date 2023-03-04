import io
import json
import requests

import polars as pl

# read config
with open("config.json") as f:
    config = json.load(f)
questdb_settings = config["questdb"]


class Database:
    def __init__(self):
        pass

    def get_data(self, query):
        """
        Gets data from a query

        Args:
            query (str): Query to run

        Returns:
            data (polars.DataFrame): Dataframe with the data
        """
        # prepare url with db config and query
        url = f"http://{questdb_settings['host']}:{questdb_settings['port']}/exp?query={query}"
        # get data
        r = requests.get(url)
        # read data into polars dataframe
        data = pl.read_csv(io.StringIO(r.text))
        # convert time to datetime 2023-02-24T17:31:54.190Z
        data = data.with_column(
            pl.col("time")
            .str.strptime(pl.Datetime, fmt="%Y-%m-%dT%H:%M:%S.%6fZ")
            .alias("time")
        )
        # sort by time
        data = data.sort(["filename", "time"])
        # return data
        return data
