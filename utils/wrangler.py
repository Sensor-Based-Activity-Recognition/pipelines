import glob
import polars as pl
import numpy as np

from tqdm.notebook import tqdm


class SensorLoggerData:
    """
    Class to get data from SensorLogger App

    Attributes:
        path (str): Path to the dataset

    Methods:
        get_data: Returns the data as a polars DataFrame
    """

    def __init__(self, path, sensors):
        """
        Args:
            path (str): Path to the dataset
        """

        self.path = path
        self.sensors = sensors

    def get_data(self):
        """
        Returns the data as a polars DataFrame

        Returns:
            data (polars.DataFrame): Dataframe with the data
        """

        # get all json files in subdirectories
        files = glob.glob(f"{self.path}/**/*.json", recursive=True)

        # sort files
        files.sort()

        # init data
        data = None

        # for each file, read the json and append to data
        for file in tqdm(
            files,
            desc="Reading Files",
            unit="files",
        ):
            # error handling
            try:
                # read json + lazy
                temp = pl.read_json(file).lazy()

                # melt data grouped by time and sensor
                temp = temp.melt(id_vars=["time", "sensor"])

                # rename variable to sensor_variable
                temp = temp.with_columns(
                    (pl.col("sensor") + "_" + pl.col("variable")).alias("variable")
                )

                # drop sensor column
                temp = temp.drop(["sensor"])

                # time to int
                temp = temp.with_columns(pl.col("time").cast(pl.Int64).alias("time"))

                # change resolution of data to 10ms
                temp = temp.with_columns(
                    (pl.col("time") // 10000000 * 10).cast(pl.Int64).alias("time")
                )

                # change time measurements from nanoseconds to milliseconds
                temp = temp.with_columns(pl.from_epoch("time", unit="ms").alias("time"))

                # convert values to float if possible
                temp = temp.with_columns(
                    pl.col("value").cast(pl.Float32, strict=False).alias("value")
                )

                # filter sensors
                temp = temp.filter(pl.col("variable").is_in(self.sensors))

                # collect 1
                temp = temp.collect()

                # pivot the data + lazy
                temp = temp.pivot(
                    index="time",
                    columns="variable",
                    values="value",
                    aggregate_function="mean",
                    sort_columns=True,
                ).lazy()

                # add file name to columns as id
                temp = temp.with_columns(
                    pl.lit(file.split("/")[-1].split(".")[0]).alias("id")
                )

                # add person name to columns as person
                temp = temp.with_columns(pl.lit(file.split("/")[-2]).alias("person"))

                # add activity name to columns as activity
                temp = temp.with_columns(pl.lit(file.split("/")[-3]).alias("activity"))

                # collect 2
                temp = temp.collect()

                # append to data
                ## if data is empty, set data to temp
                if type(data) == type(None):
                    data = temp
                ## else, concat data and temp
                else:
                    data = pl.concat([data, temp], how="diagonal")

            except Exception as e:
                print(f"Error processing {file}: {e}")

        # return data
        return data


class AppData:
    """
    Class to get data from our own App

    Attributes:
        path (str): Path to the dataset

    Methods:
        get_data: Returns the data as a polars DataFrame
    """

    # TODO: Implement Data Getter for App (first implement app xD)
    def __init__(self, path):
        """
        Args:
            path (str): Path to the dataset
        """

        self.path = path

    def get_data(self):
        """
        Returns the data as a polars DataFrame

        Returns:
            data (polars.DataFrame): Dataframe with the data
        """

        return None
