import glob
import polars as pl
import numpy as np


class SensorLoggerData:
    """
    Class to get data from SensorLogger App

    Attributes:
        path (str): Path to the dataset

    Methods:
        get_data: Returns the data as a polars DataFrame
    """

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

        # get all json files in subdirectories
        files = glob.glob(f"{self.path}/**/*.json", recursive=True)

        # sort files
        files.sort()

        # init data
        data = None

        # for each file, read the json and append to data
        for file in files:
            # error handling
            try:
                # read json
                temp = pl.read_json(file)

                # melt data grouped by time and sensor
                temp = temp.fill_null(value=np.nan)

                # melt data grouped by time and sensor
                temp = temp.melt(id_vars=["time", "sensor"])

                # drop nan values
                temp = temp.drop_nulls()

                # rename variable to sensor_variable
                temp = temp.with_columns(
                    (pl.col("sensor") + "_" + pl.col("variable")).alias("variable")
                )

                # drop sensor column
                temp = temp.drop(["sensor"])

                # convert time to datetime
                temp = temp.with_columns(pl.from_epoch("time", unit="ns").alias("time"))

                # change time measurements from nanoseconds to milliseconds
                temp = temp.with_columns(pl.from_epoch("time", unit="ms").alias("time"))

                # convert values to float if possible
                temp = temp.with_columns(
                    pl.col("value").cast(pl.Float64, strict=False).alias("value")
                )

                # pivot on time
                temp = temp.pivot(
                    index="time",
                    columns="variable",
                    values="value",
                )

                # add file name to columns as id
                temp = temp.with_columns(
                    pl.lit(file.split("/")[-1].split(".")[0]).alias("id")
                )

                # add person name to columns as person
                temp = temp.with_columns(pl.lit(file.split("/")[-2]).alias("person"))

                # add activity name to columns as activity
                temp = temp.with_columns(pl.lit(file.split("/")[-3]).alias("activity"))

                # TODO: Implement Feature Selection

                # append to data
                ## if data is empty, set data to temp
                if type(data) == type(None):
                    data = temp
                ## else, concat data and temp
                else:
                    data = pl.concat([data, temp], how="diagonal")

            # if error, print info
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
