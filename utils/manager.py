import json

from . import dbconnector
from . import pipelines

# get config
with open("config.json") as f:
    config = json.load(f)
table_name = "dev" if config["dev"] else "prod"


class PipelineRunner:
    """
    Class to run a pipeline on a dataset

    Attributes:
        path (str): Path to the dataset
        pipeline (str): Name of the pipeline to run
        resample_freq_hz (int): Resample frequency
        origin (str): Origin of the data
    """

    def __init__(self, pipeline, resample_freq_hz):
        """
        Args:
            path (str): Path to the dataset
            pipeline (str): Name of the pipeline to run
            resample_freq_hz (int): Resample frequency
            origin (str): Origin of the data
        """
        # pipeline
        self.pipeline = pipeline
        self.resample_freq_hz = resample_freq_hz

    def run(self):
        """
        Runs the pipeline on the data

        Returns:
            data (polars.DataFrame): Dataframe with the data

        Raises:
            ValueError: If the origin or pipeline is invalid
        """

        # get data
        data = dbconnector.Database().get_data(f"SELECT * FROM {table_name}")

        # run pipeline
        if self.pipeline == "Alpha":
            data = pipelines.Alpha(self.sample_freq_hz).run(data)
        elif self.pipeline == "Beta":
            data = pipelines.Beta(self.sample_freq_hz).run(data)
        elif self.pipeline != None:
            raise ValueError("Invalid pipeline")

        # return data
        return data
