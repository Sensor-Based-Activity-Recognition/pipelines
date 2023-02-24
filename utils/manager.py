from . import wrangler
from . import pipelines


class PipelineRunner:
    """
    Class to run a pipeline on a dataset

    Attributes:
        path (str): Path to the dataset
        pipeline (str): Name of the pipeline to run
        resample_freq_hz (int): Resample frequency
        origin (str): Origin of the data
    """

    def __init__(self, path, pipeline, sample_freq_hz, origin):
        """
        Args:
            path (str): Path to the dataset
            pipeline (str): Name of the pipeline to run
            resample_freq_hz (int): Resample frequency
            origin (str): Origin of the data
        """

        self.path = path
        self.pipeline = pipeline
        self.resample_freq_hz = sample_freq_hz
        self.origin = origin

    def run(self):
        """
        Runs the pipeline on the data

        Returns:
            data (polars.DataFrame): Dataframe with the data

        Raises:
            ValueError: If the origin or pipeline is invalid
        """

        # get data
        if self.origin == "SensorLogger":
            data = wrangler.SensorLoggerData(self.path).get_data()
        elif self.origin == "App":
            data = wrangler.AppData(self.path).get_data()
        else:
            raise ValueError("Invalid origin")

        # run pipeline
        if self.pipeline == "Alpha":
            data = pipelines.Alpha(self.sample_freq_hz).run(data)
        elif self.pipeline == "Beta":
            data = pipelines.Beta(self.sample_freq_hz).run(data)
        elif self.pipeline != None:
            raise ValueError("Invalid pipeline")

        # return data
        return data
