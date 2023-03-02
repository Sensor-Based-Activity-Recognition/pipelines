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

    def __init__(self, path, origin, sensors, pipeline, resample_freq_hz):
        """
        Args:
            path (str): Path to the dataset
            pipeline (str): Name of the pipeline to run
            resample_freq_hz (int): Resample frequency
            origin (str): Origin of the data
        """

        # wrangler
        self.path = path
        self.origin = origin
        self.sensors = sensors
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
        if self.origin == "SensorLogger":
            data = wrangler.SensorLoggerData(self.path, self.sensors).get_data()
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
