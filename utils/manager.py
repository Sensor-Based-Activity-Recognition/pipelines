""" Pipeline manager """

# Base Modules
import json
from urllib.parse import quote

# Internal Modules
from . import dbconnector
from . import pipelines

# get config
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)
TABLENAME = "dev" if config["dev"] else "prod"


class PipelineRunner:
    """
    Class to run a pipeline on a dataset

    Attributes:
        pipeline (str): Name of the pipeline to run
        resample_freq_hz (int): Resample frequency
        calibrated_data (bool): If should use calibrated data
    """

    def __init__(self, pipeline, resample_freq_hz, calibrated_data):
        """
        Args:
            pipeline (str): Name of the pipeline to run
            resample_freq_hz (int): Resample frequency
            calibrated_data (bool): If should use calibrated data
        """
        # pipeline
        self.pipeline = pipeline
        self.resample_freq_hz = resample_freq_hz
        self.calibrated_data = calibrated_data

    def run(self):
        """
        Runs the pipeline on the data

        Returns:
            data (polars.DataFrame): Dataframe with the data

        Raises:
            ValueError: If the pipeline is invalid
        """

        # create query
        if self.calibrated_data:
            query = f"""
            SELECT  
                timestamp,
                Accelerometer_x,
                Accelerometer_y,
                Accelerometer_z,
                Gyroscope_x,
                Gyroscope_y,
                Gyroscope_z,
                Magnetometer_x,
                Magnetometer_y,
                Magnetometer_z,
                activity,
                hash
            FROM 
                {TABLENAME};
            """

        else:
            query = f"""
            SELECT
                timestamp,
                AccelerometerUncalibrated_x,
                AccelerometerUncalibrated_y,
                AccelerometerUncalibrated_z,
                GyroscopeUncalibrated_x,
                GyroscopeUncalibrated_y,
                GyroscopeUncalibrated_z,
                MagnetometerUncalibrated_x,
                MagnetometerUncalibrated_y,
                MagnetometerUncalibrated_z,
                activity,
                hash
            FROM
                {TABLENAME};
            """

        # convert query to url format
        query = quote(query)

        # get data
        data = dbconnector.Database().get_data(query)

        # check if pipeline is skipped
        if not self.pipeline:
            return data

        # select pipeline
        pipeline = pipelines.get_pipeline(self.pipeline)

        # run and return pipeline
        return pipeline(self.resample_freq_hz).run(data)
