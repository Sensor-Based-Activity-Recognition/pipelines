# TODO: Implement Pipelines (Data Cleaning, Data Resampling, Feature Extraction)
import polars as pl


class Alpha:
    """
    Class to run the Alpha Pipeline

    Attributes:
        resample_freq_hz (int): Resample frequency

    Methods:
        run: Runs the pipeline on the data
    """

    def __init__(self, resample_freq_hz):
        """
        Args:
            resample_freq_hz (int): Resample frequenc
        """

        self.resample_freq_hz = resample_freq_hz

    def run(self, data):
        """
        Runs the pipeline on the data

        Args:
            data (polars.DataFrame): Dataframe with the data

        Returns:
            data (polars.DataFrame): Dataframe with the data
        """

        return None


class Beta:
    """
    Class to run the Beta Pipeline

    Attributes:
        resample_freq_hz (int): Resample frequency

    Methods:
        run: Runs the pipeline on the data
    """

    def __init__(self, resample_freq_hz):
        """
        Args:
            resample_freq_hz (int): Resample frequency
        """

        self.resample_freq_hz = resample_freq_hz

    def run(self, data):
        """
        Runs the pipeline on the data

        Args:
            data (polars.DataFrame): Dataframe with the data

        Returns:
            data (polars.DataFrame): Dataframe with the data
        """

        return None
