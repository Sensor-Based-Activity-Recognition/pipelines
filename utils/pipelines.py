""" Pipelines """

# Base Modules
import sys
from abc import ABCMeta, abstractmethod

# Internal Modules
from .modules import Resampler, FFT


def get_pipeline(pipeline_name):
    """Returns the pipeline class based on the pipeline name"""
    return getattr(sys.modules[__name__], pipeline_name)


class IPipeline(metaclass=ABCMeta):
    """Abstract class for pipelines"""

    def __init__(self, resample_freq_hz):
        self.resample_freq_hz = resample_freq_hz

    @staticmethod
    @abstractmethod
    def run():
        """Pipelines must be implemented in child classes!"""
        raise NotImplementedError


class Alpha(IPipeline):
    """Alpha Pipeline"""

    def run(self, data):
        # TODO: Implement Alpha Pipeline using modules.py

        # Example:
        # data = Resampler(self.resample_freq_hz).run(data)
        # data = FFT().run(data)
        # return data

        raise NotImplementedError


class Beta(IPipeline):
    """Beta Pipeline"""

    def run(self, data):
        # TODO: Implement Beta Pipeline using modules.py

        raise NotImplementedError
