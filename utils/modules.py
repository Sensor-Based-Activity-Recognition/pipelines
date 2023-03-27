""" Module Manager """

# Base Modules
from abc import ABCMeta, abstractmethod

# 3rd Party Modules
import polars as pl


class IModule(metaclass=ABCMeta):
    """Abstract class for modules"""

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def run():
        """Modules must be implemented in child classes!"""
        raise NotImplementedError


class Resampler(IModule):
    """Resampler"""

    def __init__(self, resample_freq_hz):
        super().__init__()
        self.resample_freq_hz = resample_freq_hz

    def run(self, data):
        """Resamples the data to the specified frequency"""

        raise NotImplementedError  # TODO: Implement Resampler


class FFT(IModule):
    """Fast Fourier Transform"""

    def run(self, data):
        """Calculates the FFT of the data"""
        raise NotImplementedError  # TODO: Implement FFT


# TODO: Implement other needed modules
