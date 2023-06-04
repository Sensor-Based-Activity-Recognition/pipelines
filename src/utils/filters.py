from argparse import ArgumentError
from scipy.signal import butter, lfilter
import pandas as pd
import numpy as np
from typing import Tuple, Union
import numbers


def butterworth_filter(
    df: pd.DataFrame,
    filter_type: str,
    cutoff_freq: Union[float, Tuple[float, float]],
    order: int = 3,
):
    """
    Anwendung eines Butterworth-Filters auf ein DataFrame.

    Args:
        df (pd.DataFrame): Das DataFrame, auf das der Filter angewendet wird.
        filter_type (str): Der Filtertyp ('bandpass', 'lowpass', 'highpass').
        cutoff_freq (float | [float, float]): Cutoff-Frequenz(en) für den Filter (für Bandpass: [lowcut, highcut]).
        order (int, optional): Die Ordnung des Filters. Default ist 3.

    Returns:
        pd.DataFrame: Ein neues DataFrame mit gefilterten Spalten. Nicht-numerische Spalten werden gelöscht.

    Raises:
        ArgumentError: Wenn ein unbekannter filter_type übergeben wird.
    """
    # Bestätigen, dass die Frequenz konstant ist
    assert (np.diff(df.index.values) == np.diff(df.index.values)[0]).all()

    # Sampling-Frequenz berechnen
    fs = pd.Timedelta(1, "s") / np.diff(df.index.values)[0]
    # Nyquist-Frequenz berechnen
    nyq = 0.5 * fs

    # Filterparameter auf Basis des Filtertyps festlegen
    if filter_type == "bandpass":
        assert isinstance(cutoff_freq, tuple)
        btype = "bandpass"
        cutoff = (cutoff_freq[0] / nyq, cutoff_freq[1] / nyq)

    elif filter_type == "lowpass":
        assert isinstance(cutoff_freq, numbers.Number)
        btype = "lowpass"
        cutoff = cutoff_freq / nyq

    elif filter_type == "highpass":
        assert isinstance(cutoff_freq, numbers.Number)
        btype = "highpass"
        cutoff = cutoff_freq / nyq

    else:
        raise ArgumentError("unknown filter_type")

    # Filter erstellen
    b, a = butter(order, cutoff, fs=fs, btype=btype)

    # Filter auf Spalten anwenden
    data = df.copy()
    for col in df.select_dtypes(include=["number"]).columns:  # nur numerische Spalten auswählen
        data[col] = lfilter(b, a, df[col].values)  # Filter auf Spalte anwenden und Daten speichern

    # Neues DataFrame mit gleichem Index wie das Original und gefilterten Spalten zurückgeben
    return pd.DataFrame(data=data, index=df.index)
