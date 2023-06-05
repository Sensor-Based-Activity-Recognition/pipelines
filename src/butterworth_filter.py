from utils.filters import butterworth_filter
from dill import load, dump
import sys
import time
import yaml
from tqdm import tqdm

# Argumente werden geholt.
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

# Parameter werden geholt.
params = yaml.safe_load(open("params.yaml"))[stage_name]
filter_type = params["filter_type"]  # Filtertyp wird festgelegt.
order = params["order"]  # Ordnung des Filters wird geholt.

# Filterung wird ausgeführt.
with open(input_filename, "rb") as fr:  # Daten werden geladen.
    # "cutoff frequency" Argument wird zusammengesetzt.
    if filter_type == "bandpass":  # Wenn der Filtertyp "bandpass" ist, werden zwei Cutoff-Frequenzen benötigt.
        cutoff_freq = (params["lowcut_f_Hz"], params["highcut_f_Hz"])
    else:  # Wenn der Filtertyp "lowpass" oder "highpass" ist, wird nur eine Cutoff-Frequenz benötigt.
        cutoff_freq = params["cutoff_f_Hz"]

    print(
        f"Apply {filter_type} filter on {input_filename} with cutoff: {cutoff_freq} and order: {order}"
    )  # Information über die Anwendung des Filters wird ausgegeben.

    # Filter wird auf alle Segmente angewendet.
    data = {}
    for key, segments in tqdm(load(fr).items()):
        data[key] = [
            butterworth_filter(segment, filter_type, cutoff_freq, order)
            for segment in segments
        ]

    # Daten werden gespeichert.
    with open(output_filename, "wb") as fw:
        dump(data, fw)

time.sleep(1)  # Das Skript wartet eine Sekunde bis die Datei geschlossen wird.
