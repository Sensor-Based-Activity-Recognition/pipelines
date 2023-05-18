from utils.filters import butterworth_filter
from dill import load, dump
import sys
import time
import yaml
from tqdm import tqdm

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

# get params
params = yaml.safe_load(open("params.yaml"))[stage_name]
filter_type = params["filter_type"]
order = params["order"]  # get order


# execute filtering
with open(input_filename, "rb") as fr:  # load data
    # compose cutoff frequency argument
    if filter_type == "bandpass":  # check if filter type is bandpass -> two cutoffs
        cutoff_freq = (params["lowcut_f_Hz"], params["highcut_f_Hz"])

    else:  # is filtertype lowpass or highpass -> only one cutoff
        cutoff_freq = params["cutoff_f_Hz"]

    print(
        f"Apply {filter_type} filter on {input_filename} with cutoff: {cutoff_freq} and order: {order}"
    )

    # apply filter on all segments
    data = {}
    for key, segments in tqdm(load(fr).items()):
        data[key] = [
            butterworth_filter(segment, filter_type, cutoff_freq, order)
            for segment in segments
        ]

    # save data
    with open(output_filename, "wb") as fw:
        dump(data, fw)

time.sleep(1)  # wait a second until file closed
