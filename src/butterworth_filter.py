from utils.filters import butterworth_filter
from dill import load, dump
import sys
import time
import yaml
from tqdm import tqdm

input_filename = sys.argv[1]
filter_type = sys.argv[2]
output_filename = sys.argv[3]

filter_params = yaml.safe_load(open("params.yaml"))["filters"][filter_type] #get filter params
order = filter_params["order"] #get order

# execute filtering
with open(input_filename, "rb") as fr: #load data
    #compose cutoff frequency argument
    if filter_type == "bandpass": #check if filter type is bandpass -> two cutoffs
        cutoff_freq = (filter_params["lowcut_f_Hz"], filter_params["highcut_f_Hz"])

    else: #is filtertype lowpass or highpass -> only one cutoff
        cutoff_freq = filter_params["cutoff_f_Hz"]

    print(
        f"Apply {filter_type} filter on {input_filename} with cutoff: {cutoff_freq} and order: {order}"
    )

    data = {} #filtered data stored here
    for key, segments in tqdm(load(fr).items()):
        data[key] = [butterworth_filter(segment, filter_type, cutoff_freq, order) for segment in segments]

    # dump fft of windows
    with open(output_filename, "wb") as fw:
        dump(data, fw)

time.sleep(1) #wait a second until file closed


