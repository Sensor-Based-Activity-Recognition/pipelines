from utils.filters import butterworth_filter
from dill import load, dump
import sys
import time

input_filename = sys.argv[1]
cutoff = float(sys.argv[2])
order = int(sys.argv[3])
print(
    f"Apply highpass filter on {input_filename} with cutoff of {cutoff}Hz and order of {order}"
)

# execute filtering
with open(input_filename, "rb") as fr: #load data
    data = {} #filtered data stored here
    for key, segments in load(fr).items():
        data[key] = [butterworth_filter(segment, "highpass", cutoff, order) for segment in segments]

    # dump fft of windows
    with open("data/highpass.dill", "wb") as fw:
        dump(data, fw)

time.sleep(1) #wait a second until file closed