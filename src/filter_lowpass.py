from utils.filters import butt_filter
from dill import load, dump
import sys
import time

input_filename = sys.argv[1]
cutoff = float(sys.argv[2])
order = int(sys.argv[3])
print(
    f"Apply lowpass filter on {input_filename} with cutoff of {cutoff}Hz and order of {order}"
)

# execute filtering
with open(input_filename, "rb") as fr: #load data
    data = {} #filtered data stored here
    for key, segments in load(fr).items():
        data[key] = [butt_filter(segment, "lowpass", cutoff, order) for segment in segments]

    # dump fft of windows
    with open("data/lowpass.dill", "wb") as fw:
        dump(data, fw)
    
time.sleep(1) #wait a second until file closed