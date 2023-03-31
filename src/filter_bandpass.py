from utils.filters import butt_filter
from dill import load, dump
import sys
import time

input_filename = sys.argv[1]
lowcut = float(sys.argv[2])
highcut = float(sys.argv[3])
order = int(sys.argv[4])
print(
    f"Apply bandpass filter on {input_filename} with low cutoff of {lowcut}Hz and high cutoff of {highcut}Hz and order of {order}"
)

# execute filtering
with open(input_filename, "rb") as fr: #load data
    data = {} #filtered data stored he§re
    for key, segments in load(fr).items():
        data[key] = [butt_filter(segment, "bandpass", (lowcut, highcut), order) for segment in segments]

    # dump fft of windows
    with open("data/bandpass.dill", "wb") as fw:
        dump(data, fw)

time.sleep(1) #wait a second until file closed