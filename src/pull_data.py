import json
import sys
from urllib.parse import quote

from utils import dbconnector

# get config
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)
TABLENAME = "dev" if config["dev"] else "prod"


query = f"""
    SELECT  
        timestamp,
        Accelerometer_x,
        Accelerometer_y,
        Accelerometer_z,
        Gyroscope_x,
        Gyroscope_y,
        Gyroscope_z,
        Magnetometer_x,
        Magnetometer_y,
        Magnetometer_z,
        activity,
        hash
    FROM 
        {TABLENAME};
    """

# convert query to url format
query = quote(query)

# get data
data = dbconnector.Database().get_data(query)

# save data as parquet
out_filename = sys.argv[1]
data.write_parquet(out_filename)