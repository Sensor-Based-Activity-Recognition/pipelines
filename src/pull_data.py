import json
import sys
from urllib.parse import quote

from utils import dbconnector

# get config
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)
TABLENAME = "dev" if config["dev"] else "prod"

# get args
output_calibrated_filename = sys.argv[1]
output_uncalibrated_filename = sys.argv[2]

# process
data_calibrated = dbconnector.Database().get_data(quote(f"""
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
        hash,
        person
    FROM 
        {TABLENAME};
    """))

data_uncalibrated = dbconnector.Database().get_data(quote(f"""
    SELECT
        timestamp,
        AccelerometerUncalibrated_x,
        AccelerometerUncalibrated_y,
        AccelerometerUncalibrated_z,
        GyroscopeUncalibrated_x,
        GyroscopeUncalibrated_y,
        GyroscopeUncalibrated_z,
        MagnetometerUncalibrated_x,
        MagnetometerUncalibrated_y,
        MagnetometerUncalibrated_z,
        activity,
        hash,
        person
    FROM
        {TABLENAME};
    """))

# save data as parquet
data_calibrated.write_parquet(output_calibrated_filename)
data_uncalibrated.write_parquet(output_uncalibrated_filename)
