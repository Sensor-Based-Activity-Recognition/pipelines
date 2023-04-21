import json
import sys
from urllib.parse import quote
import yaml

from utils import dbconnector

# get args
stage_name = sys.argv[1]
output_filename = sys.argv[2]

# get config
params = yaml.safe_load(open("params.yaml"))[stage_name]
params_db = params["database"]["questdb"]
fetch_get_calibrated = params["get_calibrated"]

# get table name
TABLENAME = params_db["table_name"]

if fetch_get_calibrated:
    dbconnector.Database(params_db).get_data(quote(f"""
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
    """)).write_parquet(output_filename)
else:
    dbconnector.Database(params_db).get_data(quote(f"""
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
    """)).write_parquet(output_filename)
