import json
import sys
from utils import dbconnector

# get config
with open("config.json") as f:
    config = json.load(f)
table_name = "dev" if config["dev"] else "prod"


# get data
data = dbconnector.Database().get_data(f"SELECT * FROM {table_name} LIMIT 10")

# save data as parquet
out_filename = sys.argv[1]
data.write_parquet(out_filename)