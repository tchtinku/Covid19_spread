import numpy as np
import pandas as pd
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
import shutil
from glob import glob
import os
from covid19_spread.data.usa.process_cases import get_index
import re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    Configuration.create(
        hdx_site="prod", user_agent="A_Quick_Example", hdx_read_only=True
    )
    dataset = Dataset.read_from_hdx("movement-range-maps")
    resources = dataset.get_resources()
    resource = [
        x
        for x in resources
        if re.match(".*/movement-range-data-\d{4}-\d{2}-\d{2}\.zip", x["url"])
    ]
    assert len(resource) == 1
    resource = resource[0]
    url, path = resource.download()
    if os.path.exists(f"{SCRIPT_DIR}/fb_mobility"):
        shutil.rmtree(f"{SCRIPT_DIR}/fb_mobility")
    shutil.unpack_archive(path, f"{SCRIPT_DIR}/fb_mobility", "zip")
    
    fips_map = get_index()
    fips_map["location"] = fips_map["name"] + ", " + fips_map["subregion1_name"]
    
    cols = [
        "date",
        "region",
        "all_day_bing_tiles_visited_relative_change",
        "all_day_ratio_single_tile_users",
    ]
    
    def get_county_mobility_fb(fin):
        df_mobility_global = pd.read_csv(
            fin, parse_dates=["ds"], delimiter="\t", dtype={"polygon_id": str}
        )
        df_mobility_usa = df_mobility_global.query("country == 'USA' ")
        return df_mobility_usa
    
    txt_files = glob(f"{SCRIPT_DIR}/fb_mobility/movement-range*.txt")
    assert len(txt_files) == 1
    fin = txt_files[0]
    df = get_county_mobility_fb(fin)
    df = df.rename(columns={"ds": "date", "polygon_id": "region"})      
    
    df = df.merge(fips_map, left_on="region", right_on="fips")[
        
    ]  