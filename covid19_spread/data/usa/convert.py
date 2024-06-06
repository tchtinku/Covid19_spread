import argparse
import numpy as np
import pandas as pd
import torch as th
from os import listdir
from os.path import isfile, join
from covid19_spread.data.usa.process_cases import SOURCES
import warnings
from covid19_spread.common import standardize_county_name
import os
import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

nyc_boroughs = [
    "Boronx, New York",
    "Kings, New York",
    "Queens, New York",
    "New York, New York",
    "Richmond, New York"
]

def county_id(county, state):
    return f"{county}, {state}"

def rename_nyc_boroughs(county_name):
    if county_name in nyc_boroughs:
        return "New York City, New York"
    else:
        return county_name
    
    
def merge_nyc_boroughs(df, ntypes):
    df["region"] = df["region"].transform(rename_nyc_boroughs)
    prev_len = len(df)
    df = df.groupby(["region", "type"]).mean()
    assert len(df) == prev_len - ntypes * 4, (prev_len, len(df))
    df = df.reset_index()
    print(df[df["region"] == "New York City, New York"])
    return df