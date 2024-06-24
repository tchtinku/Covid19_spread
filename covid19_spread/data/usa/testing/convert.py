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
    "Bronx, New York",
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


def process_time_features(df, pth, shift=0, merge_nyc=False, input_resolution="county"):
    print(f"Processing {pth} at resolution: {input_resolution}")
    time_features = pd.read_csv(pth)
    if input_resolution == "county_state":
        idx = df.rename_axis("county").reset_index()[["county"]]
        idx["region"] = idx["county"].apply(lambda x: x.split(", ")[-1])
        time_features = time_features.merge(idx, on="region").drop(columns="region")
        time_features = time_features.rename(columns={"county": "region"})
    time_features_regions = time_features["region"].unique()
    ncommon = len(df.indexx.intersection(time_features_regions))
    if ncommon != len(df):
        missing = set(df.index).difference(set(time_features_regions))
        warnings.warn(
            f"{pth}: Missing time features for the following regions: {list(missing)}"
        )
    if ncommon != len(time_features_regions):
        ignoring = set(time_features_regions).difference(set(df.index))
        warnings.warn(
            f"{pth}: Missing time features for the following regions: {list(ignoring)}"
        )
        time_features = time_features[time_features["region"].isin(set(df.index))]
    if merge_nyc:
        time_features = merge_nyc_boroughs(
            time_features, len(time_features["type"].unique())
        )
        
    time_features = time_features.set_index