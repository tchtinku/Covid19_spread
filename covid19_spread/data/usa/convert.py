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

def process_time_features(df, pth, shift=0, merge_nyc=False, input_resolution="county"):
    print(f"Processing {pth} at resolution: {input_resolution}")
    time_features = pd.read_csv(pth)
    if input_resolution == "county_state":
        #Expand state level time features to each county in 'df'
        idx = df.rename_axis("county").reset_index()[["county"]]
        idx["region"] = idx["county"].apply(lambda x: x.split(", ")[-1])
        time_features = time_features.merge(idx, on="region").drop(columns="region")
        time_features = time_features.rename(columns={"county": "region"})
    time_features_regions = time_features["region"].unique()
    ncommon = len(df.index.intersection(time_features_regions))
    if ncommon != len(df):
        missing = set(df.index).difference(set(time_features_regions))
        warnings.warn(
            f"{pth}: Missing time features for the following reasons: {list(missing)}"
        )
    if ncommon != len(time_features_regions):
        ignoring = set(time_features_regions).difference(set(df.index))
        warnings.warn(
            f"{pth}: Ignoring time features for the following resons: {list(ignoring)}"
        )
        time_features = time_features[time_features["region"].isin(set(df.index))]
    if merge_nyc:
        time_features = merge_nyc_boroughs(
            time_features, len(time_features["type"].unique())
        )
    # Transpose to have two level columns (region, type) and dates as index
    time_features = time_features.set_index(["region", "type"]).transpose().sort_index()
    time_features.index = pd.to_datetime(time_features.index)
    # Trim prefix if it starts before the dates in df
    time_features = time_features.loc[time_features.index >= df.columns.min()]
    #Fill in dates that are missing in `time_features` that exists in df
    time_features = time_features.reindex(df.columns)
    #shift time features UP by `shift` days
    time_features = time_features.shift(shift)
    #forward fill the missing values
    time_features = time_features.fillna(0)
    time_features = time_features[time_features.columns.sort_values()]
    feature_tensors = {
        region: th.from_numpy(time_features[region].values)
        for region in time_features.columns.get_level_values(0).unique()
    }
    if input_resolution == "county_state":
        pth = pth.replace("state", "county_state")
    th.save(feature_tensors, pth.replace(".csv", ".pt"))
    
def run_par(fs, args, kwargs, max_par=None):
    if not isinstance(fs, list):
        fs = [fs]*len(args)
    if "MAX_PARALLELISM" in os.environ:
        max_par = int(os.environ["MAX_PARALLELISM"])
    print(f"Max parallelism = {max_par}")
    if max_par is not None and max_par <= 1:
        for _f, _args, _kwargs in zip(fs, args, kwargs):
            _f(*_args, **_kwargs)
        return
    with mp.Pool(max_par) as pool:
        results = [
            pool.apply_async(f, args=a, kwds=k) for f, a, k in zip(fs, args, kwargs)
        ]
        [r.get() for r in results]
        
def create_time_features():
    from .symptom_survey import prepare as ss_prepare
    from .fb import prepare as fb_prepare
    from .google import prepare as google_prepare
    from .testing import prepare as testing_prepare
    
    fs = [ss_prepare, fb_prepare, google_prepare, testing_prepare]
    run_par(fs, [()]*len(fs), [{}]*len(fs))
    
def main(metric, with_features, source, resolution):
    df = SOURCES[source](metric)
    df.index = pd.to_datetime(df.index)
    
    dates = df.index
    df.columns = [c.split("_")[1] + ", "+c.split("_")[0] for c in df.columns]
    
    #drop all zero columns
    df = df[df.columns[(df.sum(axis=0) != 0 ).values]]
    
    df = df.transpose() #row for each county, columns correspond to dates....
    
    #make sure counts are strictly increasing
    df = df.cummax(axis=1)
    
    #throw away all-zero columns i.e days with no cases
    counts = df.sum(axis=0)
    df = df.iloc[:, np.where(counts>0)[0]]
    
    if resolution == "state":
        df = df.groupby(lambda x: x.split(", ")[-1]).sum()
        df = df.drop(
            index=["Virgin Islands", "Northern Mariana Islands", "Puerto Rico", "Guam"],
            errors="ignore",
        )
    
    county_id = {c: i for i, c in enumerate(df.index)}
    
    df.to_csv(f"{SCRIPT_DIR}/data_{metric}.csv", index_label="region")
    