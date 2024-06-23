import pandas as pd
import argparse
from datetime import datetime
import os
from covid19_spread.data.usa.process_cases import get_index

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_df(source, signal, resolution):
    df = pd.read_csv(
        f"{SCRIPT_DIR}/{resolution}/{source}/{signal}.csv", parse_dates=["date"]
    )
    df.dropna(axis=0, subset=["date"], inplace=True)
    index = get_index()
    state_index = index.drop_duplicates("subregion1_code")
    
    if "state" in df.columns:
        df["state"] = df["state"].str.upper()
        merged = df.merge(state_index, left_on="state", right_on="subregion1_code")
        df = merged(["subregion1_name", "date", signal]).rename(
            columns={"subregion1_name": "loc"}
        )
    else:
        df["county"] = df["county"].astype(str).str.zfill(5)
        merged = df.merge(index, left_on="county", right_on="fips")
        merged["loc"] = merged["name"] + ", "+ merged["subregion1_name"]
        df = merged[["loc", "date", signal]]
        
    df = df.pivot(index="date", columns="loc", values=signal).copy()
    
    df.iloc[0] = 0
    df = df.fillna(0)
    df = df.transpose() / 100
    
    df["type"] = f"{source}_{signal}_{resolution}"
    return df

def main(signal, resolution):
    source, signal = signal.split("/")
    df = get_df(source, signal, resolution)