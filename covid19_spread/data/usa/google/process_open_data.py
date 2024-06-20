import pandas
from datetime import datetime
import os
from covid19_spread.data.usa.process_cases import get_index

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    index = pandas.read_csv(
        "https://storage.googleapis.com/covid19-open-data/v2/index.csv"
    )
    
    state_index = index[(index["key"].str.match("^US_[A-Z]+$")).fillna(False)]
    index = get_index()
    
    def zscore(piv):
        piv = (piv-piv.mean(skipna=True)) / piv.std(skipna=True)
        piv = piv.fillna(method="ffill").fillna(method="bfill")
        return piv
    
    def zero_one(df):
        df = df.fillna(0)
        df = df / df.max(axis=0)
        df = df.fillna(0)
        return df
    
    def process_df(df, columns, resolution, func_normalize):
        idx = state_index if resolution == "state" else index
        merged = df.merge(idx, on="key")
        if resolution == "state":
            exclude = {"US_MP", "US_AS", "US_GU", "US_VI", "US_PR"}
            merged = merged[~merged["key"].isin(exclude)]
            merged["region"] = merged["subregion1_name"]
        else:
            merged["region"] = merged["name"] + ", " + merged["subregion1_name"]
        piv = merged.pivot(index="date", columns="region", values="columns")
        if func_normalize is not None:
            piv = func_normalize(piv)
            
        dfs = []
        for k in piv.columns.get_level_values(0).unique():
            dfs.append(piv[k].transpose())
            dfs[-1]["type"] = k
        df = pandas.concat(dfs)
        df = df[["type"] + [c for c in df.columns if isinstance(c, datetime)]]
        df.columns = [
            str(c.date()) if isinstance(c, datetime) else c for c in df.columns
        ]
        return df.fillna(0)
        