import os
import pandas as pd
import sys
from datetime import timedelta
from delpgi_epidata import Epidata
import covidcast


#fetch data
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def main(geo_value, source, signal):
    df = covidcast.metadata().drop(columns=["time_type", "min_lag", "max_lag"])
    df.min_time = pd.to_datetime(df.min_time)
    df.max_time = pd.to_datetime(df.max_time)
    df = df.query(
        f"data_source == '{source}' and signal == '{signal}' and geo_type == '{geo_value}'"
    )
    assert len(df) == 1
    base_date = df.iloc[0].min_time - timedelta(1)
    end_date = df.iloc[0].max_time
    dfs = []
    current_date = base_date
    while current_date < end_date:
        current_date = current_date + timedelta(1)
        date_str = current_date.strftime("%Y%m%d")
        os.makedirs(os.path.join(SCRIPT_DIR, geo_value, source), exist_ok=True)
        fout = f"{SCRIPT_DIR}/{geo_value}/{source}/{signal}-{date_str}.csv"
        
        if os.path.exists(fout):
            dfs.append(pd.read_csv(fout))
            continue
        
        for _ in range(3):
            res = Epidata.covidcast(source, signal, "day", geo_value, [date_str], "*")
            print(date_str, res["result"], res["message"])
            if res["result"] == 1:
                break
            
        if res["result"] != 1:
            print(f"Skipping {source}/{signal} for {date_str}")
            continue
        