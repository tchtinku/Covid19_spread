import pandas as pd
from datetime import datetime
from covid19_spread.data.usa.process_cases import get_index

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    print("Getting Google mobility data...")
    cols = [
        "date",
        "region",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline"
    ]
    
    def get_county_mobility_google(fin=None):
        if fin is None:
            fin = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
        df_Gmobility_global = pd.read_csv(
            fin, parse_dates=["date"], dtype={"census_fips_code": str}
        )
        df_Gmobility_usa = df_Gmobility_global.query("country_region_code == 'US")
        return df_Gmobility_usa

    df = get_county_mobility_google()
    df = df[~df["census_fips_code"].isnull()]
    index = get_index()
    index["region"] = index["subregion2_name"] + ", " + index["subregion1_name"]
    df = df.merge(
        index, left_on="census_fips_code", right_on="fips", suffixes=("", "_x")
    )[list(df.columns) + ["region"]]
    
    df = df[cols]
    
    val_cols = [c for c in df.columns if c not in {"region", "date"}]
    ratio = (1+df.set_index(["region", "date"]) / 100).reset_index()
    piv = ratio.pivot(index="date", columns="region", values=val_cols)
    piv = piv.rolling(7, min_periods=1).mean().transpose()
    piv.iloc[0] = piv.iloc[0].fillna(0)
    
    piv = piv.fillna(0)
    
    dfs = []
    for k in piv.index.get_level_values(0).unique():
        df = piv.loc[k].copy()
        df["type"] = k
        dfs.append(df)
    df = pd.concat(dfs)
    df = df