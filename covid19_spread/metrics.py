import numpy as np
import pandas as pd
from datetime import timedelta

def load_ground_truth(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"region": "date"})
    df.set_index("date", inplace=True)
    df = df.transpose()
    df.index = pd.to_datetime(df.index)
    return df

def rmse(pred, gt):
    return (pred - gt).pow(2).mean(axis=1).pow(1.0 / 2)

def mae(pred, gt):
    return (pred - gt).abs().mean(axis=1)

def mape(pred, gt):
    return ((pred - gt).abs() / gt.clip(1)).mean(axis=1)

def compute_metrics(df_true, df_pred, mincount=0, nanfill=False):
    if isinstance(df_true, str):
        df_true = load_ground_truth(df_true)
    if isinstance(df_pred, str):
        df_pred = pd.read_csv()
        
        