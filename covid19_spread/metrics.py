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
    return _compute_metrics(df_true, df_pred, mincount, nanfill=nanfill)    

def _compute_metrics(df_true, df_pred, mincount=0, nanfill=False):
    if nanfill:
        cols = sorted(set(df_true.columns).difference(set(df_pred.columns)))
        zeros = pd.DataFrame(np.zeros((len(df_pred), len(cols))), columns=cols)
        zeros.index = df_pred.index
        df_pred = pd.concat([df_pred, zeros], axis=1)
        
    common_cols = list(set(df_true.columns).intersection(set(df_pred.columns)))
    df_pred = df_pred[common_cols]
    df_true = df_true[common_cols]
    z = len(df_pred)
    
    basedate = df_pred.index.min()
    pdate = basedate - timedelta(1)
    
    diff = df_true.loc[pdate] - df_true.loc[basedate - timedelta(2)]
    naive = [df_true.loc[pdate] + d * diff for d in range(1, z + 1)]
    naive = pd.DataFrame(naive)
    naive.index = df_pred.index
    
    ix = df_pred.index.intersection(df_true.index)
    
    df_pred = df_pred.loc[ix]
    naive = naive.loc[ix]
    gt = df_true.loc[ix]
    
    #compute state level MAE
    state_gt = gt.transpose().groupby(lambda x: x.split(", ")[-1]).sum()
    state_pred = df_pred.transpose().groupby(lambda x: x.split(", ")[-1]).sum()
    state_mae = (state_gt.sort_index() - state_pred.sort_index()).abs().mean(axis=0)
    
    metrics = pd.DataFrame(
        [
            rmse(df_pred, gt),
            mae(df_pred, gt),
            mape(df_pred, gt),
            rmse(naive, gt),
            mae(naive, gt),
            state_mae,
            max_mae(df_pred, gt),
            max_mae(naive, gt)
        ],
        columns = df_pred.index.to_numpy(),
    )
    metrics["Measure"] = [
        "RMSE",
        "MAE",
        "MAPE",
        "RMSE_NAIVE",
        "MAE_NAIVE",
        "STATE_MAE",
        "MAX_MAE",
        "MAX_NAIVE_MAE",
    ]
    metrics.set_index("Measure", inplace=True)
    if metrics.shape[1] > 0:
        metrics.loc["MAE_MASE"] = metrics.loc["MAE"] / metrics.loc["MAE_NAIVE"]
        metrics.loc["RMSE_MASE"] = metrics.loc["RMSE"] / metrics.loc["RMSE_NAIVE"]
        
        #Stack predictions onto last ground truth date.
        #We'll take the diff and compute MAE on the new daily counts
        stack = pd.concat(
            [df_true.loc[[df_pred.index.min() - timedelta(days=1)]], df_pred]
        )
        stack_diff = stack_diff().loc[ix]
        true_diff = df_true.diff().loc[ix]
        metrics.loc["MAE_DELTAS"] = mae(stack_diff, true_diff)
        metrics.loc["RMSE_DELTAS"] = rmse(stack_diff, true_diff)
    return metrics
        