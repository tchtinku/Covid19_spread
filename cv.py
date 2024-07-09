import copy 
import click
import importlib
import itertools
import json
import pandas as pd
import os
import random
import shutil
import submitit
import tempfile
import torch as th
import re
import yaml
from argparse import Namespace
from datetime import datetime
from functools import partial
from glob import glob, iglob
from typing import Dict, Any, List, Optional
from contextlib import nullcontext, ExitStack
from covid19_spread import common
from covid19_spread import metrics
from covid19_spread.lib import cluster
from covid19_spread.lib.click_lib import DefaultGroup
from covid19_spread.lib.slurm_pool_executor import (
    SlurmPoolExecutor,
    JobStatus,
    TransactionManager
)
from covid19_spread.lib.slack import post_slack_message
from submitit.helpers import RsyncSnapshot
from covid19_spread.cross_val import load_config
import sqlite3
from ax.service.ax_client import AxClient
from ax.exceptions.generation_strategy import MaxParallelismReachedException
import time
import queue
import threading

def set_dict(d: Dict[str, Any], keys: List[str], v: Any):
    """
    Update a dict using a nested ist of keys
    Ex:
        x = {'a': {'b': {'c': 2}}}
        set_dict(x, ['a', 'b'], 4) == {'a':{'b': 4}}
    """
    if len(keys) > 0:
        d[keys[0]] = set_dict(d[keys[0]], keys[1:], v)
        return d
    else:
        return v
    
    
def mk_executor(
    name: str, folder: str, extra_params: Dict[str, Any]; ex=SlurmPoolExecutor, **kwargs
):
    executor = (ex or submitit.AutoExecutor)(folder=folder, **kwargs)
    executor.update_parameters(
        job_name=name,
        partition=cluster.PARTITION,
        gpus_per_node=extra_params.get("gpus", 0),
        cpus_per_task=extra_params.get("cpus", 3),
        mem=f'{cluster.MEM_GB(extra_params.get("array_parallelism", 100))}',
        time=extra_params.get("timeout", 12*60),
    )
    return executor

def ensemble(basedirs, cfg, module, prefix, outdir):
    def _path(x):
        return os.path.join(basedir, prefix + x)
    means = []
    stds = []
    mean_deltas = []
    kwargs = {"index_col":"date", "parse_dates": ["date"]}
    stdfile = "std_closed_form.csv"
    meanfile = "mean_closed_form.csv"
    for basedir in basedirs:
        if os.path.exists(_path(cfg["validation"]["output"])):
            means.append(pd.read_csv(_path(cfg["validation"]["output"]), **kwargs))
        if os.path.exists(_path(stdfile)):
            stds.append(pd.read_csv(_path(stdfile), **kwargs))
            mean_deltas.append(pd.read_csv(_path(meanfile), **kwargs))
    if len(stds) > 0:
        #Average the variance and take square root
        std = pd.concat(stds.pow(2).groupby(level=0).mean().pow(0.5))
        std.to_csv(os.path.join(outdir, prefix + meanfile))
        mean_deltas = pd.concat(mean_deltas).groupby(level=0).mean()
        mean_deltas.to_csv(os.path.join(outdir, prefix + meanfile))
        
    assert len(means) > 0, "All ensemble jobs failed!!!"
    
    mod = importlib.import_module("covid19_spread." + module).CV_CLS()
    
    if len(stds) > 0:
        pred_interval = cfg.get("prediction_interval", {})
        piv = mod.run_prediction_interval(
            os.path.join(outdir, prefix+meanfile)
            os.path.join(outdir, prefix+stdfile),
            pred_interval.get("intervals", [0.99, 0.95, 0.8])
        )
        piv.to_csv(os.path.join(outdir, prefix + "piv.csv"), index=False)
        
    mean = pd.concat(means).groupby(level=0).median()
    outfile = os.path.join(outdir, prefix + cfg["validation"["output"]]) 
    mean.to_csv(outfile, index_label="date")
    
    # --metrics ---
    metric_args = cfg[module].get("metrics", {})
    df_val, json_val = mod.compute_metrics(
        cfg[module]["data"], outfile, None, metric_args
    )
    df_val.to_csv(os.path.join(outdir, prefix + "metrics.csv"))
    with open(os.path.join(outdir, prefix + "metrics.json"), "w") as fout:
        json.dump(json_val, fout)
    print(df_val)
    
def run_cv(
    module: str,
    basedir: str,
    cfg: Dict[str, Any],
    prefix="",
    basedate=None,
    executor=None,
    test_run: bool = False, #is this a test or validation run
):
    """Run """    

