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
    """Run cross validation for one set of hyperparameters"""  
    try:
        basedir = basedir.replace("%j", submitit.JobEnvironment().job_id)
    except Exception:
        pass #running locally basedir is fine
    os.makedirs(basedir, exist_ok=True)
    print(f"CWD = {os.getcwd()}")
    
    def _path(path):
        return os.path.join(basedir, path)
    
    load_configs(cfg, module, _path(prefix + f"{module}.yml"))
    
    n_models = cfg[module]["train"].get("n_models", 1)
    if n_models > 1:
        launcher = map if executor is None else executor.map_array
        fn = partial(
            run_cv,
            module,
            prefix=prefix,
            basedate=basedate,
            executor=executor,
            test_run=test_run
        )
        configs = [
            set_dict(copy.deepcopy(cfg), [module, "train", "n_models"], 1)
            for _ in range(n_models)
        ]
        basedirs = [os.path.join(basedir, f"job_{i}") for i in range(n_models)]
        with ExitStack() as stack:
            if executor is not None:
                stack.enter_context(executor.set_folder(os.path.join(basedir, "%j")))
                
            jobs = list(launcher(fn, basedirs, configs))
            launcher = (
                ensemble
                if executor is None
                else partial(executor.submit_dependent, jobs, ensemble)
            )
            ensemble_job = launcher(basedirs, cfg, module, prefix, basedir)
            if executor is not None:
                #Whatever job depend on "this" job, should be extended to the newly created job
                executor.extend_dependencies(jobs + [ensemble_job])
            return jobs + [ensemble_job]
    
    #setup input/output paths
    dset = cfg[module]["data"]
    val_in = _path(prefix + "filtered_" + os.path.basename(dset))
    val_test_key = "test" if test_run else "validation"
    val_out = _path(prefix + cfg[val_test_key]["output"])
    cfg[module]["train"]["fdat"] = val_in
    
    mod = importlib.import_module("covid19_spread." + module).CV_CLS()
    
    # --- store configs to reproduce results
    log_configs(cfg, module, _path(prefix + f"{module}.yml"))
    
    ndays = 0 if test_run else cfg[val_test_key]["days"]
    if basedate is not None:
        #If we want to train from a particular basedate, then also subtract
        #out the different in days. Ex. if ground truth contains data up to 5/20/2020
        #but the basedate is 5/10/2020, then drop an extra 10 days in addition to validation days.
        gt = metrics.load_ground_truth(dset)
        assert  gt.index.max() >= basedate
        ndays += (gt.index.max() - basedate).days
        
    filter_validation_days(dset, val_in, ndays)
    #apply data pre-processing
    preprocessed = _path(prefix + "preprocessed_" + os.path.basename(dset))
    mod.preprocessed(val_in, preprocessed, cfg[module].get("preprocess", {}))
    
    mod.setup_tensorboard(basedir)
    #setup logging
    train_params = Namespace(**cfg[module]["train"])
    n_models = getattr(train_params, "n_models", 1)
    print(f"Training {n_models} models")
    # --- train ----
    model = mod.run_train(
        preprocessed, train_params, _path(prefix + cfg[module]["output"])
    )
    
    # --- simulate ----
    with th.no_grad():
        sim_params = cfg[module].get("simulate", {})
        #Returns the number of new cases for each day
        df_forecast_deltas = mod.run_simulate(
            preprocessed,
            train_params,
            model,
            sim_params=sim_params,
            days=cfg[val_test_key]["days"]
        )
        df_forecast = common.rebase_forecast_deltas(val_in, df_forecast_deltas)
        
    mob.tb_writer.close()
    
    print(f"Storing validation in {val_out}")
    df_forecast.to_csv(val_out, index_label="date")
    
    #--metrics---
    metric_args = cfg[module].get("metrics", {})
    df_val, json_val = mod.compute_metrics(
        cfg[module]["data"], val_out, model, metric_args
    )
    df_val.to_csv(_path(prefix + "metrics.csv"))
    with open(_path(prefix + "metrics.json"), "w") as fout:
        json.dump(json_val, fout)
    print(df_val)
    
    # ------ Prediction Interval ---------
    if "prediction_interval" in cfg and prefix == "final_model_":
        try:
            with th.no_grad():
                #FIXME: refactor to use rebase_forecast_deltas
                gt = metrics.load_ground_truth(val_in)
                basedate = gt.index.max()
                prev_day = gt.loc[[basedate]]
                pred_interval = cfg.get("prediction_interval", {})
                df_std, df_mean = mod.run_standard_deviation(
                    preprocessed,
                    train_params,
                    pred_interval.get("nsamples", 100),
                    pred_interval.get("intervals", [0.99, 0.95, 0.8]),
                    prev_day.values.T,
                    model,
                    pred_interval.get("batch_size", 8),
                    closed_form=True,
                )
                df_std.to_csv(_path(f"{prefix}std_closed_form.csv"), index_label="date")
                df_mean.to_csv(
                    _path(f"{prefix}mean_closed_form.csv"), index_label=date
                )
                piv = mod.run_prediction_interval(
                    _path(f"{prefix}mean_closed_form.csv"),
                    _path(f"{prefix}std_closed_form.csv"),
                    pred_interval.get("intervals", [0.99, 0.95, 0.8]),
                )
                piv.to_csv(_path(f"{prefix}piv.csv"), index=False)
        except NotImplementedError:
            pass
        
def filter_validation_days(dset: str, val_in: str, validation_days: int):
    """Fliters validation days and writes output to val_in path"""
    if dset.endswith(".csv"):
        common.drop_k_days_csv(dset, val_in, validation_days)
    elif dset.endswith(".h5"):
        common.drop_k_days(dset, val_in, validation_days)
    else:
        raise RuntimeError(f"Unrecognised dataset extension: {dset}")
    
def load_model(model_pth, cv, args):
    chkpnt = th.load(model_pth)
    cv.initialize(args)
    cv.func.load_state_dict(chkpnt)
    return cv.func

def  copy_assets(cfg, dir):
    if isinstance(cfg, dict):
        return {k: copy_assets(v, dir) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [copy_assets(x, dir) for x in cfg]
    elif isinstance(cfg, str) and os.path.exists(cfg):
        new_pth = os.path.join(dir, "assets", os.path.basename(cfg))
        shutil.copy(cfg, new_pth)
        return new_pth
    else:
        return cfg
    
def log_configs(cfg: Dict[str, Any], module: str, path: str):
    """Logs configs for job for reproducibility"""
    with open(path, "w") as f:
        yaml.dump(cfg[module], f)
        
def run_best(config, module, remote, basedir, basedate=None, executor=None):
    mod = importlib.import_module("covid19_spread." + module).CV_CLS()
    sweep_config = load_config(os.path.join(basedir, "cfg.yml"))
    best_runs = mod.model_selection(basedir, config=sweep_config[module], module=module)
    
    if remote and executor is None:
        executor = mk_executor(
            "model_selection", basedir, config[module].get("resources", {})
        )
        
    with open(os.path.join(basedir, "model_selection.json"), "w") as fout:
        json.dump([x.asdict() for x in best_runs], fout)
        
    cfg = copy.deepcopy(config)
    best_runs_df = pd.DataFrame(best_runs)
    
    def run_cv_and_copy_results(tags, module, pth, cfg, prefix):
        try:
            jobs = run_cv(
                module, 
                pth,
                cfg,
                prefix=prefix,
                basedate=basedate,
                executor=executor,
                test_run=True,
            )
            
            def rest():
                shutil.copy(
                    os.path.join(pth, f'finsl_model_{cfg["validation"]["output"]}'),
                    os.path.join(
                        os.path.dirname(pth), f"forecasts/forecast_{tag}.csv"
                    ),
                )
                if "prediction_interval" in cfg:
                    piv_pth = os.path.join(
                        pth,
                        f'final_model_{cfg["prediction_interval"]["output_std"]}'
                    )
                    if os.path.exists(piv_pth):
                        shutil.copy(
                            piv_pth,
                            os.path.join(
                                os.path.dirname(pth), f"forecasts/std_{tag}.csv"
                                ),
                        )
            if cfg[module]["train"].get("n_module", 1) > 1 and executor is not None:
                            
                
    
    
    
        

