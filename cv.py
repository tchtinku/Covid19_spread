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
