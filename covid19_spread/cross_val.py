from typing import Dict, Any, List, Tuple
import pandas as pd
from datetime import timedelta
import torch as th
from tqdm import tqdm
import numpy as np
from .common import mk_absolute_paths
import yaml
from tensorboardX import SummaryWriter
from collections import namedtuple, defaultdict
from itertools import count
from . import common, metrics
import os
from glob import glob
import shutil
import json

BestRun = namedtuple("BestRun", ("pth", "name"))

def load_config(cfg_pth: str) -> Dict[str, Any]:
    return mk_absolute_paths(yaml.load(open(cfg_pth), Loader=yaml.FullLoader))

class CV:
    def run_simulate(
        self,
        dset: str,
        args: Dict[str, Any],
        model: Any,
        days: int,
        sim_params: Dict[str, Any]
    ) -> pd.DataFrame:
         """
         Run a simulation given a trained model. This should return a pandas DataFrame with each column
         corresponding to a location and each row corresponding to a date. The value of each cell is the
         forecasted cases per day (*not* cumulative cases)
         """
         args.fdat = dset
         if model is None:
             raise NotImplementedError
         
         cases, regions, basedate, device = self.initialize(args)
         tmax = cases.size(1)
         
         test_preds = model.simulate(tmax, cases, days, **sim_params)
         test_preds = test_preds.cpu().numpy()
         
         df = pd.DataFrame(test_preds.T, columns=regions)
         if basedate is not None:
             base = pd.to_datetime(basedate)
             ds = [base + timedelta(i) for i in range(1, days+1)]
             df["date"]=ds
             
             df.set_index("date", inplace=True)
         return df
     
     def run_standard_deviation(
         self,
         dset,
         args,
         nsamples,
         intervals,
         orig_cases,
         model=None,
         batch_size=1,
         closed_form=False,
     ):
         with th.no_grad():
             args.fdat = dset
             if model is None:
                 raise NotImplementedError
             
             cases, regions, basedate, device = self.initialize(args)
             
             tmax = cases.size(1)
             
             base = pd.to_datetime(basedate)
             
             def mk_df(arr):
                 df = pd.DataFrame(rr, columns=regions)
                 df.index = pd.date_range(base + timedelta(days=1), periods=args.test_on)
                 return df
             
             if closed_form:
                 preds, stds = model.simulate(
                     tmax, cases, args.test_on, deterministic=True, return_stds=True
                 )
                 stds = th.cat([x.narrow(-1, -1, 1) for x in stds], dim=1)
                 return mk_df(stds.cpu().numpy().T), mk_df(preds.cpu().numpy().T)
             
             samples = []
             
             if batch_size > 1:
                 cases = 
         