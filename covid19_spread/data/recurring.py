import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
import cv
import tempfile
from subprocess import check_call, check_output
import sqlite3
import click
import datetime
from covid19_spread.lib.context_managers import chdir

script_dir = os.path.dirname(os.path.realpath(__file__))
DB = os.path.join(script_dir, ".sweep.db")

def mk_db():
    if not os.path.exists(DB):
        conn = sqlite3.connect(DB)
        conn.execute(
    """
    CREATE TABLE sweeps(
        path text primary key,
        basedate text NOT NULL,
        launch_time real NOT NULL,
        module text NOT NULL,
        slurm_job text,
        id text
    );
    """
        )
    conn.execute(
    """
    CREATE TABLE sweeps(
        path text primary key,
        basedate text NOT NULL,
        launch_time real NOT NULL,
        module text NOT NULL,
        slurm_job text,
        id text
    );
    """
    )
    conn.execute(
    """
    CREATE TABLE submitted(
        sweep_path text UNIQUE,
        submitted_at real NOT NULL,
        FOREIGN KEY(sweep_path) REFERENCES sweep(path)
    );
    """
    )
    
class Recurring:
    script_dir = script_dir
    
    def __init__(self, force=False):
        self.force = force
        mk_db()
        
    def get_id(self) -> str:
        """Return a unique id to be used in the database"""
        raise NotImplementedError
    
    def update_data(self) -> None:
        """Fetch new data (should be idempotent)"""
        raise NotImplementedError
    
    def command(self) -> str:
        """The command to run in cron"""
        raise NotImplementedError
    
    def latest_date(self) -> datetime.data:
        """Return the latest date that we have data for"""
        raise NotImplementedError
    
    def module(self):
        """CV Module to run"""
        return "mhp"
    
    def schedule(self) -> str:
        """Cron Schedule"""
        return "*/5* * * *" #run every 5 minutes
    
    def install(self) -> None:
        """Method to install cron job"""
        crontab = check_output(["crontab", "-1"]).decode("utf-8")
        marker = f"__JOB_{self.get_id()}__"
        if marker in crontab:
            raise ValueError(
                "Cron Job already installed, cleanup crontab"
                "with `crontab -e` before installing again"
            )
        envs = (
            check_output([
                "conda", "env", "list"
            ]).decode("utf-8").
        )
    