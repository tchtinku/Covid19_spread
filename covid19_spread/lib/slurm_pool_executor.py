import numpy as np
import submitit
from submitit.slurm.slurm import SlurmExecutor, SlurmJob
from submitit.core import core, utils
import uuid
import typing as tp
import time
import sys
import os
import sqlite3
import enum
import random
from contextlib import(
    contextmanager,
    redirect_stderr,
    redirect_stdout,
    AbstractContextManager,
)
import traceback
import itertools
import timeit
from covid19_spread.lib.context_managers import env_var

class TransactionManager(AbstractContextManager):
    """
    Class for managing exclusive database transactions. This locks the entire
    database to ensure atomicity. This allows nesting transactions, where 
    the inner transaction is idempotent
    """
    
    def __init__(self, db_pth: str, nretries: int=20):
        self.retries = nretries
        self.db_pth = db_pth
        self.conn = None
        self.cursor = None
        self.nesting = 0
        self.start_time = None
        
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state["nesting"] = 0
        state["conn"] = None
        state["cursor"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def run(self, txn, ntries: int = 100):
        exn = None
        for _ in range(ntries):
            try:
                with self as conn:
                    conn.execute("BEGIN EXCLUSIVE")
                    return txn(conn)
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                sleep_time = random.randint(0, 10)
                print(f"Transaction failed! Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                exn = e
        print("Failed too many times!!!!")
        raise exn
    
    def __exit__(self, exc_type, exc_val, tb):
        self.nesting -= 1
        print(f"Exiting transaction, nesting = {self.nesting}")
        if exc_type is not None:
            traceback.print_exc(file=sys.stdout)
            
        if self.nesting == 0:
            if exc_type is None:
                print("Committing transaction")
                self.conn.commit()
            else:
                print("Rolling back transaction")
                self.conn.rollback()
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None
            print(f"Finished transaction n {timeit.default_timer() - self.start_time}")
            self.start_time = None
            
            
class JobStatus(enum.IntEnum):
    pending = 0
    success = 1
    failure = 2
    final = 3 #pending if all other jobs are finished
    
    def __conform__(self, protocol):
        if protocol is sqlite3.PrepareProtocol:
            return self.value
        
class Worker:
    def __init__(self, db_pth: str, worker_id: int):
        self.db_pth = db_pth
        self.worker_id = worker_id
        self.sleep = 0
        self.worker_finished = False
        self.current_job = None
        
    def fetch_ready_job(self, cur):
        #select pending job that doesn't have any unfinished dependencies
        query = f"""
        SELECT
              jobs.pickle,
              jobs.job_id,
              jobs.retry_count,
              MIN(COALESCE(j2.status, {JobStatus.success})) as min_status,
              MAX(COALESCE(j2.status, {JobStatus.failure})) AS max_status
        FROM jobs
        LEFT JOIN dependencies USING(pickle)
        LEFT JOIN jobs j2 ON dependencies.depends_on=j2.pickle
        WHERE
            jobs.status={JobStatus.pending} AND
            jobs.id='{self.db_pth}' AND
            (dependencies.id='{self.db_pth}' OR dependencies.id IS NULL) AND
            (j2.id='{self.db_pth}' OR j2.id IS NULL)
        GROUP BY jobs.pickle, jobs.job_id 
            HAVING MIN(COALESCE(j2.status, {JobStatus.success})) >= {JobStatus.success} 
            AND MAX(COALESCE(j2.status, {JobStatus.success})) <= {JobStatus.success}
        LIMIT 1
        """
        cur.execute(query)
        return cur.fetchall()
    
                    