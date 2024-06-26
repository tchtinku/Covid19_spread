import os
from .. import recurring
import pandas
from ...lab.slack import post_slack_message
from datetime import date, datetime, timedelta
from .convert import main as convert

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class USARRecurring(recurring.Recurring):
    script_dir = SCRIPT_DIR
    
    def get_id(self):
        return "us-bar"
    
    def command(self):
        return f"recurring run us"
    
    def module(self):
        return "bar_time_features"
    
    def schedule(self):
        return "*/10 * * * *"
    
    def update_data(self):
        convert("cases", with_features=False, source="nyt", resolution="county")
        
    def latest_date(self):
        df = pandas.read_csv(f"{SCRIPT_DIR}/data_cases.csv", index_col="region")
        max_date = pandas.to_datetime(df.columns).max().data()
        if max_date < (date.today() - timedelta(days=1)) and datetime.now().hour > 17:
            excepted_date = date.today() - timedelta(days=1)
            msg = f"*WARNING: new data for {excepted_date} is still not available!*"
            post_slack_message(channel="#cron_errors", text=msg)
        return pandas.to_datetime(df.columns).max().date()
    
    def launch_job(self, **kwargs):
        #Make clean with features
        convert("cases", with_features=True, source="nyt", resolution="county")
        msg = f"*New Data available for US: {self.latest_date()}*"
        post_slack_message(channel="#new-data", text=msg)
        return super().launch_job(
            module="bar", cv_config="us", array_parallelism=90, **kwargs
        )
    