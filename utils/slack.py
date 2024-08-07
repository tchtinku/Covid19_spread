from slack import WebClient
from slack.errors import SlackApiError
import os
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("msgfile")
opt = parser.parse_args()

credentials = json.load(open(os.path.expanduser("~/.credentials.json")))["slack"]
client = WebClient(token=credentials["bot_token"])

with open(opt.msgfile, "r") as fin:
    msg = fin.read()
    
client.chat_postMessage(channel='#cron_errors', text=msg)