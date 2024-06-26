from slack import WebClient
import os
import json
import warnings


def post_slack_message(channel, text):
    cred_path = os.path.expanduser("~/.credentials.json")
    if not os.path.exists(cred_path):
        msg = "Could not find ~/.redentials.json with slack credentials, not posting message ..."
        warnings.warn(msg, UserWarning)
        return
    credentials = json.load(open(cred_path))
    if "slack" not in credentials or "bot_token" not in credentials["slack"]:
        warnings.warn(
            "Could not find slack credentials in ~/.credentials.json", UserWarning
        )
        return
    client = WebClient(token=credentials["slack"]["bot_token"])
    client.chat_postMessage(channel=channel, text=text)
     