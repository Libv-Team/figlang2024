import json
import os


def load_checkpoint(file_name):
    if not os.path.exists(file_name):
        # os.makedirs(file_name)
        json.dump([], open(file_name, "w"))
    result = json.load(open(file_name))
    return result


def save_checkpoint(data, file_name):
    s = file_name.split("/")[-1]
    if not os.path.exists(file_name.replace(s,"")):
        os.makedirs(file_name.replace(s,""))
    if not os.path.exists(file_name):
        json.dump([], open(file_name, "w"))
    json.dump(data, open(file_name, "w"))
