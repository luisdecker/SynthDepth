"""Utils regarding file operations"""

import json


def read_json(json_filepath):
    with open(json_filepath, "r") as f:
        return json.load(f)


def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f)
