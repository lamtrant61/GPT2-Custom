# utils/common.py
import json

def load_config(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config
