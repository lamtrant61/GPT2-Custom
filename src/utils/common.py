# utils/common.py

def load_config(filename):
    import json
    with open(filename, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config

def load_csv_data(filename):
    import pandas as pd
    data = pd.read_excel(filename)
    return data
