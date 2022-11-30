import os
from preprocessing import json_to_csv
import pandas as pd

# load in data
def load_data():
    df = pd.DataFrame()
    for filename in os.listdir("items"):
        with open(os.path.join(os.cwd(), filename), 'r', encoding = 'utf-8') as f:
            df = df.append(pd.read_json(f))
    
    df.to_csv("processed_input.csv", encoding = 'utf-8', index = False)

if __name__ == "main":
    load_data()