import csv
import pandas as pd
import json


def read_csv(csv_file_path):
    with open(csv_file_path, "r") as file:
        csv_reader = csv.reader(file)
        data_list = list(csv_reader)
    return data_list


def pd_read_csv(csv_file_path, NaN_fill=True, NaN_value=0):
    df = pd.read_csv(csv_file_path)
    # Fill NaN values with 0
    if NaN_fill:
        df = df.fillna(NaN_value)

    return df


def save_pd_csv(df, save_path):
    df.to_csv(save_path)


def read_json(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data
