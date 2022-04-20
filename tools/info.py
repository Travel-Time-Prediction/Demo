import pandas as pd
import streamlit as st
import datetime

from .process_travel_time import predict

def get_result(path, model_path, road_num):
    data_path = path + f"{road_num}.csv"
    model_path = f"E:/checkpoints/model_weight"
    result = predict(data_path, model_path, 21)
    result['time'] = pd.to_datetime(result['time'])

    return result