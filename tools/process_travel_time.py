import os
import pandas as pd
import numpy as np

from model.SAT_LSTM import AttentionalLSTM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.signal import butter, filtfilt
from .normailze import Normalizer
from .travel_time_data import TravelTimeDataset

def split_data_over(data_path):
    times = [i for i in range(21)]

    num_data_points_list = []
    df_list = []

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    period = 21
    time_part = 21 // period
    time_list = [times[i * period:i * period + period] for i in range(time_part)]
    for idx, time in enumerate(time_list):
        df_tmp = df[df.index.hour.isin(time)]
        
        num_data_points_list.append(len(df_tmp['2019-09-01' :'2019-12-31']))
        df_list.append(df_tmp['2019-09-01' :'2019-12-31'])
            
    return df_list, num_data_points_list

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def preprocess(data, window_size):
    X, y, y_date = [], [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i].to_list())
        y.append(data[i])
        y_date.append(data.index[i])
    return np.array(X), np.array(y), np.array(y_date)

def generate_train_test_val(normalize_data, window_size):
    # split data for test model
    # split data
    data_test = normalize_data[:]

    # split data y
    data_x_test, data_y_test, data_date_y_test = preprocess(data_test, window_size)

    return data_x_test, data_y_test, data_date_y_test

def init_weights(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def eval(dataset_test, data_date_y_test, scaler, model, batch_size, model_path, best_model_name, df, device, window_size):
    # init tool for train model
    # craete dataloader
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(os.path.join(model_path, best_model_name + '.pth'))
    model.load_state_dict(checkpoint['net'])

    predicted_test = np.array([])

    model.eval()
    for batch_idx, (x, y) in enumerate(test_dataloader):
        x = x.to(device)
        pred = model(x.float())
        pred = pred.cpu().detach().numpy()
        predicted_test = np.concatenate((predicted_test, pred[:, 0]))

    predict_test_series = pd.Series(predicted_test, index=data_date_y_test)
    to_plot_predicted_test = scaler.inverse_transform(predict_test_series, window_size)
    to_plot_predicted_test = to_plot_predicted_test.to_frame()
    to_plot_predicted_test = to_plot_predicted_test.rename(columns = {0:'delta_t'})
    to_plot_predicted_test['type'] = 'predict'

    df_tmp = df['delta_t'][window_size:]
    df_tmp = df_tmp.to_frame()
    df_tmp['type'] = 'target'

    result = pd.concat([df_tmp, to_plot_predicted_test])
    result = result.reset_index()
    result = result.rename(columns = {'index':'time'})

    return result
    


def predict(data_path, model_path, window_size):
    dfs, num_data_points_list = split_data_over(data_path)

    for idx, df in enumerate(dfs):
        denoise = butter_lowpass_filter(df['delta_t'].to_list(), cutoff=3, fs=10, order=2)
        dfs[idx]['denoise'] = denoise

    scaler_list = [Normalizer() for _ in range(len(dfs))]
    normalize_data_list = [scaler_list[idx].fit_transform(dfs[idx]['denoise']) for idx in range(len(dfs))]

    data_x_test_list, data_y_test_list, data_date_y_test_list = [], [], []
    for idx in range(len(normalize_data_list)):
        data_x_test, data_y_test, data_date_y_test = generate_train_test_val(normalize_data_list[idx], window_size)
        data_x_test_list.append(data_x_test)
        data_y_test_list.append(data_y_test)
        data_date_y_test_list.append(data_date_y_test)

    dataset_test = []
    for idx in range(len(data_x_test_list)):
        dataset_test.append(TravelTimeDataset(data_x_test_list[idx], data_y_test_list[idx]))

    result = []
    for idx in range(len(dataset_test)):
        model = AttentionalLSTM(input_size=1, qkv=window_size, hidden_size=256, num_layers=1, output_size=1, bidirectional=False)
        model.apply(init_weights)
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        model = model.to(device)

        road = os.path.basename(data_path).split('.')[0]
        result = eval(dataset_test[idx], data_date_y_test_list[idx], scaler_list[idx], model, 32, model_path, f"butterworth_road_{road}", dfs[idx], device, 21)

        return result

    #report = pd.DataFrame(result)