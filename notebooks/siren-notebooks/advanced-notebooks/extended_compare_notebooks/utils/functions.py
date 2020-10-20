import os

import numpy as np
import pandas as pd


def get_dataframe_histories(path_history_trains, columns):
    # columns_df = ['#params', 'seed', 'hl', 'hf', 'mse', 'psnr', 'ssim', 'train_eta']

    results_history_arr = None
    for path_history_train in path_history_trains:
        # print(path_history_train)
        if results_history_arr is None:
            results_history_arr = np.loadtxt(path_history_train)
            # print(results_history_arr)
        else:
            try:
                tmp_arr = np.loadtxt(path_history_train)
                # print(tmp_arr)
                results_history_arr = np.concatenate((results_history_arr, tmp_arr), axis = 0)
            except:
                tmp_arr = np.loadtxt(path_history_train)
                # print(tmp_arr)
                results_history_arr = np.concatenate((results_history_arr, [tmp_arr]), axis = 0)
                pass
            pass
        pass

    results_history_df = pd.DataFrame(
        data = results_history_arr,
        columns = columns)
    return results_history_df


def adjust_trains_path(root_path, trains_no, trains_datetime = None):
    if root_path == "/content":
        path_history_trains = [os.path.join(
            root_path,
            f'result_comb_train_{train_no}.txt')
        for train_no in trains_no]
    elif root_path == ".":
        path_history_trains = [os.path.join(
            root_path,
            f'result_comb_train_{train_no}.txt')
        for train_no in trains_no]
    else:
        path_history_trains = [os.path.join(
            root_path,
            train_datetime,
            "train",
            f'result_comb_train_{train_no}.txt')
            for train_datetime, train_no in zip(trains_datetime, trains_no)]
    return path_history_trains


def get_path_histories(root_path, input_data_config):
    def adjust_date_format(date_input):
        return '-'.join([xx for xx in date_input.split('-')[::-1]])
    date_inputs_tmp = list(map(adjust_date_format, input_data_config.dates_input))
    # print(date_inputs_tmp)

    train_timestamps_tmp = [train_timestamp.replace('.', '-') for train_timestamp in input_data_config.train_timestamps]
    # print(train_timestamps_tmp)

    trains_datetime = [os.path.join(date_input_tmp, train_timestamp_tmp)
                   for date_input_tmp, train_timestamp_tmp in zip(date_inputs_tmp, train_timestamps_tmp)]
    # print(trains_datetime)
    # print(input_data_config.trains_no)
    # print('Date train:', train_datetime)
    path_history_trains = []

    path_history_trains = adjust_trains_path(root_path, input_data_config.trains_no)
    print("Path location:")
    print(path_history_trains)
    return path_history_trains
    