from __future__ import print_function

from PIL import Image
from pprint import pprint

import collections
import datetime
import functools
import json
import logging
import operator
import os
import pathlib
import sys
import time
import yaml

import numpy as np
import pandas as pd

# skimage
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def  get_root_level_logger(root_path, loggger_name='train.log'):
    log_filename = os.path.join(root_path, f'{loggger_name}')
    log_filename_exists = check_file_exists(log_filename, raise_exception=False)
    if log_filename_exists:
        os.remove(log_filename)
        pass

    logging.basicConfig(filename=f'{log_filename}', level=logging.INFO)
    pass


def check_file_exists(file_path, raise_exception=True):
    if not os.path.isfile(file_path):
        if raise_exception:
            raise Exception(f"Error: file '{file_path}' does not exists!")
        return False
    return True

def check_dir_exists(dir_path, raise_exception=True):
    if not os.path.isdir(dir_path):
        if raise_exception:
            raise Exception(f"Error: directory '{dir_path}' does not exists!")
        else: return False
    return True

def get_all_files_by_ext(dir_path, ext, recursive_search = False, regex_filter = None, verbose = 0):
    files_list = []
    check_dir_exists(dir_path)
    if isinstance(ext, list) == False:
        ext = [ext]
    if recursive_search:
        for a_ext in ext:
            data = pathlib.Path(f'{dir_path}').rglob(f'*.{a_ext}')
            if data != None:
                files_list.extend(data)
    else:
        for a_ext in ext:
            data = pathlib.Path(f'{dir_path}').glob(f'*.{a_ext}')
            if data != None:
                files_list.extend(data)
                if verbose == 1:
                    data_2 = pathlib.Path(f'{dir_path}').glob(f'*.{a_ext}')
                    for path in data_2:
                        print(path.name)
                files_list.extend(data)
    if len(files_list) == 0: return []
    
    def filter_files(a_file, raise_exception = False):
        return check_file_exists(a_file, raise_exception)
    files_list = list(filter(filter_files, files_list))
    
    if regex_filter != None:
        if isinstance(regex_filter, list) == False:
            regex_filter = [regex_filter]
            filtered_files = []
            for a_regex in regex_filter:
                res_tmp = list(filter(lambda xx: a_regex.match(xx.name) != None, files_list))
                filtered_files.extend(res_tmp)
        return filtered_files
    
    return files_list


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except PermissionError as err:
            print(f"Error raised when dealing with dir '{dir_path}' creation!", file=sys.stderr)
            print(str(err))
            sys.exit(-1)
            pass
        except:
            pass
        pass
    pass


def read_conf_file(conf_file_path, raise_exception = True):
    _ = check_file_exists(file_path=conf_file_path, raise_exception=raise_exception)
    try:
        with open(conf_file_path, "r") as f:
            conf_data = yaml.load(f, Loader=yaml.FullLoader)
            pass
        return conf_data
    except Exception as err:
        raise Exception(f"Error: when reading input conf file '{conf_file_path}'")
    pass

def load_target_image(image_file_path = None):
    # image_file_path = 'test068.png'
    im = None
    if image_file_path != None:
        image_exists = check_file_exists(image_file_path, raise_exception=False)
        if image_exists:
            im = Image.open(f'{image_file_path}')
            return im
    
    im = Image.fromarray(skimage.data.camera())
    return im

def get_dataframe(conf_data):
    if 'result_timestamp' in conf_data.keys():
        result_timestamp = conf_data['result_timestamp']
        if result_timestamp == 'None' or result_timestamp is None:
            columns = conf_data['columns_df_str'].split(";")
            a_file = conf_data['result_file_path']
    
            check_file_exists(a_file, raise_exception=True)
            train_arr = np.loadtxt(a_file)
            train_df = pd.DataFrame(data = train_arr, columns = columns)
            return train_df
        
    index_timestamp = conf_data['results_timestamps'].index(result_timestamp)
    a_file, a_ts = \
        conf_data['results_file_paths'][index_timestamp], conf_data['results_timestamps'][index_timestamp]
    check_file_exists(a_file, raise_exception=True)
    train_arr = np.loadtxt(a_file)
    train_df = pd.DataFrame(data = train_arr, columns = columns)
    return train_df

def get_dict_dataframes(conf_data):
    columns = conf_data['columns_df_str'].split(";")
    result_dict_df = dict()
    for a_file, a_ts in zip(conf_data['results_file_paths'], conf_data['results_timestamps']):
        try:
            check_file_exists(a_file, raise_exception=True)
            train_arr = np.loadtxt(a_file)
            indeces = [a_ts] * len(train_arr)
            train_df = pd.DataFrame(data = train_arr, columns = columns, index=indeces)
            result_dict_df[a_ts] = train_df
        except Exception as _:
            pass
        pass
    return result_dict_df

def laod_data_from_files_list(files_list, as_df = False, columns = None):
    if files_list == None or len(files_list) == 0: return None
    data_arr = []
    def load_data_and_concat(a, b):
        b_arr = np.loadtxt(f"{b}")
        if len(a) == 0:
            return b_arr
        return np.concatenate((a, b_arr), axis=0)
    res_arr = functools.reduce(lambda a,b : load_data_and_concat(a, b), files_list, data_arr)
    if as_df is True:
        if columsn != None:
            return pd.DataFrame(res_arr, columns = columns)
        else:
            return pd.DataFrame(res_arr)
    return res_arr
    