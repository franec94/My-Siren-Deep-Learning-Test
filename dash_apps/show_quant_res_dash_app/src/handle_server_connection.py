from __future__ import print_function

SHOW_VISDOM_RESULTS = False

# ----------------------------------------------- #
# Python's Imports
# ----------------------------------------------- #

# Std Lib and others.
# ----------------------------------------------- #
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
# from contextlib import closing
from io import BytesIO
from PIL import Image
from pprint import pprint

# import psycopg2 as ps
import contextlib
import collections
import datetime
import functools
import glob
import json
import operator
import os
import sqlite3
import re
import sys
import time
import yaml

import numpy as np
import pandas as pd

from src.functions import check_file_exists

def get_constraints_for_query_db():
    typename = 'QueryConstraints'
    field_names = "image;date;timestamp;hidden_features;image_size;status".split(";")
    field_types = "str;str;str;int;str;str".split(";")

    QueryConstraints = collections.namedtuple(typename, field_names)
    # def map_func(field_type): return dict(zip(['type', 'val'], [eval(field_type), None]))
    # fields_list = list(map(map_func, field_types))
    # constraints = QueryConstraints._make(fields_list)
    # pprint(constraints)
    
    image = dict(zip(['type', 'val'], [str, list(sorted("cameramen".split(";")))]))
    date = None
    timestamp = dict(zip(['type', 'val'], [str, list(sorted("".split(";")))]))
    hidden_features = dict(zip(['type', 'val'], [int, list(sorted("".split(";")))]))
    image_size = dict(zip(['type', 'val'], [str, list(sorted("[256,256]".split(";")))]))
    status = dict(zip(['type', 'val'], [str, list(sorted("done".split(";")))]))

    fields_list = [image, date, timestamp, hidden_features, image_size, status]
    constraints = QueryConstraints._make(fields_list)

    pprint(constraints)
    return constraints

def _get_dict_dataframes(records_list, columns):
    
    if records_list is None or len(records_list) == 0: return None
    
    index_ts = list(records_list[0]._asdict().keys()).index('timestamp')
    index_fp = list(records_list[0]._asdict().keys()).index('full_path')
    
    ts_list = list(map(operator.itemgetter(index_ts), records_list))
    files_list = list(map(operator.itemgetter(index_fp), records_list))
    result_dict_df = dict()
    for a_file, a_ts in zip(files_list, ts_list):
        # print(a_file, a_ts)
        try:
            check_file_exists(a_file, raise_exception=True)
            train_arr = np.loadtxt(a_file)
            indeces = [a_ts] * len(train_arr)
            train_df = pd.DataFrame(data = train_arr, columns = columns, index=indeces)
            result_dict_df[a_ts] = train_df
        except Exception as err:
            print(str(err))
            pass
        pass
    return result_dict_df

def _filter_data(records_list, target_status = 'Done'):
    
    if records_list is None or len(records_list) == 0: return None
    
    def filter_dones(item, attribute_name = 'status', target = f'{target_status}'):
        return getattr(item, attribute_name).lower() == target.lower()
    # records_list_filtered = list(filter(filter_dones, records_list))
    
    index = list(records_list[0]._asdict().keys()).index('status')
    records_list_filtered = list(filter(lambda item: operator.itemgetter(index)(item) == f'{target_status}', records_list))
    
    return records_list_filtered

def _map_data(records_list, root_data_dir):
    
    if records_list is None or len(records_list) == 0: return None
    
    typename = 'RunsLogged2'
    field_names = "image,date,timestamp,hidden_features,image_size,status,data_downloaded,full_path"
    RunsLogged2 = collections.namedtuple(typename, field_names)
    
    def map_to_full_path(item, root_data_dir = f'{root_data_dir}', filename = 'result_comb_train.txt'):
        image_name_r = str(getattr(item, 'image'))
        date_r = str(re.sub('/', '-', getattr(item, 'date')))
        date_r = datetime.datetime.strptime(date_r, '%d-%m-%y').strftime("%d-%m-%Y")
        timestamp_r = str(getattr(item, 'timestamp'))
        full_path_list = [root_data_dir, image_name_r, date_r, timestamp_r, 'train', filename]
        full_path = functools.reduce(lambda a,b : os.path.join(a, b), full_path_list)
        # full_path = os.path.join(root_data_dir, image_name_r, date_r, timestamp_r, 'train', filename)
        # print(full_path)
        return RunsLogged2._make(list(item._asdict().values()) + [full_path])
    
    records_mapped_list = list(map(map_to_full_path, records_list))
    return records_mapped_list
    

def _get_data_from_db(conf_data, sql_statement):
    if conf_data['db_infos']['is_local_db']:
        db_resource = os.path.join(
            conf_data['db_infos']['db_location'],
            conf_data['db_infos']['db_name'])
    else:
        db_resource = conf_data['db_infos']['db_url']
    
    
    
    typename = 'RunsLogged'
    field_names = "image,date,timestamp,hidden_features,image_size,status,data_downloaded"
    RunsLogged = collections.namedtuple(typename, field_names)
    
    records_list = None
    with contextlib.closing(sqlite3.connect(f"{db_resource}")) as connection:
        with contextlib.closing(connection.cursor()) as cursor:
            # Test code snippet:
            # rows = cursor.execute("SELECT 1").fetchall()
            # print(rows)
        
            # Code snippet:
            rows = cursor.execute(f'{sql_statement}').fetchall()
            # pprint(rows)
            records_list = list(map(RunsLogged._make, rows))
            pass
        pass
    # pprint(records_list)
    return records_list
    
def get_data_from_db(conf_data):
    table_name = 'table_runs_logged'
    table_attributes = "image,date,timestamp,hidden_features,image_size,status,data_downloaded"
    
    sql_statement = f"SELECT {table_attributes} FROM {table_name}"
    if conf_data['cropped_image']['flag']:
        crop_size = conf_data['cropped_image']['crop_size']
        if isinstance(crop_size, str):
            # print(crop_size)
            crop_size = re.sub('\)', ']"', crop_size)
            crop_size = re.sub('\(', '"[', crop_size)
            crop_size = re.sub(' ', '', crop_size)
        elif isinstance(crop_size, int):
            crop_size = f'"[{crop_size},{crop_size}]"'
            pass
        sql_statement += f" WHERE image_size = {crop_size}"
        pass

    print(sql_statement)
        
        
    
    records_list = _get_data_from_db(conf_data, sql_statement)
    
    records_list_filtered = _filter_data(records_list, target_status = 'done')
    records_list_mapped = _map_data(records_list_filtered, root_data_dir = conf_data['db_infos']['root_data_dir'])
    
    # pprint(records_list_mapped)
    
    columns = conf_data['columns_df_str'].split(";")
    result_dict_df = _get_dict_dataframes(records_list_mapped, columns)
    return result_dict_df, records_list_mapped

def get_data_from_db_by_status(conf_data, status = '*'):
    table_name = 'table_runs_logged'
    table_attributes = "image,date,timestamp,hidden_features,image_size,status,data_downloaded"
    
    sql_statement = f"SELECT {table_attributes} FROM {table_name}"
    if conf_data['cropped_image']['flag']:
        crop_size = conf_data['cropped_image']['crop_size']
        if isinstance(crop_size, str):
            # print(crop_size)
            crop_size = re.sub('\)', ']"', crop_size)
            crop_size = re.sub('\(', '"[', crop_size)
            crop_size = re.sub(' ', '', crop_size)
        elif isinstance(crop_size, int):
            crop_size = f'"[{crop_size},{crop_size}]"'
            pass
        sql_statement += f" WHERE image_size = {crop_size}"
        pass
    
    if status != '*':
        sql_statement += f" and status = {status}"

    print(sql_statement + ";")
        
        
    
    records_list = _get_data_from_db(conf_data, sql_statement + ";")
    
    if status != '*':
        records_list_filtered = _filter_data(records_list, target_status = f'{status}')
    else:
        records_list_filtered = records_list
    records_list_mapped = _map_data(records_list_filtered, root_data_dir = conf_data['db_infos']['root_data_dir'])
    
    # pprint(records_list_mapped)
    
    # columns = conf_data['columns_df_str'].split(";")
    # result_dict_df = _get_dict_dataframes(records_list_mapped, columns)
    # return result_dict_df, records_list_mapped
    return records_list_mapped

def get_data_from_db_by_constraints(conf_data, constraints = '*', fetch_data_downloaded = False):
    result_dict_df = None
    table_name = 'table_runs_logged'
    table_attributes = "image,date,timestamp,hidden_features,image_size,status,data_downloaded"
    sql_statement = f"SELECT {table_attributes} FROM {table_name}"
    if constraints is not None and len(constraints) != 0:
        sql_statement = \
            f"{sql_statement}" \
            + f" WHERE {constraints}"
        pass
    print(sql_statement + ";")
    
    records_list = _get_data_from_db(conf_data, sql_statement + ";")
    
    records_list_mapped = _map_data(records_list, root_data_dir = conf_data['db_infos']['root_data_dir'])
    
    if fetch_data_downloaded is True:
        records_list_filtered = list(filter(lambda item: getattr(item, 'data_downloaded') == 'TRUE', records_list_mapped))
        result_dict_df = _get_dict_dataframes(records_list = records_list_filtered, columns = conf_data['columns_df_str'].split(";"))
        pass
    return records_list_mapped, result_dict_df, sql_statement + ";"

def check_exists_data_read_from_logs_db(db_resource, table_name, data):
    
    records_exist = []
    with contextlib.closing(sqlite3.connect(f"{db_resource}")) as connection:
        with contextlib.closing(connection.cursor()) as cursor:
            # Test code snippet:
            # rows = cursor.execute("SELECT 1").fetchall()
            # print(rows)
        
            for a_record in data:
                # print(a_record['timestamp'])
                t = a_record['timestamp']['type']
                val = a_record['timestamp']['vals']
                if t is str:
                    val = f"'{val}'"
                else:
                    val = t(val)
                    pass
                sql_statement = f'SELECT COUNT(*) FROM {table_name}' + \
                    f" WHERE timestamp = {val}"
                # print(sql_statement + ";")
                rows = cursor.execute(f'{sql_statement}' + ';').fetchall()
                # print(rows)
                if rows[0] != (0,):
                    records_exist.append(a_record['timestamp']['vals'])
            pass
        pass
    return records_exist

def insert_data_read_from_logs_db(db_resource, table_name, data):
    with contextlib.closing(sqlite3.connect(f"{db_resource}")) as connection:
        with contextlib.closing(connection.cursor()) as cursor:
            # Test code snippet:
            # rows = cursor.execute("SELECT 1").fetchall()
            # print(rows)
        
            for ii, (attr_names, attr_values) in enumerate(data):
                print('-' * 50)
                print('Record no.', ii+1)
                print('-' * 50)
                sql_statement = f'INSERT INTO {table_name} {attr_names} VALUES {attr_values}'
                print(f'{sql_statement}' + ';')
                rows = cursor.execute(f'{sql_statement}' + ';').fetchall()
                print(rows)
            pass
            connection.commit()
        pass
    pass