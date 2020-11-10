from datetime import datetime
# from google.colab import files

from pathlib import Path
from collections import namedtuple
from io import BytesIO
from pprint import pprint

# import psycopg2 as ps
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
import seaborn as sns
# sns.set_theme(style="whitegrid")
import ipywidgets as widgets
# back end of ipywidgets
from IPython.display import display

import io
from googleapiclient.http import MediaIoBaseDownload
import zipfile

import collections
import itertools
import functools
import glob
import operator
import os
import re
import yaml
import numpy as np
import pandas as pd

from PIL import Image

# skimage
# ----------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# sklearn
# ----------------------------------------------- #
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from utils.handle_server_connection import get_data_from_db, get_data_from_db_by_status, get_data_from_db_by_constraints, insert_data_read_from_logs_db, check_exists_data_read_from_logs_db
from utils.functions import read_conf_file, load_target_image, get_dict_dataframes, get_dataframe, check_file_exists
from utils.db_tables import TableRunsDetailsClass, TableRunsDetailsTupleClass

def fetch_data(conf_data):
    records_list = None
    if not conf_data['data_fetch_strategy']['fetch_from_db']:
        if conf_data['is_single_run']:
            train_df = get_dataframe(conf_data)
        else:
            ts_list = '1603410154-248962,1603421693-497763'.split(",")
            # ts_list = '1603421693-497763'.split(",")
            result_dict_df = get_dict_dataframes(conf_data)
                # train_df: pd.DataFrame = result_dict_df['1603410154-248962'] # train_df: pd.DataFrame = result_dict_df['1603478755-305517']
    
            data = list(map(operator.itemgetter(1), filter(lambda item: item[0] in ts_list, result_dict_df.items())))
            train_df = pd.concat(data)
            print(collections.Counter(train_df['hl']))
            pass
    else:
        result_dict_df, records_list = get_data_from_db(conf_data)
        # data = list(map(operator.itemgetter(1), filter(lambda item: item[0] in ts_list, result_dict_df.items())))
        data = list(map(operator.itemgetter(1), result_dict_df.items()))
        train_df = pd.concat(data)
        print(collections.Counter(train_df['hl']))
        pass
    return train_df, result_dict_df, records_list

def fetch_data_by_status(conf_data, status = '*'):
    records_list = None
    if not conf_data['data_fetch_strategy']['fetch_from_db']:
        return None
    else:
        records_list = get_data_from_db_by_status(conf_data, status = status)
        pass
    return records_list

def chain_constraints_as_str(constraints, fetch_data_downloaded = False):
        def map_constraint(item):
            # pprint(item)
            attr_name = item[0]
            attr_type = item[1]['type']
            attr_vals = item[1]['val']
            """
            if len(attr_vals) == 0: return ''
            if len(attr_vals) == 1:
                if len(attr_vals[0]) == 0: return ''
            """
            
            def reduce_func(a, b, attr_type = attr_type):
                if attr_type is int:
                    return f'{attr_name} = {b}' if a == '' else f" {a} OR {attr_name} = {b} "
                elif attr_type is str:
                    b_ = f"'{b}'"
                    return f"{attr_name} = {b_}" if a == '' else f" {a} OR {attr_name} = {b_} "
                else:
                    raise Exception(f'Error {str(attr_type)} not allowed!')
            res = functools.reduce(reduce_func, attr_vals, f'')# f'{attr_name} = ')
            # print(res)
            return res
        
        
        def filter_unecessary_constraints(item):
            vals = operator.itemgetter(1)(item)
            if vals != None:
                if len(vals['val']) == 1:
                    if len(vals['val'][0]) == 0:
                        # print('zero')
                        return False
                    pass
                return True
            return False
         
        chained_constraints = str(functools.reduce(lambda a,b: f'({b})' if a == '' else f" {a} AND ({b}) ",
            list(map(map_constraint, 
                     # filter(lambda item: operator.itemgetter(1)(item) != None, constraints._asdict().items())
                     filter(filter_unecessary_constraints, constraints._asdict().items())
            )), f''
        ))
        return chained_constraints

def fetch_data_by_constraints(conf_data, constraints, fetch_data_downloaded = False):
    records_list, res = None, None
    
    typename = 'QueryConstraints2'
    field_names = "image;date;timestamp;hidden_features;image_size;status".split(";")
    
    QueryConstraints2 = collections.namedtuple(typename, field_names)
    chained_constraints = chain_constraints_as_str(constraints)
    
    records_list, result_dict_df, query_str = get_data_from_db_by_constraints(conf_data, chained_constraints, fetch_data_downloaded=fetch_data_downloaded)
    
    return records_list, result_dict_df, query_str, chained_constraints

def get_info_from_logged_parser(parser_logged_files, train_log_files, original_images_dict, gpu_mode = False):
    trd = TableRunsDetailsClass()
    failed_read = []
    
    opt_list = sorted('logging_root,hidden_features,hidden_layers,seeds,lr,num_epochs,sidelength'.split(','))
    
    tmp_list = 'hf,hl,image_name,lr,epochs,seed'.split(',')
    # opt_list = 'logging_root,hidden_features,hidden_layers,seeds,lr,num_epochs,sidelength'.split(',')
    relation_dict = dict(zip(opt_list[:-1], tmp_list))
    def to_tuple(item):
        item_list = item.strip().split(" ")
        key = item_list[0]
        if key == 'logging_root':
            # vals = [os.path.basename(item_list)]
            vals = [os.path.basename(item_list[1:][0])]
        else: 
            vals = item_list[1:]
        return (key, vals) 
    
    for ii, (parser_logged_file, train_log_file) in enumerate(zip(parser_logged_files, train_log_files)):
        print('-' * 50)
        print('File no.', ii+1)
        print('-' * 50)
        
        exists_1 = check_file_exists(parser_logged_file, raise_exception=False)
        pprint(dict(filename=parser_logged_file,exists_code=exists_1))

        exists_2 = check_file_exists(train_log_file, raise_exception=False)
        pprint(dict(filename=train_log_file,exists_code=exists_1))

        if exists_1 is False or exists_2 is False:
            print('Skipped.')
            continue
        try:
            with open(f'{parser_logged_file}', 'r') as f:
                res_dict = dict(zip(opt_list, [None] * len(opt_list)))
                content = f.read().split('\n')
                # pprint(content)
                res = list(filter(lambda l: True in [l.startswith(f'{opt}') for opt in opt_list], list(filter(lambda l: l.startswith('Command'), content))[0].split('--')))
                res_dict.update(**dict(map(to_tuple, res)))

                res = dict(map(lambda lr: (lr.strip().split(':')[0][2:], [lr.strip().split(':')[1].strip()]), list(filter(lambda l: l.find('--lr:') != -1, content))))
                res_dict.update(res)
                res_dict = dict(map(lambda x: x if x[1] != None else (x[0], ['NULL']), res_dict.items()))
                # pprint(res_dict)
            
                res_comb = list(functools.reduce(lambda a,b: b if a is None else [','.join([x for x in item]) for item in itertools.product(a, b)], res_dict.values(), None))
                flag = False
                res_comb_list = [dict(zip(res_dict.keys(), x.split(','))) for x in res_comb]
                # pprint(res_comb_list)
                pass
            ts = None
            opt_list_2 = 'timestamp'.split(',')
            with open(f'{train_log_file}', 'r') as f:
                content = f.read().split('\n')
                # pprint(content)
                try:
                    ts = list(filter(lambda l: True in [opt in l for opt in opt_list_2], content))[0].split('timestamp=')[1].split(']')[0]
                except:
                    ts = re.findall('[0-9]{10}\-[0-9]{6}', f'{train_log_file}')[0]
                # res_dict.update(**dict(map(to_tuple, res)))
                pass
            # print(ts)
            
            for attempt in res_comb_list:
                a_record = TableRunsDetailsTupleClass()
                updated_dict = trd.update_record_by_dict(a_record, attempt, relation_dict)
                
                updated_dict['timestamp'] = ts
                # pprint(res_dict)
                if res_dict['sidelength'][0] != 'NULL':
                    updated_dict['cropped_width'] = str(res_dict['sidelength'][0])
                    updated_dict['cropped_heigth'] = str(res_dict['sidelength'][0])
                    updated_dict['is_cropped'] = str('TRUE')
                else:
                    updated_dict['is_cropped'] = str('FALSE')
                    pass
                # pprint(updated_dict['image_name']['vals'])
                image_name = updated_dict['image_name']['vals']
                # pprint(original_images_dict.keys())
                if image_name in list(original_images_dict.keys()):
                    h = original_images_dict[f'{image_name}']['heigth']
                    w = original_images_dict[f'{image_name}']['width']
                    updated_dict['width'] = w
                    updated_dict['heigth'] = h
                    # print(w, h)
                    pass
                else:
                    image_name = re.findall('siren\-train\-logs.*', train_log_file)[0].split('\\')[1]
                    h = original_images_dict[f'{image_name}']['heigth']
                    w = original_images_dict[f'{image_name}']['width']
                    updated_dict['width'] = w
                    updated_dict['heigth'] = h
                    # raise Exception(f'{image_name} not found')
                    pass
                if gpu_mode is True:
                    updated_dict['gpu'] = str('TRUE')
                else:
                    updated_dict['gpu'] = str('FALSE')
                    pass
                
                trd.append(updated_dict)
                pass
        except Exception as err:
            print(str(err))
            # pprint(content)
            # pprint(res_dict)
            # pprint(res_comb_list)
            print(ts)
            flag = True
            failed_read.append(f'{parser_logged_file}')
        finally:
            if flag:
                print('Failed.')
            else:    
                print('Success.')
                pass
            pass
        pass
    return trd, failed_read

def insert_data_read_from_logs(conf_data, table_name, records):
    
    if conf_data['db_infos']['is_local_db']:
        db_resource = os.path.join(
            conf_data['db_infos']['db_location'],
            conf_data['db_infos']['db_name'])
    else:
        db_resource = conf_data['db_infos']['db_url']
        pass
    
    records_exist = check_exists_data_read_from_logs_db(db_resource, table_name, records)
    
    def filter_func(item, black_list = records_exist):
        ts = item['timestamp']['vals']
        if ts in black_list: return False
        return True
    # pprint(records_exist)
    if len(records_exist) == 0:
        print('no records filtered!')
        filtered_recs = records
    else:
        print('some records to be filtered!')
        filtered_recs = list(filter(filter_func, records))
        pass
    # pprint(filtered_recs)
    trd = TableRunsDetailsClass()
    
    # pprint(filtered_recs[0])
    for a_record_dict in filtered_recs:
        a_record = trd.make_record(list(a_record_dict.values()))
        # pprint(a_record)
        trd.append(a_record)
        pass
    data = trd.get_all_records_processed_for_query_insert()
    # pprint(data)
    insert_data_read_from_logs_db(db_resource, table_name, data)
    pass