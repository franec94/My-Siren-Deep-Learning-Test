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


def get_input_data_configs(path_conf_file = 'config.yml'):
    try:
        with open(path_conf_file) as f:
            config = yaml.safe_load(f)
            pass
        # pprint(config)

        config_sorted = sorted(config.items(), key=operator.itemgetter(0))
        # pprint(config)

        InputDataConfig = namedtuple('InputDataConfig', list(map(operator.itemgetter(0), config_sorted)))
        # print(InputDataConfig.__doc__)

        input_data_config = InputDataConfig._make(list(map(operator.itemgetter(1), config_sorted)))
        # pprint(input_data_config)
    except Exception as err:
        print(str(err))
        raise Exception(f'Failed to read configuration from {path_conf_file} cofiguration file!')
    finally:
        print('config file read with success.')
        pass
    return input_data_config

def get_output_graphics_configs(basedir_path_out_images = '.', path_conf_file = 'graphics_config.yml', train_no = 0):
    try:
        with open(path_conf_file) as f:
            config = yaml.safe_load(f)
            pass
        # pprint(config)
        
        
        image_kinds = config['kind']
        # image_kind_str = "df_scatter;scatter;bar;reg;point;box;violin;complex"
        # images_kind = list(map(lambda xx: f"{xx}plot", filter(lambda xx: len(xx) != 0, sorted(image_kind_str.split(";")))))

        images_kind = list(map(lambda xx: f"{xx}plot", filter(lambda xx: len(xx) != 0, sorted(image_kinds))))

        ImagesConf = namedtuple('ImagesConf', images_kind)
        
        half_name = f"mse_psnr_et_al_vs_no_params_train_no_{train_no}.png"
        def full_path_out_images(item, root_path = basedir_path_out_images, half_name = half_name):
            return os.path.join(root_path, f"{item}_{half_name}")

        # image_names = list(map(lambda xx: f"{xx}_{half_name}", images_kind))
        image_names = list(map(full_path_out_images, images_kind))
        
        images_conf = ImagesConf._make(image_names)
    except Exception as err:
        print(str(err))
        raise Exception(f'Failed to read configuration from {path_conf_file} cofiguration file!')
    finally:
        print('config file read with success.')
        pass
    return images_conf