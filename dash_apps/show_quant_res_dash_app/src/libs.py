from __future__ import print_function


from src.timestamps_runs import DIR_DATA_CSV_LIST

NOTIFICATIONS_ENABLED_VIA_TOAST = True
SHOW_VISDOM_RESULTS = False
DARK_BACKGROUND_PLT = True
SHOW_RESULTS_BY_TABS = True
FLAG_FETCH_ALL_CSV_FILES = True
DASH_TEMPLATES_LIST = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]

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
import argparse
import contextlib
import collections
import copy
import datetime
import functools
import glob
import itertools
import json
import operator
import os
import pathlib
import pickle
import re
import sqlite3
import shutil
import sys
import win10toast
import time
import yaml

# Plotly imports.
# ----------------------------------------------- #
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.express as px

# Dash imports.
# ----------------------------------------------- #
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# FastAi imports.
# ----------------------------------------------- #
from fastcore.foundation import *
from fastcore.meta import *
from fastcore.utils import *
from fastcore.test import *
from nbdev.showdoc import *
from fastcore.dispatch import typedispatch
from functools import partial
import inspect

from fastcore.imports import in_notebook, in_colab, in_ipython

# Constraint imports.
# ----------------------------------------------- #
if in_colab():
    from google.colab import files
    pass

if  (in_notebook() or in_ipython()) and SHOW_VISDOM_RESULTS:
    import visdom
    pass

if in_colab() or in_notebook() or in_colab():
    # Back end of ipywidgets.
    from IPython.display import display    
    import ipywidgets as widgets
    pass

# Data Scienc & Machine Learning main imports.
# ----------------------------------------------- #
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if DARK_BACKGROUND_PLT:
    plt.style.use('dark_background')
    pass
import seaborn as sns # sns.set_theme(style="white") # sns.set(style="whitegrid", color_codes=True)
# sns.set(style="darkgrid", color_codes=True)

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

from sklearn.model_selection import ParameterGrid

# ----------------------------------------------- #
# Custom Imports
# ----------------------------------------------- #

from src.functions import check_dir_exists, get_all_files_by_ext, laod_data_from_files_list
from src.functions import read_conf_file, load_target_image, get_dict_dataframes, get_dataframe
from src.make_graphics import compare_compressions
from src.work import fetch_data, fetch_data_by_status, fetch_data_by_constraints, get_info_from_logged_parser, insert_data_read_from_logs
from src.handle_server_connection import get_data_from_db, get_data_from_db_by_status, get_constraints_for_query_db
from src.db_tables import TableRunsDetailsClass, TableRunsDetailsTupleClass

from src.handle_dataframes import prepare_and_merge_target_dfs, calculate_several_jpeg_compression, get_cropped_by_center_image
from src.custom_dash_app import get_dash_app
