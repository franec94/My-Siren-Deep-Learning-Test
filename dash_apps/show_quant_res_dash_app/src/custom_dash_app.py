from __future__ import print_function

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
import copy
import datetime
import functools
import glob
import itertools
import json
import operator
import os
import re
import sqlite3
import sys
import time
import yaml

# Dash imports.
# ----------------------------------------------- #
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from src.libs import SHOW_RESULTS_BY_TABS

def get_dash_app(figs_list, n_figs, tab_names_list):
    app = dash.Dash('Siren+Jpeg Results', external_stylesheets=[dbc.themes.DARKLY])

    if SHOW_RESULTS_BY_TABS:
        tab_names = iter(tab_names_list)
        tab_list = [] # card_list = []
        
        # n = len(figs_list) // n_figs
        for start_pos, end_pos in zip(range(0, len(figs_list), n_figs), range(n_figs, len(figs_list)+1, n_figs)):
            tab_list.append(dbc.Tab(dbc.Card(figs_list[start_pos:end_pos], body=True), label=f'{next(tab_names)}'))
        # tab_list.append(dbc.Tab(dbc.Card(figs_list[start_pos:end_pos], body=True), label=f'{next(tab_names)}'))
        app.layout = dbc.Tabs(tab_list, id="tabs-with-classes")
    else:
        app.layout = html.Div(figs_list)
        pass
    return app
