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
        tab_list = []; card_list = []
        # tab_names = iter(['scatter-mereged (PSNR,SSIM, CR)', 'box-mereged (PSNR,SSIM, CR)', 'kde-mereged (PSNR,SSIM, CR)', 'mse-siren (SCATTER,BOX,KDE)', 'summary']) # , 'graphics options'])
        tab_names = iter(tab_names_list)
        for ii, a_fig in enumerate(figs_list):
            """if ii % n_figs == 0:
                if card_list != None:
                    tab_list.append(dbc.Tab(dbc.Card(card_list, body=True), label=f'{next(tab_names)}'))
                    pass
                card_list = []
                pass
            card_list.append(a_fig)"""
            card_list.append(a_fig)
            pass
        # tab_list.append(dbc.Tab(dbc.Card(card_list, body=True), label=f'{next(tab_names)}'))
        tab_list.append(dbc.Tab(dbc.Card(card_list, body=True), label=f'{next(tab_names)}'))
        app.layout = dbc.Tabs(tab_list, id="tabs-with-classes")
    else:
        app.layout = html.Div(figs_list)
        pass
    return app
