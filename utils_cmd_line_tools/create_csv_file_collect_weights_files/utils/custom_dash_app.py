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

from utils.libs import SHOW_RESULTS_BY_TABS

def get_dash_app(figs_list, figs_list_2, n_figs):
    opts = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    opts_list= list(map(lambda item: dict(label=dict,value=dict), opts))

    app = dash.Dash('Siren+Jpeg Results', external_stylesheets=[dbc.themes.DARKLY])
    value_theme = None
    value_old = 'plotly_dark'

    if SHOW_RESULTS_BY_TABS:
        tab_list = []; card_list = None
        tab_names = iter(['scatter-mereged (PSNR,SSIM, CR)', 'box-mereged (PSNR,SSIM, CR)', 'kde-mereged (PSNR,SSIM, CR)', 'mse-siren (SCATTER,BOX,KDE)', 'summary']) # , 'graphics options'])
        for ii, a_fig in enumerate(figs_list):
            if ii % n_figs == 0:
                if card_list != None:
                    tab_list.append(dbc.Tab(dbc.Card(card_list, body=True), label=f'{next(tab_names)}'))
                    pass
                card_list = []
                pass
            card_list.append(a_fig)
            pass

        tab_list.append(dbc.Tab(dbc.Card(card_list, body=True), label=f'{next(tab_names)}'))
        
        tab_list.append(dbc.Tab(dbc.Card(figs_list_2, body=True), label=f'{next(tab_names)}'))
        tab_list.append(dbc.Tab(dbc.Card(figs_list + figs_list_2, body=True), label=f'{next(tab_names)}'))
    
        """
        opts = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
        opts_list = list(map(lambda item: dict(label=item, value=item), opts))
        tab_list.append(dbc.Tab(dbc.Card(
            [
                html.Div([
                dcc.Dropdown(
                    id='demo-dropdown',
                        options=opts_list,
            value='plotly_dark'
        )]),
        html.Div(id='dd-output-container')
            ]
            , body=True), label=f'{next(tab_names)}'))
        app.layout = dbc.Tabs(children=tab_list, id="tabs-with-classes")
    
        @app.callback(
            dash.dependencies.Output('dd-output-container', 'children'),
            [dash.dependencies.Input('demo-dropdown', 'value')])
        def update_output(value):
            value_theme = value
            return 'You have selected "{}"'.format(value)
        """
        app.layout = dbc.Tabs(tab_list, id="tabs-with-classes")
    else:
        app.layout = html.Div(figs_list)
        pass
    return app
