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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash('Siren+Jpeg Results', external_stylesheets=[dbc.themes.DARKLY])
a_div = html.Div([
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montreal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='NYC'
    ),
    html.Div(id='dd-output-container')
])
a_div_2 = html.Div([
    dcc.Dropdown(
        id='demo-dropdown-2',
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montreal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='NYC'
    ),
    html.Div(id='dd-output-container-2')
])
a_tab = dbc.Tab(dbc.Card([a_div, a_div_2], body=True), label=f'Drpdown menu\'')
app.layout = a_tab

@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)
    pass