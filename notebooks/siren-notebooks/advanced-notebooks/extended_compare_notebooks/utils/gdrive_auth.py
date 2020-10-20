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

from apiclient import discovery
from httplib2 import Http
import oauth2client
from oauth2client import file, client, tools

def make_grdive_auth(path_data = '.'):
    creds = None
    try:
        obj = lambda: None
        lmao = {"auth_host_name":'localhost', 'noauth_local_webserver':'store_true', 'auth_host_port':[8080, 8090], 'logging_level':'ERROR'}
        for k, v in lmao.items():
            setattr(obj, k, v)
    
        # authorization boilerplate code
        SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
        store = file.Storage(os.path.join(path_data, 'token.json'))
        creds = store.get()
        # The following will give you a link if token.json does not exist, the link allows the user to give this app permission
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets(os.path.join(path_data,'client_id.json'), SCOPES)
            creds = tools.run_flow(flow, store, obj)
            pass
    except Exception as err:
        print(str(err))
        raise Exception('Auth to gdrive failed!')
    finally:
        print('Gdrive auth done.')
        pass
    return creds
    

def fetch_data_from_gdrive(creds, path_history_train, file_id, fetch_data_from_gdrive_checkbox):
    try:
        if fetch_data_from_gdrive_checkbox:
            if os.path.exists(f'{path_history_train}') is False:
                DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http()))
                request = DRIVE.files().get_media(fileId=file_id)

                # replace the filename and extension in the first field below
                # fh = io.FileIO(f'filename.zip', mode='w')
                fh = io.FileIO(f'{path_history_train}', mode='w')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print("Download %d%%." % int(status.progress() * 100))
                    pass
                pass
            else:
                print(f"Already exists: {path_history_train}")
                pass
        else:
            print(f"No data retrieved from gdrive.")
            pass
    except Exception as err:
        print(str(err))
        raise Exception('Retrieveing data from gdrive failed.')
    finally:
        pass
    pass
