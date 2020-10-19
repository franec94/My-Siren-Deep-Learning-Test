from datetime import datetime
# from google.colab import files

from pathlib import Path
from collections import namedtuple
from io import BytesIO
from pprint import pprint

# import psycopg2 as ps
import matplotlib.pyplot as plt
plt.style.use('dark_background')
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
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def calculate_several_jpeg_compression(image, qualities, left, top, right, bottom):
    # Named tuple for creating a record related to
    # a trial for compressing the target image.
    name_obj = 'WeightsPsnr'
    attributes = ['psnr', 'quality', 'file_size_bits', 'bpp', 'width', 'heigth', 'CR']
    WeightsPsnr = collections.namedtuple(name_obj, attributes) 

    # List used to save results and keep trace of failures, if any.
    result_tuples = []
    failure_qualities = []

    # Gather results.
    for edges in [(left, top, right, bottom)]: # for edges in edges_list:
    
        # Firstly crop image to desired shape.    
        left, top, right, bottom = list(map(int, edges))
        im_tmp = image.crop((left, top, right, bottom))
    
        # Get size cropped image
        cropped_file_size_bits = None
        with BytesIO() as f:
            im_tmp.save(f, format='PNG')
            cropped_file_size_bits = f.getbuffer().nbytes * 8
            pass
    
        # Then test the effect of several different quality values
        # used in compression transform.
        for quality in qualities:
            try:
                # Convert to JPEG specifying quality of compression.
                with BytesIO() as f:
                    # im_tmp.save(f'myimg.jpg', quality = int(quality))
                    # im_jpeg = Image.open('myimg.jpg')
                
                    im_tmp.save(f, format='JPEG', quality = int(quality))
                    f.seek(0)
                    im_jpeg = Image.open(f)
                    assert im_jpeg.size == im_tmp.size, "im_jpeg.size != im_tmp.size"
    
                    # Calculate quantities to be stored for this trial
            
                    # data used for deriving scores
                    width, height = im_jpeg.size[0], im_jpeg.size[1]
                    pixels = width * height
                    # compressed_file_size_bits = Path('myimg.jpg').stat().st_size * 8
                    compressed_file_size_bits = f.getbuffer().nbytes * 8
            
                    # Scores
                    bpp = compressed_file_size_bits / pixels    
                    psnr_score = psnr(np.asarray(im_tmp), np.asarray(im_jpeg), data_range=255)
                    CR = cropped_file_size_bits / compressed_file_size_bits
            
                    # Store results into a list
                    values = [psnr_score, quality, compressed_file_size_bits, bpp, width, height, CR]
                    result_tuples.append(WeightsPsnr._make(values))
                    pass
            except Exception as err:
                # Keep track of unaccepted quality values for compressing the image
                print(err)
                failure_qualities.append(quality)
            pass
        pass
    return result_tuples, failure_qualities