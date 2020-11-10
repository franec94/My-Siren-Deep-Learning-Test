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

# Data Scienc & Machine Learning main imports.
# ----------------------------------------------- #
import scipy
import numpy as np
import pandas as pd

# skimage
# ----------------------------------------------- #
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def get_new_targets(target, size):
    offset = target // 2
    if target % 2 == 0:
        extreme_1 = size // 2
        residual = 0
    else:
        extreme_1 = size // 2 - 1
        residual = 1
        pass
    extreme_2 = size // 2
    return extreme_1 - offset + residual, extreme_2 + offset + residual

def get_cropped_by_center_image(im, target = 25):
    width, height = im.size

    if isinstance(target, int):
        target = (target, target)
        pass

    left, right = get_new_targets(target[0], width)
    top, bottom = get_new_targets(target[1], height)

    # print(im.crop((left, top, right, bottom)).size)
    # print((left, top, right, bottom))

    im_cropped = im.crop((left, top, right, bottom))
    return im_cropped

def calculate_several_jpeg_compression(image, image_dim_bits, qualities):
    # Named tuple for creating a record related to
    # a trial for compressing the target image.
    typename = 'WeightsPsnr'
    fields_name = ['mse', 'psnr', 'ssim', 'quality', 'file_size_bits', 'bpp', 'width', 'heigth', 'CR']
    WeightsPsnr = collections.namedtuple(typename, fields_name) 

    # List used to save results and keep trace of failures, if any.
    result_tuples = []
    failure_qualities = []

    # Then test the effect of several different quality values
    # used in compression transform.
    for quality in qualities:
        try:
            # Convert to JPEG specifying quality of compression.
            with BytesIO() as f:
                # im_tmp.save(f'myimg.jpg', quality = int(quality))
                # im_jpeg = Image.open('myimg.jpg')
                
                image.save(f, format='JPEG', quality = int(quality))
                f.seek(0)
                compressed_file_size_bits = f.getbuffer().nbytes * 8
                im_jpeg = Image.open(f)
                assert im_jpeg.size == image.size, "im_jpeg.size != image.size"
    
                # Calculate quantities to be stored for this trial
            
                # data used for deriving scores
                width, height = im_jpeg.size[0], im_jpeg.size[1]
                pixels = width * height
                # compressed_file_size_bits = Path('myimg.jpg').stat().st_size * 8
                compressed_file_size_bits = f.getbuffer().nbytes * 8
            
                # Scores
                bpp = compressed_file_size_bits / pixels
                mse_score = mean_squared_error(np.asarray(image), np.asarray(im_jpeg))
                psnr_score = psnr(np.asarray(image), np.asarray(im_jpeg), data_range=255)
                ssim_score = ssim(np.asarray(image), np.asarray(im_jpeg), data_range=255)
                CR = image_dim_bits / compressed_file_size_bits
            
                # Store results into a list
                values = [mse_score, psnr_score, ssim_score, quality, compressed_file_size_bits, bpp, width, height, CR]
                result_tuples.append(WeightsPsnr._make(values))
        except Exception as err:
            # Keep track of unaccepted quality values for compressing the image
            print(str(err))
            failure_qualities.append(quality)
            pass
        pass
    return result_tuples, failure_qualities


def calculate_several_jpeg_compression_with_crops(image, qualities, cropping_list):
    # Named tuple for creating a record related to
    # a trial for compressing the target image.
    typename = 'WeightsPsnr'
    fields_name = ['psnr', 'ssim', 'quality', 'file_size_bits', 'bpp', 'width', 'heigth', 'CR']
    WeightsPsnr = collections.namedtuple(typename, fields_name) 

    # List used to save results and keep trace of failures, if any.
    result_tuples = []
    failure_qualities = []

    # Gather results.
    for edges in cropping_list: # for edges in edges_list:
    
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
                    ssim_score = ssim(np.asarray(im_tmp), np.asarray(im_jpeg), data_range=255)
                    CR = cropped_file_size_bits / compressed_file_size_bits
            
                    # Store results into a list
                    values = [psnr_score, ssim_score, quality, compressed_file_size_bits, bpp, width, height, CR]
                    result_tuples.append(WeightsPsnr._make(values))
                    pass
            except Exception as err:
                # Keep track of unaccepted quality values for compressing the image
                print(err)
                failure_qualities.append(quality)
            pass
        pass
    return result_tuples, failure_qualities

def get_ready_siren_df_to_merge(siren_df, image):
    with BytesIO() as f:
        image.save(f, format='PNG')
        file_size_bits = f.getbuffer().nbytes * 8
        pass
    # Define "BPP attribute" and add it to existing df.
    siren_df['bpp'] = siren_df['#params'].values * 32 / (image.size[0] * image.size[1])
    
    # Define "file_size_bits attribute" and add it to existing df.
    siren_df['file_size_bits'] = siren_df['#params'].values * 32
    # Define "CR attribute" and add it to existing df.
    siren_df['CR'] = file_size_bits / (siren_df['#params'].values * 32)
    
    # Define "Compression Label" and add it to existing df.
    hf_arr = siren_df['hf'].values.astype(dtype = np.int)
    create_label_lambda = lambda hf: f'siren-{hf}'
    siren_df['compression'] = list(map(create_label_lambda, hf_arr))
    
    return siren_df

def get_ready_jpeg_df_to_merge(jpeg_df):
    # Define "Compression Label" and add it to existing df.
    jpeg_df['compression'] = ['jpeg'] * jpeg_df.shape[0]
    return jpeg_df

def prepare_and_merge_target_dfs(siren_df, jpeg_df, *args, **kwargs):
    # pprint(kwargs)
    
    # Target image, either cropped or not, depending on the kind of run.
    image = kwargs['image']
    
    # Prepare siren_df for merging.
    siren_df = get_ready_siren_df_to_merge(siren_df, image=image)
    
    # Prepare jpeg_df for merging.
    jpeg_df = get_ready_jpeg_df_to_merge(jpeg_df)
    
    # Get columns to be merged and new columsn for
    # rename them after merging.
    siren_columns_for_merge = kwargs['siren_columns_for_merge']
    jpeg_columns_for_merge = kwargs['jpeg_columns_for_merge']
    columns_names_merge = kwargs['columns_names_merge']
    
    # Performe merging.
    data_frames_list = [
        siren_df[siren_columns_for_merge],
        jpeg_df[jpeg_columns_for_merge],
    ]
    merged_df = pd.concat(data_frames_list, names = columns_names_merge)
    return merged_df, siren_df, jpeg_df