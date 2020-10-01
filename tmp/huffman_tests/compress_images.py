# ============================================= #
# Standard Imports
# ============================================= #
from __future__ import print_function
from __future__ import division

from collections import  Counter
from pprint import pprint
from PIL import  Image
from PIL import ImageChops

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sts

import argparse
import copy
import datetime
import os
import random
import skimage
import shutil
import sys
import time
# import visdom
import warnings

from utils.compression import compress_image
from utils.decompression import decompress_image

parser = argparse.ArgumentParser(description='Custom Huffman Encoding Script')
parser.add_argument('--input-dir', type=str, dest="input_dir",
                    help='Location of input resource to be compressed, from local file system.')
parser.add_argument('--output-dest', type=str, dest="output_file",
                    help='Location of output compressed result stats, into local file system.')
parser.add_argument('--channels', type=str, dest="channels", choices=["RGB", "L"],
                    help='Type Channels of input Image.')

# =============================================================================================== #
# Util Function
# =============================================================================================== #

def images_equal(file_name_a, file_name_b):
    image_a = Image.open(file_name_a)
    image_b = Image.open(file_name_b)

    diff = ImageChops.difference(image_a, image_b)

    return diff.getbbox() is None

def process_a_image(a_image, verbose = 0):
    start = time.time()

    # Compress_image
    compression_ratio = compress_image(a_image, 'answer.txt')

    if verbose > 0:
        print('-' * 40)

    stop = time.time()
    times = (stop - start) * 1000
    if verbose > 0:
        print('-' * 40)

        # Display Enc/Dec elapsed time
        print('Run time takes %d miliseconds' % times)
    return compression_ratio


def show_hist_kde_compression_ratio(filename, title, samples, bins = 10):
    n = len(samples)
    
    h, e = np.histogram(samples, bins=bins, density=True)
    x = np.linspace(e.min(), e.max())

    # plot the histogram
    plt.figure(figsize=(8,6))
    plt.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label='histogram')

    # plot the real KDE
    kde = sts.gaussian_kde(samples)
    plt.plot(x, kde.pdf(x), c='C1', lw=8, label='KDE')

    # resample the histogram and find the KDE.
    resamples = np.random.choice((e[:-1] + e[1:])/2, size=n*5, p=h/h.sum())
    rkde = sts.gaussian_kde(resamples)

    # plot the KDE
    plt.plot(x, rkde.pdf(x), '--', c='C3', lw=4, label='resampled KDE')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    pass


# =============================================================================================== #
# Main Function
# =============================================================================================== #

def main(args):

    in_dir_path = args.input_dir
    out_file_name = args.output_file

    if os.path.exists(in_dir_path) is False:
        print(f"Error: {in_dir_path} does not exists!", file=sys.stderr)
        return -1
    if os.path.isdir(in_dir_path) is False:
        print(f"Error: {in_dir_path} is not a file!", file=sys.stderr)
        return -1

    images = list()
    bsd68_path = in_dir_path
    for (dirpath, dirnames, filenames) in os.walk(bsd68_path):
        full_file_path_list = list()
        for a_file in sorted(filenames):
            full_file_path_list.append(os.path.join(dirpath, a_file))
            pass
        images.extend(full_file_path_list)
        pass

    samples = list()
    for _, a_image in enumerate(images[:]):
        compression_ratio = process_a_image(a_image)
        samples.append(compression_ratio)
        pass

    filename = 'compression-ratio.png'
    title = 'Compression Ratio: n = %d samples' % len(images[:])
    samples = np.array(samples, dtype=np.float)
    show_hist_kde_compression_ratio(filename, title, samples, bins = 10)

    df = pd.DataFrame(data = samples[:, np.newaxis], columns = ["Compression Ratio"])

    filename = 'compression-ratio-boxplot.png'
    boxplot = df.boxplot(column = ['Compression Ratio'])
    plt.savefig(filename)
    plt.show()

    full_stats_path = os.path.join(out_file_name, "compression-ratio.csv")
    df.to_csv(full_stats_path)
    
    return 0

# =============================================================================================== #
# Entry Point
# =============================================================================================== #

if __name__ == "__main__":

    # Parse input arguments
    args, unknown = parser.parse_known_args()

    exit_code = main(args)
    sys.exit(exit_code)
    pass