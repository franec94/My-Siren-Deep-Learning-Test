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
parser.add_argument('--input-file', type=str, dest="input_file",
                    help='Location of input resource to be compressed, from local file system.')
parser.add_argument('--output-dest', type=str, dest="output_file",
                    help='Location of output compressed result, into local file system.')
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


# =============================================================================================== #
# Main Function
# =============================================================================================== #

def main(args):

    in_file_name = args.input_file
    out_file_name = args.output_file

    if os.path.exists(in_file_name) is False:
        print(f"Error: {in_file_name} does not exists!", file=sys.stderr)
        return -1
    if os.path.isfile(in_file_name) is False:
        print(f"Error: {in_file_name} is not a file!", file=sys.stderr)
        return -1
    
    start = time.time()

    # Compress_image
    compress_image(in_file_name, 'answer.txt')

    print('-' * 40)

    # Decompress image
    decompress_image('answer.txt', out_file_name, channels = args.channels)

    stop = time.time()
    times = (stop - start) * 1000
    print('-' * 40)

    # Display Enc/Dec elapsed time
    print('Run time takes %d miliseconds' % times)
    print('Images equal = %s' % images_equal(in_file_name, out_file_name))
    
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