# ============================================= #
# Standard Imports
# ============================================= #
from __future__ import print_function
from __future__ import division

from collections import  Counter
from pprint import pprint
from PIL import Image

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

from huffman_enc import encode_file, decode_file, crea_huffman_tree_from_file, show_encoding

parser = argparse.ArgumentParser(description='Custom Huffman Encoding Script')
parser.add_argument('--input-file', type=str, dest="input_file",
                    help='Location of input resource to be compressed, from local file system.')
parser.add_argument('--output-dest', type=str, dest="output_file",
                    help='Location of output compressed result, into local file system.')


def main(args):

    input_filename = args.input_file
    output_filename = args.output_file

    if os.path.exists(input_filename) is False:
        print(f"Error: {input_filename} does not exists!", file=sys.stderr)
        return -1
    if os.path.isfile(input_filename) is False:
        print(f"Error: {input_filename} is not a file!", file=sys.stderr)
        return -1
    
    start = time.time()
    tree = crea_huffman_tree_from_file(input_filename)

    # pprint(tree)
    print('Symbol --> Code')
    print('-' * 40)
    show_encoding(tree)
    print()

    print('Encoded File')
    print('-' * 40)
    encode_file(tree, input_filename, output_filename)
    print()
    
    print('Dencoded File')
    print('-' * 40)
    decode_file(tree, output_filename)
    print()

    stop = time.time()
    times = (stop - start) * 1000
    print('-' * 40)
    stop = time.time()
    # Display Enc/Dec elapsed time
    print('Run time takes %d miliseconds' % times)
    
    return 0

if __name__ == "__main__":

    # Parse input arguments
    args, unknown = parser.parse_known_args()

    exit_code = main(args)
    sys.exit(exit_code)
    pass