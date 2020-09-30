# https://stackoverflow.com/questions/58894191/image-compression-in-python

from PIL import  Image
from PIL import ImageChops

import os
import sys, string
import copy
import time

from collections import Counter
from itertools import chain

from pprint import pprint


from utils.compression import compress_image
from utils.custom_argparser import get_custom_argparser
from utils.decompression import decompress_image


# =============================================================================================== #
# Util Function
# =============================================================================================== #


def images_equal(file_name_a, file_name_b):
    image_a = Image.open(file_name_a)
    image_b = Image.open(file_name_b)

    diff = ImageChops.difference(image_a, image_b)

    return diff.getbbox() is None

# =============================================================================================== #
# Main
# =============================================================================================== #

def main(args):

    file_name = args.input_file_path

    base_file_name, file_extension = os.path.splitext(file_name)
    file_out = os.path.join(args.output_path, f'flag_out' + file_extension)

    start = time.time()

    # Compress_image
    compress_image(file_name, 'answer.txt')
    
    print('-' * 40)

    # Decompress image
    decompress_image('answer.txt', file_out)

    stop = time.time()
    times = (stop - start) * 1000

    print('-' * 40)

    # Display Enc/Dec elapsed time
    print('Run time takes %d miliseconds' % times)
    print('Images equal = %s' % images_equal(file_name, file_out))

    return 0

# =============================================================================================== #
# Entry Point
# =============================================================================================== #

if __name__ == "__main__":

    # Parse input arguments
    parser = get_custom_argparser()
    args, unknown = parser.parse_known_args()

    print("-" * 40)
    pprint(args)
    print("-" * 40)
    pprint(unknown)

    print()

    # run main function
    exit_code = main(args)

    sys.exit(exit_code)
    pass
