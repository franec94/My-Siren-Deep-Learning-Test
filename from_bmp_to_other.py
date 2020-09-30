from PIL import  Image
from PIL import ImageChops

import argparse
import glob
import os
import shutil
import sys
import string
import copy
import time

from collections import Counter
from itertools import chain

from pprint import pprint


def get_argparser_for_coverting():
    parser = argparse.ArgumentParser(description='Command ine util, based on python3 for converting files from .bmp to other image format.')
    parser.add_argument('--input-files-path', type=str, dest="input_files_path",
                    help='path to input filens.')
    parser.add_argument('--output-path', type=str, default=".", dest="output_path",
                    help='output location of resulting decoded file.')
    parser.add_argument('--extension', type=str, default="png", dest="extension", choices=["png", "jpg"],
                    help='Extension to convert for (default: png), allowed [png|jpg].')
    return parser


def conver_bmp_image(image_file, dest_path, extension):
    # image = Image.open('lenagray.bmp')
    # image.show()
    
    file_name = os.path.basename(image_file)[:-4]

    full_name = os.path.join(dest_path, f"{file_name}.{extension}")
    Image.open(image_file).save(full_name)
    
    return file_name, full_name


def main(args):

    # Get input and output locations for src and dst files storage
    src_path_files = args.input_files_path
    dst_path_files = args.output_path

    # Check input files path
    try:
        if os.path.exists(src_path_files) == False:
            print(f"Error: {src_path_files} does not exists", file=sys.stderr)
            sys.exit(-1)
        elif os.path.isdir(src_path_files) == False:
            print(f"Error: {src_path_files} is not a dir path", file=sys.stderr)
            sys.exit(-1)
    except Exception as err:
        print(str(err))
        sys.exit(-1)
        pass
    
    # Create if not exists dest location for resulting files
    try:
        os.makedirs(dst_path_files)
    except WindowsError as err:
        pass
    except Exception as err:
        print(str(err))
        sys.exit(-1)
        pass

    result_files_list = os.listdir(src_path_files) # pprint(result)

    result_files_list = [ os.path.join(src_path_files, xi) for xi in result_files_list ]
    result_files_list = [ xi for xi in result_files_list if os.path.isfile(xi)]
    result_files_bmp = [ xi for xi in result_files_list if xi.endswith(".bmp") ]

    pprint(result_files_bmp)

    for _, a_file in enumerate(result_files_bmp):
        file_name, full_name = conver_bmp_image(a_file, dst_path_files, args.extension)
        print(file_name, "-->", full_name)
        pass

    return 0

if __name__ == "__main__":
    
    # Parse input arguments
    parser = get_argparser_for_coverting()
    args, unknown = parser.parse_known_args()

    pprint(args)
    print("-" * 40)
    pprint(unknown)
    print("-" * 40)
    print()

    exit_code = main(args)
    sys.exit(exit_code)
    pass