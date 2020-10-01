# ============================================= #
# Standard Imports
# ============================================= #
from __future__ import print_function
from __future__ import division

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


parser = argparse.ArgumentParser(description='Custom Huffman Encoding Script')
parser.add_argument('--input-file', type=str, dest="input_file",
                    help='Location of input resource to be compressed, from local file system.')
parser.add_argument('--output-dest', type=str, dest="output_file",
                    help='Location of output compressed result, into local file system.')


def get_stats_words_sorted(lines):
    words = dict()
    for a_line in lines:
        tmp_words = list(a_line)
        for a_word in tmp_words:
            if len(a_word) == 0: continue
            words.setdefault(a_word, 0)
            words[a_word] += 1

    sorted_words = dict(sorted(words.items(), key = lambda x: x[1], reverse = False))
    return sorted_words


def create_node(item):
    return {
        'item': item[0],
        'freq': item[1],
        'code': None,
        'l': None,
        'r': None
    }


def create_pqueue(a_dict):
    p_queue = list(map(create_node, a_dict.items()))

    pprint(p_queue)
    return p_queue

def build_tree(p_queue):
    while len(p_queue) != 1:
        if len(p_queue) >= 2:
            el0 = p_queue[0]
            el1 = p_queue[1]

            p_queue.remove(el0)
            p_queue.remove(el1)

            el0['code'] = "0"
            el1['code'] = "1"

            a_node = create_node((None, None))

            freq = el0['freq'] + el1['freq']
            a_node['freq'] = freq

            a_node['l'] = el0
            a_node['r'] = el1

            flag_added = False
            for ii in range(len(p_queue)):
                curr_node = p_queue[ii]
                if curr_node['freq'] >= freq:
                    p_queue.insert(ii, a_node)
                    flag_added = True
                    break
            if flag_added is False:
                p_queue.append(a_node)
        else:
            break
    return p_queue[0]


def get_code(tree, item):
    if tree['l'] == None and tree['r'] == None:
        if tree['item'] == item: return tree['code']
        else: return None

    if tree['l'] != None:
        code = get_code(tree['l'], item)
        if code != None:
            if tree['code'] != None:
                return tree['code'] + code
            else:
                return code
    if tree['r'] != None:
        code = get_code(tree['r'], item)
        if code != None:
            if tree['code'] != None:
                return tree['code'] + code
            else:
                return code
    return None

def compress_file(input_filename, output_filename):
    lines = None
    with open(input_filename, "r") as f:
        lines = f.read().split("\n")

    sorted_words = get_stats_words_sorted(lines)
    
    p_queue = create_pqueue(sorted_words)
    tree = build_tree(p_queue)

    pprint(tree)

    all_words = list(sorted_words.keys())
        
    for a_word in all_words:
        a_code = get_code(tree, a_word)
        print(a_word, "--->", a_code)
    with open(input_filename, "r") as fin:
        with open(output_filename, "w") as fout:
            for a_line in fin.read().split("\n"):
                for a_word in a_line:
                    a_code = get_code(tree, a_word)
                    print(a_code, sep="", end="", file=fout)
                    pass
                pass
            print("", file=fout)
    return


def main(args):

    input_filename = args.input_file
    output_filename = args.output_file

    if os.path.exists(input_filename) is False:
        print(f"Error: {input_filename} does not exists!", file=sys.stderr)
        return -1
    if os.path.isfile(input_filename) is False:
        print(f"Error: {input_filename} is not a file!", file=sys.stderr)
        return -1
    
    compress_file(input_filename, output_filename)
    
    return 0

if __name__ == "__main__":

    # Parse input arguments
    args, unknown = parser.parse_known_args()

    exit_code = main(args)
    sys.exit(exit_code)
    pass