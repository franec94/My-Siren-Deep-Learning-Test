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
        tmp_words = a_line.split(" ")
        for a_word in tmp_words:
            if len(a_word) == 0: continue
            words.setdefault(a_word, 0)
            words[a_word] += 1

    sorted_words = dict(sorted(words.items(), key = lambda x: x[1], reverse = True))
    return sorted_words


def get_code(tree, item):
    if tree['l'] == None and tree['r'] == None:
        if tree['item'] == item: return None, True
        else: return None, False

    if tree['l'] != None:
        code, res = get_code(tree['l'], item)
        if res == True:
            if code is None:
                return "0", True
            else:
                return "0" + code, True
    if tree['r'] != None:
        code, res = get_code(tree['r'], item)
        if res == True:
            if code is None:
                return "1", True
            else:
                return "1" + code, True
    return None, False


def create_node(item):
    return {
        'item': item[0],
        'freq': item[1],
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
            el0 = p_queue.pop()
            el1 = p_queue.pop()

            freq = el0['freq'] + el1['freq']
            a_node = create_node((None, freq))
            a_node['l'] = el0
            a_node['r'] = el1

            flag_added = False
            for ii in range(len(p_queue)):
                curr_node = p_queue[ii]
                if curr_node['freq'] >= freq:
                    p_queue.insert(ii - 1, a_node)
                    flag_added = True
                    break
            if flag_added is False:
                p_queue.append(a_node)
        else:
            break
    return p_queue[0]
def main(args):

    input_filename = args.input_file
    output_filename = args.output_file

    if os.path.exists(input_filename) is False:
        print(f"Error: {input_filename} does not exists!", file=sys.stderr)
        return -1
    if os.path.isfile(input_filename) is False:
        print(f"Error: {input_filename} is not a file!", file=sys.stderr)
        return -1
    
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
    
    return 0

if __name__ == "__main__":

    # Parse input arguments
    args, unknown = parser.parse_known_args()

    exit_code = main(args)
    sys.exit(exit_code)
    pass