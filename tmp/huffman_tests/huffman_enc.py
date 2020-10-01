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

# --------------------------------------------- #
# Utils
# --------------------------------------------- #

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

# --------------------------------------------- #
# Tree Section
# --------------------------------------------- #

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
    # pprint(p_queue)
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


def crea_huffman_tree_from_file(input_filename):
    lines = None
    with open(input_filename, "r") as f:
        lines = f.read().split("\n")

    sorted_words = get_stats_words_sorted(lines)
    
    p_queue = create_pqueue(sorted_words)
    tree = build_tree(p_queue)
    return tree


def pre_order_tree_traversal(tree, a_code):
    if tree['l'] == None and tree['r'] == None:
        print(tree['item'], "--->", a_code + tree['code'])

    if tree['code'] != None:
        tmp_code = a_code + tree["code"]
    else:
        tmp_code = a_code
    if tree['l'] != None:
        pre_order_tree_traversal(tree['l'], tmp_code)
    if tree['r'] != None:
        pre_order_tree_traversal(tree['r'], tmp_code)
    pass
    
    pass


def show_encoding(tree):
    pre_order_tree_traversal(tree, "")
    pass

# --------------------------------------------- #
# Encoding & Decoding Sections
# --------------------------------------------- #

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


def encode_file(tree, input_filename, output_filename):
    with open(input_filename, "r") as fin:
        with open(output_filename, "w") as fout:
            for a_line in fin.read().split("\n"):
                for a_word in a_line:
                    a_code = get_code(tree, a_word)
                    print(a_code, sep="", end="", file=fout)
                    pass
                pass
            print("", file=fout)
    pass


def decode_stream(a_stream, tree, pos):
    if tree['l'] == None and tree['r'] == None:
        return pos, tree['item']
    if a_stream[pos] == "0":
        return decode_stream(a_stream, tree['l'], pos + 1)
    return decode_stream(a_stream, tree['r'], pos + 1)


def decode_file(tree, input_filename):
    with open(input_filename, "r") as fin:
        for a_line in fin.read().split("\n"):
            print(a_line)
            pos = 0
            while len(a_line[pos:]) != 0:
                pos, symbol = decode_stream(a_line, tree, pos)
                print(symbol, sep="", end="")
                if pos > len(a_line): break
            print("")
    pass
