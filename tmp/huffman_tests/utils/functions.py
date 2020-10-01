  
# =============================================================================================== #
# Imports
# =============================================================================================== #

from PIL import  Image
from PIL import ImageChops

import os
import sys, string
import copy
import time

from collections import Counter
from itertools import chain

from pprint import pprint

# =============================================================================================== #
# Calculating Symbol Counts
# =============================================================================================== #

def count_symbols(image):
    pixels = image.getdata()

    if image.mode == 'L': values = pixels
    else: values = chain.from_iterable(pixels)
    
    counts = Counter(values).items()
    return sorted(counts, key=lambda x:x[::-1])

# =============================================================================================== #
# Building the Tree
# =============================================================================================== #

def build_tree(counts) :
    nodes = [entry[::-1] for entry in counts] # Reverse each (symbol,count) tuple
    while len(nodes) > 1 :
        leastTwo = tuple(nodes[0:2]) # get the 2 to combine
        theRest = nodes[2:] # all the others
        combFreq = leastTwo[0][0] + leastTwo[1][0]  # the branch points freq
        nodes = theRest + [(combFreq, leastTwo)] # add branch point to the end
        nodes.sort(key = lambda x: x[0]) # sort it into place
    return nodes[0] # Return the single tree inside the list

# =============================================================================================== #
# Trimming the Tree
# =============================================================================================== #

def trim_tree(tree) :
    p = tree[1] # Ignore freq count in [0]
    if type(p) is tuple: # Node, trim left then right and recombine
        return (trim_tree(p[0]), trim_tree(p[1]))
    return p # Leaf, just return it

# =============================================================================================== #
# Assigning Codes
# =============================================================================================== #

def assign_codes_impl(codes, node, pat):
    if type(node) == tuple:
        assign_codes_impl(codes, node[0], pat + [0]) # Branch point. Do the left branch
        assign_codes_impl(codes, node[1], pat + [1]) # then do the right branch.
    else:
        codes[node] = pat # A leaf. set its code


def assign_codes(tree):
    codes = {}
    assign_codes_impl(codes, tree, [])
    return codes

# =============================================================================================== #
# Encoding
# =============================================================================================== #

def to_binary_list(n):
    """Convert integer into a list of bits"""
    return [n] if (n <= 1) else to_binary_list(n >> 1) + [n & 1]


def from_binary_list(bits):
    """Convert list of bits into an integer"""
    result = 0
    for bit in bits:
        result = (result << 1) | bit
    return result


def pad_bits(bits, n):
    """Prefix list of bits with enough zeros to reach n digits"""
    assert(n >= len(bits))
    return ([0] * (n - len(bits)) + bits)


def raw_size(width, height, channels = 3):
    header_size = 2 * 16 # height and width as 16 bit values
    pixels_size = channels * 8 * width * height # RGB 3 channels | Grayscale 1 channel, 8 bits per channel
    return (header_size + pixels_size) / 8
