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


from utils.functions import raw_size, count_symbols, build_tree, trim_tree, assign_codes, pad_bits, from_binary_list
from utils.classes import InputBitStream

# =============================================================================================== #
# Decompression
# =============================================================================================== #

def decode_header(bitstream):
    height = from_binary_list(bitstream.read_bits(16))
    width = from_binary_list(bitstream.read_bits(16))
    return (height, width)


# https://stackoverflow.com/a/759766/3962537
def decode_tree(bitstream):
    flag = bitstream.read_bits(1)[0]
    if flag == 1: # Leaf, read and return symbol
        return from_binary_list(bitstream.read_bits(8))
    left = decode_tree(bitstream)
    right = decode_tree(bitstream)
    return (left, right)


def decode_value(tree, bitstream):
    bit = bitstream.read_bits(1)[0]
    node = tree[bit]
    if type(node) == tuple:
        return decode_value(node, bitstream)
    return node


def decode_pixels(height, width, channels, tree, bitstream):
    pixels = bytearray()
    for i in range(height * width * channels):
        pixels.append(decode_value(tree, bitstream))
    if channels == 3:
        return Image.frombytes('RGB', (width, height), bytes(pixels))
    else:
        return Image.frombytes('L', (width, height), bytes(pixels))


def decompress_image(in_file_name, out_file_name, channels):
    if channels == 'L': channels = 1
    else: channels = 3

    print('Decompressing "%s" -> "%s"' % (in_file_name, out_file_name))

    print('Reading...')
    stream = InputBitStream(in_file_name)
    print('* Header offset: %d' % stream.bytes_read)
    height, width = decode_header(stream)
    stream.flush() # Ensure next chunk is byte-aligned
    print('* Tree offset: %d' % stream.bytes_read)    
    trimmed_tree = decode_tree(stream)
    stream.flush() # Ensure next chunk is byte-aligned
    print('* Pixel offset: %d' % stream.bytes_read)
    
    image = decode_pixels(height, width, channels, trimmed_tree, stream)
    stream.close()
    print('Read %d bytes.' % stream.bytes_read)

    print('Image size: (height=%d, width=%d)' % (height, width))
    print('Trimmed tree: %s' % str(trimmed_tree))
    image.save(out_file_name)
    pass