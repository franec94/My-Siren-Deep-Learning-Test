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

from utils.functions import raw_size, count_symbols, build_tree, trim_tree, assign_codes, pad_bits, to_binary_list
from utils.classes import OutputBitStream

def compressed_size(counts, codes):
    header_size = 2 * 16 # height and width as 16 bit values

    tree_size = len(counts) * (1 + 8) # Leafs: 1 bit flag, 8 bit symbol each
    tree_size += len(counts) - 1 # Nodes: 1 bit flag each
    if tree_size % 8 > 0: # Padding to next full byte
        tree_size += 8 - (tree_size % 8)

    # Sum for each symbol of count * code length
    pixels_size = sum([count * len(codes[symbol]) for symbol, count in counts])
    if pixels_size % 8 > 0: # Padding to next full byte
        pixels_size += 8 - (pixels_size % 8)

    return (header_size + tree_size + pixels_size) / 8


def encode_header(image, bitstream):
    height_bits = pad_bits(to_binary_list(image.height), 16)
    bitstream.write_bits(height_bits)    
    width_bits = pad_bits(to_binary_list(image.width), 16)
    bitstream.write_bits(width_bits)
    pass


def encode_tree(tree, bitstream):
    if type(tree) == tuple: # Note - write 0 and encode children
        bitstream.write_bit(0)
        encode_tree(tree[0], bitstream)
        encode_tree(tree[1], bitstream)
    else: # Leaf - write 1, followed by 8 bit symbol
        bitstream.write_bit(1)
        symbol_bits = pad_bits(to_binary_list(tree), 8)
        bitstream.write_bits(symbol_bits)
        pass
    pass


def encode_pixels(image, codes, bitstream):
    for pixel in image.getdata():
        if image.mode == 'L':
            bitstream.write_bits(codes[pixel])
        else: 
            for value in pixel:
                bitstream.write_bits(codes[value])
                pass
            pass
        pass
    pass

def compress_image(in_file_name, out_file_name, verbose = 0):
    if verbose > 0:
        print('Compressing "%s" -> "%s"' % (in_file_name, out_file_name))
        pass
    
    image = Image.open(in_file_name)
    if verbose > 0:    
        print('Image shape: (height=%d, width=%d)' % (image.height, image.width))
        print('Image mode: %s' % (image.mode))
        pass
    
    if image.mode == 'L': channels = 1
    else: channels = 3
    size_raw = raw_size(image.height, image.width, channels = channels)
    if verbose > 0:
        print('RAW image size: %d bytes' % size_raw)
        pass

    counts = count_symbols(image)
    if verbose == 2:
        print('Counts:')
        pprint(counts)
        pass

    tree = build_tree(counts)
    if verbose == 2:
        print('Tree:')
        print(str(tree))
        pass

    trimmed_tree = trim_tree(tree)
    if verbose == 2:
        print('Trimmed tree:')
        print(str(trimmed_tree))
        pass

    
    codes = assign_codes(trimmed_tree)
    if verbose == 2:
        print('Codes:')
        print(codes)
        pass

    size_estimate = compressed_size(counts, codes)
    if verbose == 2:
        print('Estimated size: %d bytes' % size_estimate)


    if verbose == 2:
        print('Writing...')
    stream = OutputBitStream(out_file_name)
    if verbose == 2:
        print('* Header offset: %d' % stream.bytes_written)
    encode_header(image, stream)
    stream.flush() # Ensure next chunk is byte-aligned
    if verbose == 2:
        print('* Tree offset: %d' % stream.bytes_written)
    encode_tree(trimmed_tree, stream)
    stream.flush() # Ensure next chunk is byte-aligned
    if verbose == 2:
        print('* Pixel offset: %d' % stream.bytes_written)
    encode_pixels(image, codes, stream)
    stream.close()

    size_real = stream.bytes_written
    if verbose == 2:
        print('Wrote %d bytes.' % size_real)
    if verbose == 2:
        print('Estimate is %scorrect.' % ('' if size_estimate == size_real else 'in'))
        print('Compression ratio: %0.2f' % (float(size_raw) / size_real))
    
    return float(size_raw) / size_real