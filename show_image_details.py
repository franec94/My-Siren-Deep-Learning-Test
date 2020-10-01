#!/usr/bin/env python3
# -*- enc:utf-8 -*-

# ============================================= #
# Standard Imports
# ============================================= #
from __future__ import print_function
from __future__ import division

from PIL import Image
from PIL.ExifTags import TAGS

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


# ============================================= #
# Torch Imports
# ============================================= #

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


parser = argparse.ArgumentParser(description="Show input Image main properties.")
parser.add_argument("--input-path", type=str, dest="input_path", help="Path to input image, local path.")
parser.add_argument("--seed", type=int, default=0, dest="seed", help="Seed value (default:0).")

# Define a method for printing image attributes
def printImageAttributes(imageObject, imagePath):

    # Retrieve the attributes of the image

    fileFormat      = imageObject.format        # Format of the image
    imageMode       = imageObject.mode          # Mode of the image
    imageSize       = imageObject.size          # Size of the image - tupe of (width, height)
    colorPalette    = imageObject.palette       # Palette used in the image

    # Print the attributes of the image
    print("Attributes of image:%s"%imagePath)

    print("The file format of the image is:%s"%fileFormat)

    print("The mode of the image is:%s"%imageMode)

    print("The size of the image is:width %d pixels,height %d pixels"%imageSize)

    print("Color palette used in image:%s"%colorPalette)

    print("Keys from image.info dictionary:%s")

    for key, value in imageObject.info.items() :
        print(key)
        print(value)
        pass
    pass


def main(args):

    file_path = args.input_path
    if os.path.exists(file_path) is False:
        print(f"Error: {file_path} does not exist!", file=sys.stderr)
        sys.exit(-1)
    if os.path.isfile(file_path) is False:
        print(f"Error: {file_path} is not a file!", file=sys.stderr)
        sys.exit(-1)
    
    # read the image data using PIL
    image = Image.open(file_path)
    # image.show()

    # Create an image object with color palette and show the image

    imageWithColorPalette = image.convert("P", palette=Image.ADAPTIVE, colors=8)
    imageWithColorPalette.show()
    
    # Print the attributes of the images
    print("-" * 40)
    printImageAttributes(image, file_path)
    print("-" * 40)
    printImageAttributes(imageWithColorPalette, file_path)
    
    sys.exit(0)
    
    # extract EXIF data
    exifdata = image.getexif()
    
    # iterating over all EXIF data fields
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes 
        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")
    return 0

if __name__ == "__main__":

    # Check desired Python's Libraries
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # Parse input arguments
    args, unknown = parser.parse_known_args()


    # Set seeds for experiments repeatability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    feature_extract = True
    
    # run main function
    exit_code = main(args)

    sys.exit(exit_code)
    pass

