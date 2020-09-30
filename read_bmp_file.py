import os
import struct
import sys
import time

import imghdr

from PIL import Image


if __name__ == "__main__":

    input_filename = "lenagray.bmp"

    f = open(input_filename,"rb")
    data = bytearray(f.read())
    f.close()

    print(data[:14])
    print(data[:14].decode())
    print(imghdr.what(input_filename))
    

    bmp = open(input_filename, 'rb')
    print('Type:', bmp.read(2).decode())
    print('Size: %s' % struct.unpack('I', bmp.read(4)))
    print('Reserved 1: %s' % struct.unpack('H', bmp.read(2)))
    print('Reserved 2: %s' % struct.unpack('H', bmp.read(2)))
    print('Offset: %s' % struct.unpack('I', bmp.read(4)))

    print('DIB Header Size: %s' % struct.unpack('I', bmp.read(4)))
    print('Width: %s' % struct.unpack('I', bmp.read(4)))
    print('Height: %s' % struct.unpack('I', bmp.read(4)))
    print('Colour Planes: %s' % struct.unpack('H', bmp.read(2)))
    print('Bits per Pixel: %s' % struct.unpack('H', bmp.read(2)))
    print('Compression Method: %s' % struct.unpack('I', bmp.read(4)))
    print('Raw Image Size: %s' % struct.unpack('I', bmp.read(4)))
    print('Horizontal Resolution: %s' % struct.unpack('I', bmp.read(4)))
    print('Vertical Resolution: %s' % struct.unpack('I', bmp.read(4)))
    print('Number of Colours: %s' % struct.unpack('I', bmp.read(4)))
    print('Important Colours: %s' % struct.unpack('I', bmp.read(4)))

    bmp.close()
    pass