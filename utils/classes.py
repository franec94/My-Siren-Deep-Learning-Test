from PIL import  Image
from PIL import ImageChops

import os
import sys, string
import copy
import time

from collections import Counter
from itertools import chain

from pprint import pprint

from utils.functions import *

# =============================================================================================== #
# Output Bitstream
# =============================================================================================== #

class OutputBitStream(object): 
    def __init__(self, file_name): 
        self.file_name = file_name
        self.file = open(self.file_name, 'wb') 
        self.bytes_written = 0
        self.buffer = []
        pass

    def write_bit(self, value):
        self.write_bits([value])
        pass

    def write_bits(self, values):
        self.buffer += values
        while len(self.buffer) >= 8:
            self._save_byte()
            pass
        pass     

    def flush(self):
        if len(self.buffer) > 0: # Add trailing zeros to complete a byte and write it
            self.buffer += [0] * (8 - len(self.buffer))
            self._save_byte()
        assert(len(self.buffer) == 0)
        pass

    def _save_byte(self):
        bits = self.buffer[:8]
        self.buffer[:] = self.buffer[8:]

        byte_value = from_binary_list(bits)
        self.file.write(bytes([byte_value]))
        self.bytes_written += 1
        pass

    def close(self): 
        self.flush()
        self.file.close()
        pass
    pass

# =============================================================================================== #
# Input Bitstream
# =============================================================================================== #

class InputBitStream(object): 
    def __init__(self, file_name): 
        self.file_name = file_name
        self.file = open(self.file_name, 'rb') 
        self.bytes_read = 0
        self.buffer = []
        pass

    def read_bit(self):
        return self.read_bits(1)[0]

    def read_bits(self, count):
        while len(self.buffer) < count:
            self._load_byte()
        result = self.buffer[:count]
        self.buffer[:] = self.buffer[count:]
        return result

    def flush(self):
        assert(not any(self.buffer))
        self.buffer[:] = []
        pass

    def _load_byte(self):
        value = ord(self.file.read(1))
        self.buffer += pad_bits(to_binary_list(value), 8)
        self.bytes_read += 1
        pass

    def close(self): 
        self.file.close()
        pass
    pass
