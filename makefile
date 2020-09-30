# ============================================= #
# Variables
# ============================================= #

INTERPRETER = python3


# Stats about generic .BMP file
# --------------------------------------------- #
READ_BMP_FILE_SCRIPT = read_bmp_file.py
FILE_NAME_BMP = test.bmp
ARGS_READ_BMP_FILE = $(FILE_NAME_BMP)


# Stats about generic .BMP file
# --------------------------------------------- #
USE_HUFFMAN_SCRIPT = UseHuffman.py
ARGS_USE_HUFFMANE = 


# Try UseHuffman.py
# --------------------------------------------- #
TEST_HUFFMAN_SCRIPT = test_enc_dec_file_huffman.py
ARGS_TEST_HUFFMAN = --input-file-path ".\resources\bmp\flag.bmp" --output-path ".\"



# ============================================= #
# Makefile Tasks
# ============================================= #

read_bmp_file_stats:
	$(INTERPRETER) $(READ_BMP_FILE_SCRIPT) $(ARGS_READ_BMP_FILE)

use_huffmane_stats:
	$(INTERPRETER) $(USE_HUFFMAN_SCRIPT) $(ARGS_USE_HUFFMANE)

test_huffmane_stats:
	$(INTERPRETER) $(TEST_HUFFMAN_SCRIPT) $(ARGS_TEST_HUFFMAN)
