# --------------------------------------------- #
# Globals
# --------------------------------------------- #
INTERPRETER = python3

# File: main_test_txt2enc.py
# --------------------------------------------- #
SCRIPT_TXT_2_ENC = main_test_txt2enc.py
ARGS_TXT_2_ENC = --input-file ./input.txt --output-dest ./out.txt

# File: main_test_img2enc.py
# --------------------------------------------- #
SCRIPT_IMG_2_ENC = main_test_img2enc.py
ARGS_IMG_2_ENC = --input-file ../../testsets/BSD68/test001.png --output-dest ./out.png --channels L


# File: compress_images.py
# --------------------------------------------- #
SCRIPT_DIR_IMGS_2_ENC = compress_images.py
ARGS_DIR_IMGS_2_ENC = --input-dir ../../testsets/BSD68/ --output-dest ./out.csv --channels L


# --------------------------------------------- #
# Tasks
# --------------------------------------------- #
test_huffman_enc_txt2enc:
	chmod u+x $(SCRIPT_TXT_2_ENC)
	$(INTERPRETER) $(SCRIPT_TXT_2_ENC) $(ARGS_TXT_2_ENC)

test_huffman_enc_txt2enc:
	chmod u+x $(SCRIPT_IMG_2_ENC)
	$(INTERPRETER) $(SCRIPT_IMG_2_ENC) $(ARGS_IMG_2_ENC)

test_huffman_enc_txt2enc:
	chmod u+x $(SCRIPT_DIR_IMGS_2_ENC)
	$(INTERPRETER) $(SCRIPT_DIR_IMGS_2_ENC) $(ARGS_DIR_IMGS_2_ENC)
