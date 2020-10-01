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


# --------------------------------------------- #
# Tasks
# --------------------------------------------- #
test_huffman_enc_txt2enc:
	chmod u+x $(SCRIPT_TXT_2_ENC)
	$(INTERPRETER) $(SCRIPT_TXT_2_ENC) $(ARGS_TXT_2_ENC)

test_huffman_enc_txt2enc:
	chmod u+x $(SCRIPT_IMG_2_ENC)
	$(INTERPRETER) $(SCRIPT_IMG_2_ENC) $(ARGS_IMG_2_ENC)
