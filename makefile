# --------------------------------------------- #
# Global Variables
# --------------------------------------------- #

INTERPRETER = python3

# main.py
# --------------------------------------------- #
SCRIPT = main.py
ARGS = --lr 1e-4 --seed 0 --gpu 0


# show_image_details.py
# --------------------------------------------- #
IMAGE_DETAILS_SCRIPT = show_image_details.py
ARGS_IMAGE_DETAILS = --seed 1234 --input-path mtestsets\BSD68\test001.png

# --------------------------------------------- #
# Tasks
# --------------------------------------------- #

run_script:
	chmod u+x $(SCRIPT)
	$(INTERPRETER) $(SCRIPT) $(ARGS)

show_image_details:
	chmod u+x $(IMAGE_DETAILS_SCRIPT)
	$(INTERPRETER) $(IMAGE_DETAILS_SCRIPT) $(ARGS_IMAGE_DETAILS)