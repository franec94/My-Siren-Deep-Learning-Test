INTERPRETER = python3
SCRIPT = main.py

ARGS = --lr 1e-4 --seed 0 --gpu 0

run_script:
	$(INTERPRETER) $(SCRIPT)