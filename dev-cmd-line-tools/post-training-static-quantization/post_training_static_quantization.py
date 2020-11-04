#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------- #
# Imports
# ----------------------------------------------------------------------------------------------- #
from src.utils.libs import *

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)


# ----------------------------------------------------------------------------------------------- #
# Globals
# ----------------------------------------------------------------------------------------------- #

opt, parser, device = None, None, None

DEBUG_MODE = True

# ----------------------------------------------------------------------------------------------- #
# Util-Functions for Main-Function
# ----------------------------------------------------------------------------------------------- #

def main():

    # --- Get cmd line options and parser objects.
    global device
    global opt
    global parser
    # check_cmd_line_options()

    # --- Check verbose style.
    if hasattr(opt, 'verbose'):
        if opt.verbose not in [0, 1, 2]:
            raise ValueError(f"opt.verbose = {opt.verbose} not allowed!")

    # --- Create logging dirs.
    # root_path, curr_date, curr_timestamp = \
    root_path, _, _ = \
        create_train_logging_dir(opt, debug_mode=DEBUG_MODE)
    
    # --- Create root logger.
    get_root_level_logger(root_path, debug_mode=DEBUG_MODE)

    # --- Log parsed cmd args.
    log_parser(root_path, parser, debug_mode = DEBUG_MODE)
    logging.info(parser.format_values())

    # --- Filter unwanted resources.
    opt = filter_model_files_opt_args(opt)
    opt = map_filter_model_dirs_opt_args(opt)
    opt = filter_model_files_csv_opt_args(opt)

    # if args.model_files is None or args.model_files is []: raise Exception("Error: no models to process!")

    # --- Do job.
    if opt.model_files is None or opt.model_files is []:
        print("No input .ph files provided!")
    else:
        print("TODO: processing input .ph files.")
        pass
    if opt.model_dirs is None or opt.model_files is []:
        print("No input dirs files provided!")
    else:
        print("TODO: processing input dirs.")
        pass
    if opt.log_models is None or opt.log_models is []:
        print("No input csv files provided!")
    else:
        print("TODO: processing input .csv files.")
        pass

    
    pass


if __name__ == "__main__":

    # Initialize option and parser objects.
    opt, parser = get_cmd_line_opts()
    
    # Set seeds for experiment re-running.
    if hasattr(opt, 'seed'): seed = opt.seed
    else: seed = 0
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run training.
    main()
    pass