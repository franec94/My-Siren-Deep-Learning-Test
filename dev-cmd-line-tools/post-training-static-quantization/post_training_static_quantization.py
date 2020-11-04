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

args, parser, device = None, None, None

# ----------------------------------------------------------------------------------------------- #
# Util-Functions for Main-Function
# ----------------------------------------------------------------------------------------------- #

def main():

    # --- Get cmd line options and parser objects.
    global device
    global args
    global parser
    # check_cmd_line_options()

    # --- Get input image to be compressed.
    img_dataset, img, image_resolution = \
        get_input_image(opt)

    # --- Check verbose style.
    if hasattr(opt, 'verbose'):
        if opt.verbose not in [0, 1, 2]:
            raise ValueError(f"opt.verbose = {opt.verbose} not allowed!")

    # --- Create logging dirs.
    root_path, curr_date, curr_timestamp = \
        create_train_logging_dir(opt)
    
    # --- Create root logger.
    get_root_level_logger(root_path)

    # --- Log parsed cmd args.
    parser_logged = os.path.join(root_path, 'parser_logged.txt')
    with open(parser_logged, "w") as f:
        f.write(parser.format_values())
        pass
    parser_pickled = os.path.join(root_path, 'parser.pickle')
    with open(parser_logged, "w") as f:
        pickle.dump(parser, f)
        pass
    logging.info(parser.format_values())

    args = filter_model_files_opt_args(args)
    args = map_filter_model_dirs_opt_args(args)
    args = filter_model_files_csv_opt_args(args)

    # if args.model_files is None or args.model_files is []: raise Exception("Error: no models to process!")

    if args.model_files is None or args.model_files is []:
        print("No input .ph files provided!")
    else:
        print("TODO: processing input .ph files.")
        pass
    if args.model_dirs is None or args.model_files is []:
        print("No input dirs files provided!")
    else:
        print("TODO: processing input dirs.")
        pass
    if args.log_models is None or args.log_models is []:
        print("No input csv files provided!")
    else:
        print("TODO: processing input .csv files.")
        pass

    
    pass


if __name__ == "__main__":

    # Initialize option and parser objects.
    args, parser = get_cmd_line_opts()
    
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