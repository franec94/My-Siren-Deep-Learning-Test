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

DEBUG_MODE = False

# ----------------------------------------------------------------------------------------------- #
# Util-Functions for Main-Function
# ----------------------------------------------------------------------------------------------- #

def process_plain_mode(opt):
    """
    Process models by means of plain mode.
    """
    df_results = None

    print("Work: processing input .csv files - Plain mode...")
    result_tuples, files_not_found = \
        evaluate_post_train_models_by_csv_list(
            opt.log_models,
            opt)
    if result_tuples == None or len(result_tuples) == 0:
        print("No models evaluated in Plain mode.")
    else:
        print("Some models evaluated in Plain mode.")
        result_tuples = list(map(operator.methodcaller('_asdict'), result_tuples))
        df_results = pd.DataFrame(data = result_tuples)
        print(df_results.head(5))
        pass
    if files_not_found != None and len(files_not_found) != 0:
        print(files_not_found[:5])
    else:
        print("All model files processed.")
        pass
    tot = len(result_tuples) + len(files_not_found)
    print(f"PROCESSED {len(result_tuples)} | SKIPPED {len(files_not_found)}| TOT {tot}")
    return result_tuples, files_not_found, df_results


def process_posterior_quantization_mode(opt, root_path):
    """
    Process models by means of posterior quantization mode.
    """
    print("Work: processing input .csv files - Quantization mode...")
    result_tuples, files_not_found = \
        evaluate_post_train_posterion_quantized_models_by_csv_list(
            opt.log_models,
            opt)
    if result_tuples == None or len(result_tuples) == 0:
        print("No models evaluated in Plain mode.")
    else:
        print("Some models evaluated in quant mode.")
        result_tuples = list(map(operator.methodcaller('_asdict'), result_tuples))
        df_results = pd.DataFrame(data = result_tuples)

        # print(df_results.head(5))

        df_filename = os.path.join(root_path, 'result_quant.csv')
        df_results.to_csv(f'{df_filename}')

        print()
        str_h = '=' * 50  + ' Head results: ' + '=' * 50
        print('_' * len(str_h))
        print(str_h)
        table = tabulate.tabulate(tabular_data=df_results.values[:5], headers=df_results.columns)
        print(table)

        print()
        str_h = '=' * 50  + ' Tail results: ' + '=' * 50
        print('_' * len(str_h))
        print(str_h)
        table = tabulate.tabulate(tabular_data=df_results.values[-5:-1], headers=df_results.columns)
        print(table)
        pass
    if files_not_found != None and len(files_not_found) != 0:
        print(files_not_found[:5])
    else:
        print("All model files processed.")
        pass
    tot = len(result_tuples) + len(files_not_found)
    print(f"PROCESSED {len(result_tuples)} | SKIPPED {len(files_not_found)}| TOT {tot}")
    return result_tuples, files_not_found

# ----------------------------------------------------------------------------------------------- #
# Main-Function
# ----------------------------------------------------------------------------------------------- #
def main():
    """Main Function"""

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
    root_path, curr_date, curr_timestamp = \
        create_train_logging_dir(opt, debug_mode=DEBUG_MODE)
    print(f'Date: {curr_date} | Timestamp: {curr_timestamp}')
    print(f'Created root dir: {root_path}')
    
    # --- Create root logger.
    get_root_level_logger(root_path, debug_mode=DEBUG_MODE)

    # --- Log parsed cmd args.
    log_parser(root_path, parser, opt, debug_mode = DEBUG_MODE)
    logging.info(f'Date: {curr_date} | Timestamp: {curr_timestamp}')
    logging.info(f'Created root dir: {root_path}')
    logging.info(parser.format_values())

    # --- Check quantization tech, if provided:
    logging.info(f'check_quantization_tech_provided ...')
    opt = check_quantization_tech_provided(opt)
    logging.info(f'check_quantization_tech_provided: OK')
    # pprint(opt)

    logging.info(f'check_sidelength ...')
    opt = check_sidelength(opt)
    logging.info(f'check_sidelength: OK')
    # pprint(opt)

    # --- Check frequences if any.
    logging.info(f'check_frequencies ...')
    check_frequencies(opt)
    logging.info(f'check_frequencies: OK')
    # pprint(opt)

    logging.info(f'Check dynamic quant kind of transform...')
    opt = check_quant_size_for_dynamic_quant(opt)
    logging.info(f'Check dynamic quant kind of transform: OK')

    # --- Filter unwanted resources.
    logging.info(f'filter_model_files_opt_args ...')
    opt = filter_model_files_opt_args(opt)
    logging.info(f'filter_model_files_opt_args OK')

    logging.info(f'map_filter_model_dirs_opt_args ...')
    opt = map_filter_model_dirs_opt_args(opt)
    logging.info(f'filter_model_files_opt_args OK')

    logging.info(f'filter_model_files_csv_opt_args ...')
    opt = filter_model_files_csv_opt_args(opt)
    logging.info(f'filter_model_files_csv_opt_args OK')

    # if args.model_files == None or args.model_files == []: raise Exception("Error: no models to process!")

    # --- Do job.
    if opt.model_files == None or opt.model_files == []:
        print("No input .pth files provided!")
    else:
        print("TODO: processing input .pth files.")
        evaluate_models_from_files(opt)
        pass
    
    if opt.model_dirs == None or opt.model_dirs == []:
        print("No input dirs files provided!")
    else:
        print("TODO: processing input dirs.")
        pass

    if opt.log_models == None or opt.log_models == []:
        print("No input csv files provided!")
    else:
        # print("TODO: processing input .csv files.")
        if opt.plain_eval_mode:
            """
            _, _, df_results = process_plain_mode(opt)
            if isinstance(df_results, pd.DataFrame):
                if df_results.shape[0] != 0:
                    full_path = [root_path, 'processed_plain_mode.csv']

                    filename_path = os.path.join(*full_path)
                    print(f'Saving file: {filename_path}...')
                    df_results.to_csv(filename_path)
                    pass
                else:
                    print("Empty Dataframe found in plain mode!")
                    pass
            else:
                print("No dataframe gotten in plain mode!")
                pass
            """
            pass
        if opt.post_train_quant_eval_mode:
            logging.info(f'process_posterior_quantization_mode ...')
            process_posterior_quantization_mode(opt, root_path)
            logging.info(f'process_posterior_quantization_mode: DONE.')
            pass
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

    pprint(opt)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run training.
    main()
    pass