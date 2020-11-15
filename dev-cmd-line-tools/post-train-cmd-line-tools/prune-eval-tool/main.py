"""Main Python source file for evaluating Prune techniques."""
#!/usr/bin/env python3
 # -*- coding: utf-8 -*-

from src.libs import *

# Setup warnings
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

def _log_main(msg, header_msg = None, logging=None, verbose = 0):
    """Log information messages to logging and tqdm objects.
    Params:
    -------
    `msg` either a str object or list of objects to be logged.\n
    `header_msg` str object used as header or separator, default None means no header will be shown.\n
    `logging` logging python's std lib object, if None no information will be logged via logging.\n
    `tqdm` tqdm python object, if None no information will be logged via tqdm.\n
    Return:
    -------
    None
    """
    if header_msg != None:
        if verbose > 0:
            print(header_msg)
        if logging != None:
            logging.info(header_msg)
    if verbose > 0:
        print(msg)
    if logging != None:
        logging.info(msg)
    pass


def main(opt):
    """Main function for evaluating Prune techniques.
    Params
    ------
    `args` - Python Namespace object.\n
    """

    field_names = "Date,Timestamp,No_Cuda_Devices,Device,Quant_Backend".split(",")
    field_vals = []
    MainInfos = collections.namedtuple('MainInfos', field_names)

    # --- Check verbose style.
    if opt.verbose not in [0, 1, 2]:
        raise ValueError(f"opt.verbose = {opt.verbose} not allowed!")

    # --- Create logging dirs.
    root_path, curr_date, curr_timestamp = \
        create_train_logging_dir(opt)
    field_vals.extend([curr_date, curr_timestamp])

    # --- Get root level logger.
    get_root_level_logger(root_path)
    
    # --- Save parser.
    _log_main(msg = "Saving parser...", header_msg = None, logging=logging, verbose=opt.verbose)
    log_parser(root_path, parser, opt, debug_mode = False)
    logging.info(parser.format_values())
    _log_main(msg = "Done.", header_msg = None, logging=logging, verbose=opt.verbose)

    # --- Load image.
    _log_main(msg = "Loading image as PyTorch Dataset...", header_msg = None, logging=logging, verbose=opt.verbose)
    image_dataset, _, _ = \
        get_input_image(opt)
    _log_main(msg = "Done.", header_msg = None, logging=logging, verbose=opt.verbose)

    device, no_cuda_device, quant_engine = \
        set_device_and_backend_for_torch(opt)
    field_vals.extend([no_cuda_device, device, quant_engine])

    _log_main(msg = "Check dyanmic quant tech.", header_msg = None, logging=logging, verbose=opt.verbose)
    opt = check_quant_size_for_dynamic_quant(opt)
    _log_main(msg = "Done.", header_msg = None, logging=logging, verbose=opt.verbose)

    # --- Show some infos from main function.
    _log_main(msg = "Show some program infos.", header_msg = None, logging=logging)
    table_vals = list(MainInfos._make(field_vals)._asdict().items())
    table = tabulate.tabulate(table_vals, headers="Info,Val".split(","))
    _log_main(msg = f"{table}", header_msg = f'{"-" * 25} Program Details {"-" * 25}', logging=logging, verbose=1)

    eval_info_list, df = \
        compute_prune_unstructured_results(opt, image_dataset, verbose = 0)
    
    if eval_info_list != []:
        df.to_csv(root_path + 'results.csv')
        print(df.head(5))
        print(df.tail(5))
        pass

    if opt.run_dash:
        """
        _log_main(msg = f"Prepare Dash App...", header_msg = f'{"-" * 25} Program Details {"-" * 25}', logging=logging, verbose=1)
        tab_names = 'Results Siren(MSE,PSNR,SSIM);Results Merged(MSE,PSNR,SSIM)'.split(";")
        app = get_dash_app(
            figs_list = complex_figs_dash_list,
            n_figs = len(y_list),
            tab_names_list = tab_names)

        _log_main(msg = f"'Dash app start running...'", header_msg = f'{"-" * 25} Program Details {"-" * 25}', logging=logging, verbose=1)
        app.run_server(debug=True, use_reloader=False, host='localhost') 
        """
    pass


if __name__ == "__main__":
    """Python Entry Point for evaluating Prune techniques."""

    opt, parser = get_cmd_line_opts()

    # Set seeds for experiment re-running.
    if hasattr(opt, 'seed'): seed = opt.seed[0]
    else: seed = 0
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(opt)
    pass
