#!/usr/bin/env python3
 # -*- coding: utf-8 -*-

from src.utils.libs import *

# ----------------------------------------------------------------------------------------------- #
# Globals
# ----------------------------------------------------------------------------------------------- #

opt, parser, device = None, None, None

# ----------------------------------------------------------------------------------------------- #
# Util Functions for Main
# ----------------------------------------------------------------------------------------------- #

def check_cmd_line_options():
    """Show parser and options"""

    global opt, parser
    
    print(opt)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())    # useful for logging where different settings came from.
    pass


def set_device_for_torch(opt, engine = 'fbgemm'):
    """Set device which can be either CPU or GPU, or CUDA, tested in reverse order, from CUDA up to CPU.
    Set torch.backends.quantized.engine which can be either FBGEMM (for server machines) or QNNPACK (for modbile devices).
    """
    try:
        if opt.quantization_enabled == None:
            device = (torch.device('cuda:0') if torch.cuda.is_available()
            else torch.device('gpu'))
        else:
            device = torch.device('cpu')    
            torch.backends.quantized.engine = engine
            pass
    except:
        device = torch.device('cpu')
        pass
    print(f"Training on device {device}.")
    logging.info(f"Training on device {device}.")

    print(f"# cuda device: {torch.cuda.device_count()}")
    logging.info(f"# cuda device: {torch.cuda.device_count()}")

    if torch.cuda.device_count() > 0:
        print(f"Id current device: {torch.cuda.current_device()}")
        logging.info(f"Id current device: {torch.cuda.current_device()}")
        pass

# ----------------------------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------------------------- #

def main():
    """Main Function"""

    # --- Get cmd line options and parser objects.
    global device
    global opt
    global parser
    # check_cmd_line_options()

    # --- Get input image to be compressed.
    img_dataset, img, image_resolution = \
        get_input_image(opt)

    # --- Get Hyper-params list.
    grid_arch_hyperparams = \
        get_arch_hyperparams(opt, image_resolution)

    # --- Check verbose style.
    if opt.verbose not in [0, 1, 2]:
        raise ValueError(f"opt.verbose = {opt.verbose} not allowed!")

    # --- Set Hyper-params to be tested.
    # logging.info("Set Hyper-params to be tested")
    opt, pos_start, pos_end = \
        set_hyperparams_to_be_tested(opt, grid_arch_hyperparams)

    # --- Create logging dirs.
    root_path, curr_date, curr_timestamp = \
        create_train_logging_dir(opt)
    print(f'Date: {curr_date} | Timestamp: {curr_timestamp}')
    print(f'Created root dir: {root_path}')
    # --- Create root logger.
    get_root_level_logger(root_path)

    logging.info(f'Date: {curr_date} | Timestamp: {curr_timestamp}')
    logging.info(f'Created root dir: {root_path}')

    # --- Log parsed cmd args.
    """
    parser_logged = os.path.join(root_path, 'parser_logged.txt')
    with open(parser_logged, "w") as f:
        f.write(parser.format_values())
        pass
    """
    log_parser(root_path, parser, opt, debug_mode = False)
    logging.info(parser.format_values())

    # --- Show overall number of trials.
    show_number_of_trials(opt, grid_arch_hyperparams, via_tabulate = False)

    # --- Set device upon which compute model's fitting
    # or evaluation, depending on the current desired task.
    set_device_for_torch(opt)

    # --- Check quantization tech, if provided:
    opt = check_quantization_tech_provided(opt)

    # --- Start training.
    start_time = time.time()
    print(f"Start training [{curr_date}][timestamp={curr_timestamp}] ...")
    logging.info(f"Start training [{curr_date}][timestamp={curr_timestamp}] ...")

    train_extended_compare.train_extended_protocol_compare_archs(
        grid_arch_hyperparams=grid_arch_hyperparams[pos_start:pos_end],
        img_dataset=img_dataset,
        device = device.type,
        opt=opt,
        model_dir=root_path,
        verbose=opt.verbose,
    )

    print(f"End training [{curr_date}][timestamp={curr_timestamp}] eta: {time.time() - start_time} seconds.")
    logging.info(f"End training [{curr_date}][timestamp={curr_timestamp}] eta: {time.time() - start_time} seconds.")
    
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
