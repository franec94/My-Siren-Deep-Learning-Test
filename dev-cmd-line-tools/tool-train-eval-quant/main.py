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


def _get_number_archs(opt):
    """Get number of architectures to be evaluated when script is running.
    Params
    ------
    `opt` - python Namespace object from which deriving number of architectures to be tested.\n
    Return
    ------
    `int` - representing number of architectures to be tested.
    """
    opt_dict = collections.OrderedDict(
        n_hf=opt.n_hf,
        n_hl=opt.n_hl,
        lr=opt.lr,
        epochs=opt.num_epochs,
        seed=opt.seed,
        dynamic_quant=[opt.dynamic_quant],
        sidelength=opt.sidelength,
        batch_size=opt.batch_size,
        verbose=[opt.verbose]
    )
    opt_hyperparm_list = list(ParameterGrid(opt_dict))
    return len(opt_hyperparm_list)


def _get_size_of_model(model):
    """Return model size as file size corresponding to model's state dictionary when saved temporarily to 
    disk.
    Params
    ------
    `model` - PyTorch like model.\n
    Return
    ------
    `model_size` - int python object, size of state dictionary expressed in byte.\n
    """
    torch.save(model.state_dict(), "temp.p")
    # print('Size (MB):', os.path.getsize("temp.p")/1e6)
    model_size = os.path.getsize("temp.p")
    os.remove('temp.p')
    return model_size


def set_device_and_backend_for_torch(opt):
    """Set device which can be either CPU or GPU, or CUDA, tested in reverse order, from CUDA up to CPU.
    Set torch.backends.quantized.engine which can be either FBGEMM (for server machines) or QNNPACK (for modbile devices).
    Params
    ------
    `opt` - python Namespace object from which deriving device and quantzed backend setup.\n
    Return
    ------
    `device` - torch.device onto which some computations will be performed.\n
    `torch.cuda.device_count()` - int python object representing number of cuda devices available.\n
    `opt.quant_engine` - str obejct representing engine for quant backend\n
    """
    try:
        if opt.cuda:
            device = (torch.device('cuda:0') if torch.cuda.is_available()
            else torch.device('gpu'))
            torch.backends.quantized.engine = opt.quant_engine
        else:
            device = torch.device('cpu')
            torch.backends.quantized.engine = opt.quant_engine
    except:
        device = torch.device('cpu')
        torch.backends.quantized.engine = opt.quant_engine
        pass
    return device, torch.cuda.device_count(), opt.quant_engine


def _get_data_for_train(img_dataset, sidelength, batch_size):
    """Get data ready to be feed into a DNN model as input data for training and evaluating phase, respectively.
    Params
    ------
    `img_dataset` - PyTorch's DataSet like object representing the data against which evaluate models(base model and quantized models, if any).\n
    `sidelength` - eithr int object or lsit,tuple, representing width and height for center cropping input image.\n
    `batch_size` - int object for dividing input data into several batches.\n
    Return
    ------
    `train_dataloader` - PyTorch DataLoader instance.\n
    `val_dataloader` - PyTorch DataLoader instance.\n
    """
    coord_dataset = Implicit2DWrapper(
        img_dataset, sidelength=sidelength, compute_diff=None)

    # --- Prepare dataloaders for train and eval phases.
    train_dataloader = DataLoader(
        coord_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True, num_workers=0)

    val_dataloader = DataLoader(
        coord_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True, num_workers=0)
    
    return train_dataloader, val_dataloader


def _evaluate_dynamic_quant(opt, dtype, img_dataset, model = None, model_weight_path = None, device = 'cpu', qconfig = 'fbgemm', verbose = 0):
    """
    Evaluate model exploiting PyTorch built-it dynamic quant mode, a.k.a. Post training dynamic quantization mode.
    
    Params
    ------
    `opt` - Namespace python like object with attributes necessary to run the evaluation tasks required.\n
    `dtype` - either torch.qint8 or torch.qfloat16 instances, for quantizing model's weigths.\n
    `img_dataset` - PyTorch's DataSet like object representing the data against which evaluate models(base model and quantized models, if any).\n
    `model` - PyTorch like object representing a Neural Network model.\n
    `model_weight_path` - str like object representing local file path for model's weights to be exploited when evaluating quantized models.\n
    `device` - str object, kind of device upon which model will be loaded, allowed only CPU, since PyTorch framework supports just that setup, by now.\n
    `qconfig` - str object, quantization backed type, allowed fbgemm for x86 server architectures, or QNNAM for mobile architectures.\n
    `verbose` - int python object, for deciding verbose strategy, available options: 0 = no info displayed to stdout, 1 = info displayed to stdout object.\n
    Return
    ------
    `eval_scores, eta_eval, size_model` - np.ndarray object with values related to the following scores: MSE, PSNR, MSSi.\n
    `eta_eval` - python float object, representig time elapsed when evaluation was carried out, expressed in seconds.\n
    `size_model` - python int object representing model' size expressed in Bytes.\n
    """
    arch_hyperparams = collections.OrderedDict(
        hidden_layers=opt.n_hl[0],
        hidden_features=opt.n_hf[0],
        sidelength=opt.sidelength[0],
        dtype=dtype
    )
    if verbose > 0:
        pprint(arch_hyperparams)

    OptModel = collections.namedtuple('OptModel', list(arch_hyperparams.keys()))
    opt_model = OptModel._make(arch_hyperparams.values())
    if verbose > 0:
        pprint(opt_model)

    eval_scores, eta_eval, size_model = \
        compute_quantization_dyanmic_mode(
                model_path = model_weight_path,
                arch_hyperparams = arch_hyperparams,
                img_dataset = img_dataset,
                opt = opt_model,
                fuse_modules = None,
                device = f'{device}',
                qconfig = f'{qconfig}',
                model_fp32 = model)

    return eval_scores, eta_eval, size_model


def _evaluate_model(model, opt, img_dataset, model_weight_path = None, logging=None, verbose = 1):
    """Evaluate model after training.
    Params
    ------
    `model` - PyTorch like object representing a Neural Network model.\n
    `opt` - Namespace python like object with attributes necessary to run the evaluation tasks required.\n
    `img_dataset` - PyTorch's DataSet like object representing the data against which evaluate models(base model and quantized models, if any).\n
    `model_name` - str like object, representing a identifier with which referring to the current trial to be evaluated.\n
    `model_weight_path` - str like object representing local file path for model's weights to be exploited when evaluating quantized models.\n
    `logging` - logging python's std library object for logging reasons to a log file.\n
    `verbose` - int python object, for deciding verbose strategy, available options: 0 = no info displayed to stdout, 1 = info displayed to stdout object.\n
    Return
    ------
    `eval_info_list` - python list object containing collections.namedtuple instances with results from different evaluations.\n
    """
    eval_dataloader, _ = \
        _get_data_for_train(img_dataset, sidelength=opt.sidelength[0], batch_size=opt.batch_size[0])

    eval_field_names = "model_type,mse,psnr_db,ssim,eta_seconds,footprint_byte,footprint_percent".split(",")
    EvalInfos = collections.namedtuple("EvalInfos", eval_field_names)
    eval_info_list = []

    # tot_weights_model = sum(p.numel() for p in model.parameters())
    eval_scores, eta_eval = \
        evaluate_model(
            model=model,
            eval_dataloader=eval_dataloader,
            device='cuda')
    basic_size_model = _get_size_of_model(model)
    eval_info = EvalInfos._make(['Basic'] + list(eval_scores) + [eta_eval, basic_size_model, 100.0])
    eval_info_list.append(eval_info)

    if opt.dynamic_quant != []:
        for a_dynamic_type in opt.dynamic_quant:
            eval_scores, eta_eval, model_size = \
                _evaluate_dynamic_quant(
                    opt,
                    dtype=a_dynamic_type,
                    img_dataset=img_dataset,
                    model = copy.deepcopy(model),
                    model_weight_path = model_weight_path,
                    device = 'cpu',
                    qconfig = 'fbgemm')
            eval_info = EvalInfos._make([f'Quant-{str(a_dynamic_type)}'] + list(eval_scores) + [eta_eval, model_size, model_size / basic_size_model * 100])
            eval_info_list.append(eval_info)
            pass
        pass

    table_vals = list(map(operator.methodcaller("values"), map(operator.methodcaller("_asdict"), eval_info_list)))
    table = tabulate.tabulate(table_vals, headers=eval_field_names)
    _log_main(msg = f"{table}", header_msg = None, logging=logging, verbose=verbose)
    pass


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

    field_names = "Date,Timestamp,No_Cuda_Devices,Device,Quant_Backend,Tot_Runs".split(",")
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
    n = _get_number_archs(opt)
    field_vals.extend([no_cuda_device, device, quant_engine, n])

    # --- Check dynamic quant if any.
    # opt = check_quantization_tech_provided(opt)

    _log_main(msg = "Check dyanmic quant tech.", header_msg = None, logging=logging, verbose=opt.verbose)
    opt = check_quant_size_for_dynamic_quant(opt)
    _log_main(msg = "Done.", header_msg = None, logging=logging, verbose=opt.verbose)

    # --- Show some infos from main function.
    _log_main(msg = "Show some program infos.", header_msg = None, logging=logging)
    table_vals = list(MainInfos._make(field_vals)._asdict().items())
    table = tabulate.tabulate(table_vals, headers="Info,Val".split(","))
    _log_main(msg = f"{table}", header_msg = f'{"-" * 25} Program Details {"-" * 25}', logging=logging, verbose=1)

    # --- Train model(s)
    n = _get_number_archs(opt)
    if n == 1:
        model_trained, model_weight_path, _ = \
            train_model(opt = opt,
                image_dataset=image_dataset,
                model_dir = root_path,
                save_results_flag = True)
    else:
        eval_results_list = \
            train_model(opt = opt,
                image_dataset=image_dataset,
                model_dir = root_path,
                save_results_flag = True)
        table_vals = list(map(operator.methodcaller("values"), map(operator.methodcaller("_asdict"), eval_results_list)))
        table = tabulate.tabulate(table_vals, headers=list(eval_results_list[0]._asdict().keys()))
        _log_main(msg = f"{table}", header_msg = None, logging=logging, verbose=1)

        data = list(map(operator.methodcaller("_asdict"), eval_results_list))
        columns = list(eval_results_list[0]._asdict().keys())
        table_df = pd.DataFrame(data = data, columns = columns)

        table_path = os.path.join(root_path, "results.csv")
        table_df.to_csv(f"{table_path}")
        pass
    
    # --- Evaluate model, if just one model has been requested to be
    # trained, and evalute flag was suggested from command line.
    if n == 1:
        if opt.evaluate:
            _evaluate_model(
                model=model_trained,
                opt=opt,
                img_dataset=image_dataset,
                model_weight_path=model_weight_path)
        pass
    pass


if __name__ == "__main__":

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
