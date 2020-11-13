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


def set_device_and_backend_for_torch(opt):
    """Set device which can be either CPU or GPU, or CUDA, tested in reverse order, from CUDA up to CPU.
    Set torch.backends.quantized.engine which can be either FBGEMM (for server machines) or QNNPACK (for modbile devices).
    """
    try:
        if opt.cuda:
            device = (torch.device('cuda:0') if torch.cuda.is_available()
            else torch.device('gpu'))
            torch.backends.quantized.engine = opt.engine
    except:
        device = torch.device('cpu')
        torch.backends.quantized.engine = opt.engine
        pass
    return device, torch.cuda.device_count(), opt.engine


def _get_data_for_train(img_dataset, sidelength, batch_size):
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


def _evaluate_dynamic_quant(opt, dtype, img_dataset, model = None, model_weight_path = None, device = 'cpu', qconfig = 'fbgemm'):
    arch_hyperparams = dict(
        hidden_layers=opt.n_hl[0],
        hidden_features=opt.n_hf[0],
        sidelength=opt.sidelength[0],
        dtype=dtype
    )
    eval_scores, eta_eval, size_model = \
        compute_quantization_dyanmic_mode(
                model_path = model_weight_path,
                arch_hyperparams = arch_hyperparams,
                img_dataset = img_dataset,
                opt = opt,
                fuse_modules = None,
                device = f'{device}',
                qconfig = f'{qconfig}',
                model_fp32 = model)

    return eval_scores, eta_eval, size_model


def _evaluate_model(model, opt, img_dataset, model_weight_path = None, logging=None):

    eval_dataloader, _ = \
        _get_data_for_train(img_dataset, sidelength=opt.sidelength[0], batch_size=opt.batch_size[0])

    eval_field_names = "model_type,mse,psnr,ssim,eta,footprint_byte,footprint_percent".split(",")
    EvalInfos = collections.namedtuple("EvalInfos", eval_field_names)
    eval_info_list = []

    tot_weights_model = sum(p.numel() for p in model.parameters())
    eval_scores, eta_eval = \
        evaluate_model(
            model=model,
            eval_dataloader=eval_dataloader,
            device='cuda')
    eval_info = EvalInfos._make(['Basic'] + list(eval_scores) + [eta_eval, tot_weights_model * 4, 100.0])
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
            eval_info = EvalInfos._make([f'Quant-{str(a_dynamic_type)}'] + list(eval_scores) + [eta_eval, model_size, model_size / tot_weights_model * 4])
            eval_info_list.append(eval_info)
            pass
        pass

    table_vals = list(map(operator.methodcaller("items"), map(operator.methodcaller("_asdict"), eval_info_list)))
    table = tabulate.tabulate(table_vals, headers=eval_field_names)
    _log_main(msg = f"{table}", header_msg = None, logging=logging)
    pass


def _log_main(msg, header_msg = None, logging=None):
    if header_msg != None:
        print(header_msg)
        if logging != None:
            logging.info(header_msg)
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
    _log_main(msg = "Saving parser...", header_msg = None, logging=logging)
    log_parser(root_path, parser, opt, debug_mode = False)
    logging.info(parser.format_values())
    _log_main(msg = "Done.", header_msg = None, logging=logging)

    # --- Load image.
    _log_main(msg = "Loading image as PyTorch Dataset...", header_msg = None, logging=logging)
    image_dataset, _, _ = \
        get_input_image(opt)
    _log_main(msg = "Done.", header_msg = None, logging=logging)

    device, no_cuda_device, quant_engine = \
        set_device_and_backend_for_torch(opt)
    field_vals.extend([device, no_cuda_device, quant_engine, -1])

    # --- Check dynamic quant if any.
    # opt = check_quantization_tech_provided(opt)

    # --- Show some infos from main function.
    _log_main(msg = "Show some program infos.", header_msg = None, logging=logging)
    table_vals = list(MainInfos._make(field_vals)._asdict().items())
    table = tabulate.tabulate(table_vals, headers="Info,Val".split(","))
    _log_main(msg = f"{table}", header_msg = f'{"-" * 25} Program Details {"-" * 25}', logging=logging)

    # --- Train model(s)
    model_trained, model_weight_path, train_scores_path = \
        train_model(opt = opt,
            image_dataset=image_dataset,
            model_dir = root_path,
            save_results_flag = True)
    
    # --- Evaluate model, if just one model has been requested to be
    # trained, and evalute flag was suggested from command line.
    if model_trained != None and model_weight_path != None and train_scores_path != None:
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
    main(opt)
    pass
