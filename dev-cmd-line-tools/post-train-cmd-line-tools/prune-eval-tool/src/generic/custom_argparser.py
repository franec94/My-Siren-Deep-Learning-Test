import configargparse

DYNAMIC_QUAT_SIZES = "qint8,qfloat16".split(",")

UNSTR_PRUNE_TECHS = 'L1Unstructured,RandomUnstructured'.split(",")


def get_cmd_line_opts():
    """Get command line options parsed from command line once the program is running.
    Return
    ------
    opt - Namespace python object corresponding to the command line options provided to the program.\n
    parser - parser used for parsing command line options provided to the program.\n
    """

    # Define command line argument parser.
    parser = configargparse.ArgumentParser()
    parser.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    # Options for storing results.
    parser.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    parser.add_argument('--experiment_name', type=str, required=True,
        help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    parser.add_argument('--verbose', required=False, type=int, default=0,
        help='Verbose style logging (default: 0, a.k.a silent mode), allowed: [0 for silent, 1 for complete, 2 for minimal].'
    )

    # Options for loading data to be processed.
    parser.add_argument('--image_filepath', type=str, default=None, required=False, dest='image_filepath',
               help='Path to input image to be compressed (default: None). If not specified, It is used cameramen image as target image to be compressed.',
    )

    # Options for building Model, via hyper-params.
    parser.add_argument('--sidelength', nargs='+', type=int, required=False, default=[], dest='sidelength',
               help='Sidelength to which resize input image to be compressed (default: empty list, which means no cropping input image)'
    )
    parser.add_argument('--n_hf', nargs='+', type=int, required=False, default=[64], dest='n_hf',
        help='A number of hidden features or a list of hidden features to be evaluated (default: [64])).'
    )
    parser.add_argument('--n_hl',  nargs='+', type=int, required=False, default=[3], dest='n_hl',
        help='A number of hidden layers or a list of hidden layers to be evaluated  (default: [3]).'
    )
    
    # Options for running training phase.
    parser.add_argument('--batch_size', nargs='+', type=int, default=[1])

    parser.add_argument('--dynamic_quant', required=False, nargs='+', type=str, default=[], dest='dynamic_quant',
        help='Set it to enable dynamic quantization training. (Default: empty list, Allowed: [qint8, float16])'
    )
    parser.add_argument('--frequences',  nargs='+', type=int, required=False, default=[],
        help='List of frequences to be employed when quantization_enabled flag is set to paszke_quant (default: None).'
    )
    parser.add_argument('--cuda',  required=False, action="store_true", default=False, dest='cuda',
        help='Set this flag to enable Eval on CUDA device, otherwise training will be performed on CPU device (default: False).'
    )
    parser.add_argument('--quant_engine',  required=False, type=str, default='fbgemm',  dest='quant_engine',
        help='Kind of quant engine (default: fbgemm).'
    )

    # Options for pruning.
    parser.add_argument('--global_pruning_rates', nargs='+', type=float, required=False, default=[], dest='global_pruning_rates',
               help='Rates for pruning model globally (default: empty list, which means no global rate pruning approach will be done)'
    )
    parser.add_argument('--global_pruning_abs', nargs='+', type=int, required=False, default=[], dest='global_pruning_rates',
               help='Absolute number of weigths to be pruned from model globally (default: empty list, which means no global abs pruning will be done)'
    )
    parser.add_argument('--global_pruning_techs', nargs='+', type=str, required=False, default=[], dest='global_pruning_rates',
               help=f'Pruing techs to prune model globally (default: empty list, which means no global pruning will be done) Allowed: [{str(UNSTR_PRUNE_TECHS)}]'
    )

    parser.add_argument('--run_dash',  required=False, action="store_true", default=False, dest='run_dash',
        help='Set this flag to enable run dash app with results (default: False).'
    )
    opt = parser.parse_args()
    return opt, parser