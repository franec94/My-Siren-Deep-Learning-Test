import configargparse

DYNAMIC_QUAT_SIZES = "qint8,qfloat16".split(",")


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
    parser.add_argument('--enable_tensorboard_logging', default=False, required=False, action='store_true',
        help='Flag that enable store results for later investigation via tensorboard util.')
    parser.add_argument('--epochs_til_ckpt', type=int, default=50,
               help='Time interval in seconds until checkpoint is saved.')
    parser.add_argument('--num_attempts', type=int, required=False, default=1,
        help='Number of attempts per architecture sticking a particular seed value (default: 1).'
    )
    parser.add_argument('--verbose', required=False, type=int, default=0,
        help='Verbose style logging (default: 0, a.k.a silent mode), allowed: [0 for silent, 1 for complete, 2 for minimal].'
    )

    # Regularization Options
    parser.add_argument('--lambda_L_1', nargs='+', required=False, type=float, default=[0.0], dest="lambda_L_1",
        help='Is this option is set, then L1-norm regularization is enabled and the value of such an option describes how much weigths L1-norm as regularization term (Default: 0.0, that is not L1-norm exploited).'
    )
    parser.add_argument('--lambda_L_2', nargs='+', required=False, type=float, default=[0.0], dest="lambda_L_2",
        help='Is this option is set, then L1-norm regularization is enabled and the value of such an option describes how much weigths L2-norm as regularization term (Default: 0.0, that is not weigth decay exploited).'
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
    parser.add_argument('--lr', nargs='+', type=float, default=[1e-4], help='learning rate. default=1e-4')
    parser.add_argument('--num_epochs', nargs='+', type=int, default=[10000], dest='num_epochs',
               help='Number of epochs to train for.')
    parser.add_argument('--seed',  nargs='+', type=int, required=False, default=[0],
        help='List of seeds (default: [0]).'
    )


    # Options for evaluating model, after training.
    parser.add_argument("--train", required=False, action="store_true", default=False, dest='evaluate',
        help="Flag for training mode. (Default: False)"
    )
    parser.add_argument("--evaluate", required=False, action="store_true", default=False, dest='evaluate',
        help="Flag for evaluating model after training"
    )
    parser.add_argument('--dynamic_quant', required=False, nargs='+', type=str, default=[], dest='dynamic_quant',
        help='Set it to enable dynamic quantization training. (Default: empty list, Allowed: [qint8, float16])'
    )
    parser.add_argument('--frequences',  nargs='+', type=int, required=False, default=[],
        help='List of frequences to be employed when quantization_enabled flag is set to paszke_quant (default: None).'
    )
    parser.add_argument('--cuda',  required=False, action="store_true", default=False, dest='cuda',
        help='Set this flag to enable training on CUDA device, otherwise training will be performed on CPU device (default: False).'
    )
    parser.add_argument('--quant_engine',  required=False, type=str, default='fbgemm',  dest='quant_engine',
        help='Kind of quant engine (default: fbgemm).'
    )

    opt = parser.parse_args()
    return opt, parser