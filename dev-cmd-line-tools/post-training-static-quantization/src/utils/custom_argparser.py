import configargparse


def get_cmd_line_opts():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    p.add_argument('--experiment_name', type=str, required=True,
        help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--enable_tensorboard_logging', default=False, required=False, action='store_true',
        help='Flag that enable store results for later investigation via tensorboard util.')


    # General input image options
    p.add_argument('--image_filepath', type=str, default=None, required=False,
               help='Path to input image to be compressed (default: None).'
                'If not specified, It is used cameramen image as target image to be compressed.',
               )

    
    # General training options
    p.add_argument('--model_files',  nargs='+', type=str, required=False, default=[],
        help='List of model files (default: [], empty list).'
    )
    p.add_argument('--model_dirs',  nargs='+', type=str, required=False, default=[],
        help='List of dirs where looking for model files (default: [], empty list).'
    )


    # p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
    opt = p.parse_args()
    return opt, p