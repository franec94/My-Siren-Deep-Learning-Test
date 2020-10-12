import configargparse


def get_cmd_line_opts():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    # p.add_argument('--experiment_name', type=str, required=True,
    #           help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')


    # General input image options
    p.add_argument('--image_filepath', type=str, default=None, required=False,
               help='Path to input image to be compressed (default: None).'
                'If not specified, It is used cameramen image as target image to be compressed.',
               )

    
    # General training options
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')
    
    """
    p.add_argument('--comb_archs_path', type=str, required=True,
        help='File path containing combinations for creating different archs for the given input image.'
    )
    """
    p.add_argument('--seeds',  nargs='+', type=int, required=False, default=[0, 42, 123, 1234],
        help='List of seeds (default: [0, 42, 123, 1234]).'
    )
    p.add_argument('--num_hidden_features', type=int, required=False, default=5,
        help='Number of hidden features to be employed across differen trials (default: 5).'
    )
    p.add_argument('--hidden_layers',  nargs='+', type=int, required=False, default=[1,2,3,4,5],
        help='List of hidden layers (default: [1,2,3,4,5]).'
    )
    p.add_argument('--num_attempts', type=int, required=False, default=5,
        help='Number of attempts per architecture sticking a particular seed value (default: 5).'
    )
    p.add_argument('--show_timetable_estimate', required=False, action='store_true',
        help='Flag to display timetable estimate.'
    )
    p.add_argument('--verbose', required=False, type=int, default=0,
        help='Verbose style logging (default: 0, a.k.a silent mode), allowed: [0 for silent, 1 for complete, 2 for minimal].'
    )
    p.add_argument('--resume_from', required=False, type=int, default=0,
        help='Ordinal number representing position within array of hidden features'
    )


    # p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
    opt = p.parse_args()
    return opt, p