import configargparse


def get_cmd_line_opts():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--evaluate', default=False, type=bool, action='store_true', help='Include this option in order to evaluate model against input file.')
    p.add_argument('--seed', default=0, type=int, help='Seed for re-running experiments (default: 0).')
    p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')


    # General input image options
    p.add_argument('--image_filepath', type=str, default=None, required=False,
               help='Path to input image to be compressed (default: None).'
                'If not specified, It is used cameramen image as target image to be compressed.',
               )
    p.add_argument('--sidelength', type=int, required=False,
               help='Sidelength to which resize input image to be compressed.')

    
    # General training options
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')
    
    p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

    p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)'
                    'and "siren" (for Siren based neural network architectures)'
                    )

    p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
    opt = p.parse_args()
    return opt, p