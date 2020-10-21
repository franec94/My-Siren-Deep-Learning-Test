import configargparse


def get_cmd_line_opts():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    # Specify input image to be processed or alternatively default camera image will be employed.
    p.add_argument('--input_image', type=str, default=None, help='Input image to be tested. If none skimage camera data is used.')

    # Specify paths where to store/record data/results
    p.add_argument('--logging_path', type=str, default='log', help='Path where log will be stored')
    p.add_argument('--output_path', type=str, default='output', help='Path where output will be stored')

    # Specify wheter to erase previous contents, if any, stored within dirs
    # where new results will be storeed.
    p.add_argument('--erase_content_prev_logging', default=False, action='store_true',
        help='Erase output from previous analysis, within logging path')
    p.add_argument('--erase_content_prev_output', default=False, action='store_true',
        help='Erase output from previous analysis, within output path')

    opt = p.parse_args()
    return opt, p