import configargparse


def get_cmd_line_opts():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--input_image', default=None, help='Input image to be tested. If none skimage camera data is used.')

    opt = p.parse_args()
    return opt, p