import argparse

def get_custom_argparser():
    parser = argparse.ArgumentParser(description='Huffmane Coding - Tests')
    parser.add_argument('--input-file-path', type=str, dest="input_file_path",
                    help='path to input filename.')
    parser.add_argument('--output-path', type=str, default=".", dest="output_path",
                    help='output location of resulting decoded file.')
    return parser