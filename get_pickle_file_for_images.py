from utils.preproc import *
import argparse
from utils.utils import load_cfg


def parse_args():
    parser = argparse.ArgumentParser("opt lstm model")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    get_pickle_file_from_all_pics(config)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')
