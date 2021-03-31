import argparse
from utils.utils import load_cfg
from utils.nn_utils import *


def parse_args():
    parser = argparse.ArgumentParser("train model, you can run following models: audio_cnn, audio_lstm, "
                                     "audio_blstm, audio_stacked_lstm")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-r', '--restore', type=bool)
    parser.add_argument('-cont', '--continue_at', type=int)

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)
    model_name = args.model

    if not args.restore:
        run_model(model_name, config)
    else:
        cont_run_model(model_name, config, args.continue_at)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')


