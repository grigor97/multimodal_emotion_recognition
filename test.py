import argparse
from utils.utils import load_cfg
from utils.nn_models import *


def parse_args():
    parser = argparse.ArgumentParser("opt lstm model")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    opts = ['SGD', 'RMSprop']
    lrs = [0.1, 0.3, 0.01, 0.03, 0.001, 0.003, 0.0001, 0.0003, 0.00001, 0.00003, 0.000001]
    for opt in opts:
        for lr in lrs:
            run_model('audio_cnn',
                      config,
                      restore=False,
                      continue_at=1,
                      optimizer=opt,
                      lr=lr,
                      batch_size=64,
                      num_epochs=50)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')
