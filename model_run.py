import argparse
from utils.utils import load_cfg
from utils.nn_utils import run_model


def parse_args():
    parser = argparse.ArgumentParser("train model")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')
    parser.add_argument('-m', '--model', type=str)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='you can run following models: audio_cnn, audio_lstm, audio_blstm, audio_stacked_lstm')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)
    model_name = args.model

    run_model(model_name, config)



if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')


