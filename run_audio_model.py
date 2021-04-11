import argparse
from src.nn_audio_models import *


def parse_args():
    parser = argparse.ArgumentParser("train model, you can run following models: audio_cnn, audio_lstm, "
                                     "audio_blstm, audio_stacked_lstm")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-r', '--restore', type=bool)
    parser.add_argument('-cont', '--continue_at', type=int)
    parser.add_argument('-opt', '--optimizer', type=str, default='RMSprop')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-iter', '--iterations', type=int, default=100)

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)
    model_name = args.model

    logs_path = config['logs']['logs_path']
    # train_data, test_data = load_data(config)
    train_data, val_data, test_data = load_subset_labels_data(config)

    run_model(model_name,
              train_data,
              val_data,
              test_data,
              logs_path,
              restore=args.restore,
              continue_at=args.continue_at,
              optimizer=args.optimizer,
              lr=args.learning_rate,
              batch_size=args.batch_size,
              num_epochs=args.iterations)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')
