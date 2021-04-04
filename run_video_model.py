import numpy as np
import argparse
from utils.utils import *
from src.nn_video_models import *


def parse_args():
    parser = argparse.ArgumentParser("train video model, you can run following models: video_cnn, ")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')
    parser.add_argument('-m', '--model', type=str, default='video_cnn')
    parser.add_argument('-r', '--restore', type=bool, default=False)
    parser.add_argument('-cont', '--continue_at', type=int, default=1)
    parser.add_argument('-opt', '--optimizer', type=str, default='RMSprop')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-iter', '--iterations', type=int, default=100)

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)
    model_name = args.model

    # tf.random.set_seed(random_seed)
    logs_path = config['logs']['logs_path']
    train_pkl = config['data']['train_pkl']
    test_pkl = config['data']['test_pkl']

    train = load_pickle(train_pkl)
    test = load_pickle(test_pkl)
    # loading datasets
    audio_train = train['train_audio_data']
    audio_train = np.array(audio_train)
    pic_train = train['train_pic_data']
    labels_train = train['train_label_data']

    audio_test = test['test_audio_data']
    audio_test = np.array(audio_test)
    pic_test = test['test_pic_data']
    labels_test = test['test_label_data']

    print("shapes of train is {}, {} and shape of label is {}".format(audio_train.shape,
                                                                      pic_train.shape,
                                                                      labels_train.shape))
    print("shapes of test is {}, {} and shape of label is {}".format(audio_test.shape,
                                                                     pic_test.shape,
                                                                     labels_test.shape))

    train_data = (audio_train, pic_train, labels_train)
    test_data = (audio_test, pic_test, labels_test)

    run_video_model(model_name,
                    train_data,
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
