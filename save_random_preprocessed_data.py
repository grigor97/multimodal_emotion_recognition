import os
import argparse
from utils.nn_utils import *

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("saves some random data into logs")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    num = 5
    logs_path = config['logs']['logs_path']
    train_data, test_data = load_video_data(config)

    audio_train, pic_train, labels_train = train_data
    audio_test, pic_test, labels_test = test_data

    train_length = pic_train.shape[0]
    test_length = pic_test.shape[0]

    train_idx = np.random.randint(train_length, size=num)
    test_idx = np.random.randint(test_length, size=num)

    train_samples = pic_train[train_idx, :, :, 9]
    test_samples = pic_test[test_idx, :, :, 9]

    for i in range(num):
        tr_im = Image.fromarray(train_samples[i])
        te_im = Image.fromarray(test_samples[i])

        if not os.path.exists(logs_path + 'samples/'):
            os.makedirs(logs_path + 'samples/')

        tr_im.save(logs_path + 'samples/' + str(i) + 'train.jpeg')
        te_im.save(logs_path + 'samples/' + str(i) + 'test.jpeg')


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')
