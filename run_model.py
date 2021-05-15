from src.preprocessing import *
from src.nn_video_models import *
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser("run model on examples")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')
    parser.add_argument('-p', '--path', type=str, default='logs/video.mp4')

    return parser.parse_args()


def main(args):
    t0 = time.time()
    print(args.path)

    pic, audio = testing_model(args.path)

    print(pic.shape, audio.shape)
    audio_train_dim = audio.shape[1]
    opt = tf.keras.optimizers.Adam(lr=1e-5)
    model = create_video_batchnorm_cnn_model(opt,
                                             audio_train_dim,
                                             pic.shape,
                                             4,
                                             [0.25, 0.25, 0.25, 0.25])

    model_path = "logs/video_batchnorm_cnn150Adam1e-5/mdl_wts.hdf5"
    model.load_weights(model_path)

    sample = {'audio_input': audio, 'pic_input': pic}
    label = model.predict(sample)

    t1 = time.time()

    total = t1 - t0

    print("time is   ", total)

    emotions = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3}
    print(emotions)
    print("emotion probs are :     ", label)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')
