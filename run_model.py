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
    pic = np.transpose(pic, (1, 2, 0))
    pic = np.expand_dims(pic, axis=0)

    train_mean = np.load('logs/train_mean.npy')
    safe_max = np.load('logs/safe_max.npy')
    # print("mean shape --->>> ", train_mean.shape)
    # print("max shape --->>> ", safe_max.shape)
    audio -= train_mean
    audio /= safe_max
    audio = audio.reshape(1, audio.shape[0])
    # audio = np.expand_dims(audio, axis=0)
    print(pic.shape, audio.shape)
    audio_train_dim = audio.shape[1]
    opt = tf.keras.optimizers.Adam(lr=1e-5)
    model = create_video_batchnorm_cnn_model(opt,
                                             162,
                                             (50, 50, 20),
                                             4,
                                             [0.25, 0.25, 0.25, 0.25])

    model_path = "logs/video_batchnorm_cnn150Adam0.001/mdl_wts.hdf5"
    # model_path = "logs/video_batchnorm_cnn150Adam1e-05/mdl_wts.hdf5"
    # model_path = "logs/video_batchnorm_cnn150Adam1e-05/mdl_wts.hdf5"
    model.load_weights(model_path)

    print("summary", model.summary())
    print("input shape", model.input_shape)
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
