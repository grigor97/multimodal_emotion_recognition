import argparse
from src.nn_video_models import *


def parse_args():
    parser = argparse.ArgumentParser("grid search for video models. models are: video_cnn,"
                                     "video_batchnorm_cnn, video_big_cnn")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    # tf.random.set_seed(random_seed)
    logs_path = config['logs']['logs_path']
    # train_data, test_data = load_data(config)
    train_data, test_data = load_subset_labels_data(config)

    num_epochs = 100
    opts = ['SGD', 'RMSprop', 'Adam']
    lrs = [0.3, 0.01, 0.03, 0.001, 0.003, 0.0001, 0.0003, 0.00001, 0.00003, 0.000001, 0.000000001]
    # model_names = ['video_big_batchnorm_cnn', 'video_batchnorm_cnn', 'video_big_cnn',
    # 'video_cnn', 'video_bcnn', 'video_bcnn', 'video_blstm']
    model_names = ['video_blstm']
    best_acc, best_optimizer, best_lr, best_batch_size, best_model_name = 0, None, None, None, None
    for model_name in model_names:
        for opt in opts:
            for lr in lrs:
                acc = run_video_model(model_name,
                                      train_data,
                                      test_data,
                                      logs_path,
                                      restore=False,
                                      continue_at=1,
                                      optimizer=opt,
                                      lr=lr,
                                      batch_size=32,
                                      num_epochs=num_epochs)

                if best_acc < acc:
                    best_acc = acc
                    best_lr = lr
                    best_optimizer = opt
                    best_batch_size = 64
                    best_model_name = model_name

                    with open(logs_path + 'video_best_model_params.txt', 'a+') as f:
                        f.write("best accuracy is ")
                        f.write(str(best_acc) + ',   ')
                        f.write("best lr is ")
                        f.write(str(best_lr) + ',   ')
                        f.write("bets opt is ")
                        f.write(str(best_optimizer) + ',   ')
                        f.write("bets batch size is ")
                        f.write(str(best_batch_size) + ',   ')
                        f.write("bets model is ")
                        f.write(str(best_model_name) + ',   ')
                        f.write("epochs count is ")
                        f.write(str(num_epochs) + ',   ')
                        f.write('---------------------------------')
                        f.write('\n')


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')
