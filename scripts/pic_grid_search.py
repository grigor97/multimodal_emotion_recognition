import argparse
from src.nn_picture_models import *


def parse_args():
    parser = argparse.ArgumentParser("grid search for pic models. models are: picture_cnn,")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    # tf.random.set_seed(random_seed)
    logs_path = config['logs']['logs_path']
    # train_data, test_data = load_data(config)
    train_data, val_data, test_data = load_subset_labels_data(config)

    num_epochs = 200
    opts = ['SGD', 'RMSprop', 'Adam']
    lrs = [0.001, 0.003, 0.0001]
    model_names = ['picture_cnn']
    best_acc, best_optimizer, best_lr, best_batch_size, best_model_name = 0, None, None, None, None
    for model_name in model_names:
        for opt in opts:
            for lr in lrs:
                acc = run_picture_model(model_name,
                                        train_data,
                                        val_data,
                                        test_data,
                                        logs_path,
                                        restore=False,
                                        continue_at=1,
                                        optimizer=opt,
                                        lr=lr,
                                        batch_size=64,
                                        num_epochs=num_epochs)

                if best_acc < acc:
                    best_acc = acc
                    best_lr = lr
                    best_optimizer = opt
                    best_batch_size = 64
                    best_model_name = model_name

                    with open(logs_path + 'picture_best_model_params.txt', 'a+') as f:
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
