from src.preprocessing import *
import argparse
from utils.utils import load_cfg


def parse_args():
    parser = argparse.ArgumentParser("prepearing data")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    save_test_data_path = config['data']['save_test_data_path']
    test_paths = config['data']['test_paths']
    save_test_name = config['data']['save_test_name']

    save_train_data_path = config['data']['save_train_data_path']
    save_train_name = config['data']['save_train_name']
    train_paths = config['data']['train_paths']

    print('starting prepare test data ----------------------------------')

    test_df_paths = pd.read_csv(test_paths)
    prepare_whole_data(test_df_paths, save_test_data_path, save_test_name)

    print('starting prepare train data ----------------------------------')

    train_df_paths = pd.read_csv(train_paths)
    prepare_whole_data(train_df_paths, save_train_data_path, save_train_name)

    print('starting prepare final data ----------------------------------')

    get_pickle_file_from_all_pics_and_audios(config)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')
