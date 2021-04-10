from src.preprocessing import *
from src.extracting_video_paths import *
import argparse
from utils.utils import load_cfg


def parse_args():
    parser = argparse.ArgumentParser("prepearing data")
    parser.add_argument('-c', '--config', type=str, default='configs/config_paths.yml')

    return parser.parse_args()


def main(args):
    config = load_cfg(args.config)

    # save_test_data_path = config['data']['save_test_data_path']
    # test_paths = config['data']['test_paths']
    # save_test_name = config['data']['save_test_name']
    #
    # save_train_data_path = config['data']['save_train_data_path']
    # save_train_name = config['data']['save_train_name']
    # train_paths = config['data']['train_paths']
    #
    # preprocessed_data_path = config['data']['preprocessed_data_path']
    # if not os.path.exists(preprocessed_data_path):
    #     os.makedirs(preprocessed_data_path)
    #
    # iemocap_data_path = config['data']['iemocap_data_path']
    #
    # enterfece_data_path = config['data']['enterfece_data_path']
    # ravdess_data_path = config['data']['ravdess_data_path']
    # iemocap_cliped_video_path = config['data']['iemocap_cliped_video_path']
    #
    # filtered_save_path = config['data']['filtered_save_path']
    # if not os.path.exists(filtered_save_path):
    #     os.makedirs(filtered_save_path)
    #
    # print('starting getting paths and dividing iemocap videos ------------------------')
    #
    # get_enterface_paths(enterfece_data_path, preprocessed_data_path)
    # get_ravdess_paths(ravdess_data_path, preprocessed_data_path)
    # iemocap_divide_videos_to_clips(iemocap_data_path, preprocessed_data_path, iemocap_cliped_video_path)
    #
    # get_final_paths(preprocessed_data_path, filtered_save_path)
    #
    # print('starting prepare test data ----------------------------------')
    #
    # if not os.path.exists(save_test_data_path):
    #     os.makedirs(save_test_data_path)
    # test_df_paths = pd.read_csv(test_paths)
    # prepare_whole_data(test_df_paths, save_test_data_path, save_test_name)
    #
    # print('starting prepare train data ----------------------------------')
    #
    # if not os.path.exists(save_train_data_path):
    #     os.makedirs(save_train_data_path)
    # train_df_paths = pd.read_csv(train_paths)
    # prepare_whole_data(train_df_paths, save_train_data_path, save_train_name)

    print('starting prepare final data ----------------------------------')

    get_pickle_file_from_all_pics_and_audios_1(config)


if __name__ == '__main__':
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print('sth is wrong')
