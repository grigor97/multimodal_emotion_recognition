from extracting_video_paths import *
from preprocessing import *

# preprocessed_data_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/preprocessed_data/'
# iemocap_data_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/IEMOCAP_full_release/"
# enterfece_data_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/enterface_database"
# ravdess_data_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/RAVDESS'
# iemocap_cliped_video_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/iemocap/'


preprocessed_data_path = "/home/student/keropyan/data/preprocessed_data/"
iemocap_data_path = "/home/student/keropyan/data/IEMOCAP_full_release/"
enterfece_data_path = "/home/student/keropyan/data/enterface_database/"
ravdess_data_path = '/home/student/keropyan/data/RAVDESS'
iemocap_cliped_video_path = '/home/student/keropyan/data/iemocap/'
filtered_save_path = '/home/student/keropyan/data/preprocessed_data/filtered_emotions_paths/'

# get_enterface_paths(enterfece_data_path, preprocessed_data_path)
#
# get_ravdess_paths(ravdess_data_path, preprocessed_data_path)
#
# iemocap_divide_videos_to_clips(iemocap_data_path, preprocessed_data_path, iemocap_cliped_video_path)


get_final_paths(preprocessed_data_path, filtered_save_path)


# save_train_data_path = "/home/student/keropyan/data/preprocessed_data/train_data/"
# save_test_data_path = "/home/student/keropyan/data/preprocessed_data/test_data/"
#
# train_paths = "/home/student/keropyan/data/preprocessed_data/filtered_emotions_paths/train_paths.csv"
# test_paths = "/home/student/keropyan/data/preprocessed_data/filtered_emotions_paths/test_paths.csv"
#
# train_df_paths = pd.read_csv(train_paths)
# test_df_paths = pd.read_csv(test_paths)
#
# prepare_whole_data(train_df_paths, save_train_data_path, 'final_train_paths.csv')
# prepare_whole_data(test_df_paths, save_test_data_path, 'final_test_paths.csv')
