from extracting_video_paths import *

# preprocessed_data_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/preprocessed_data/'
# iemocap_data_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/IEMOCAP_full_release/"
# enterfece_data_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/enterface_database"
# ravdess_data_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/RAVDESS'
# iemocap_cliped_video_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/iemocap/'


preprocessed_data_path = "../data/preprocessed_data/"
iemocap_data_path = "../data/IEMOCAP_full_release/"
enterfece_data_path = "../data/enterface_database/"
ravdess_data_path = '../data/RAVDESS'
iemocap_cliped_video_path = '../data/iemocap/'
filtered_save_path = '../data/preprocessed_data/filtered_emotions_paths/'

get_enterface_paths(enterfece_data_path, preprocessed_data_path)

get_ravdess_paths(ravdess_data_path, preprocessed_data_path)

iemocap_divide_videos_to_clips(iemocap_data_path, preprocessed_data_path, iemocap_cliped_video_path)

get_final_paths(preprocessed_data_path, filtered_save_path)
