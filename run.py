from preprocessing import *

# TODO fix paths
preprocessed_data_path = "/home/student/keropyan/data/preprocessed_data/"
iemocap_data_path = "/home/student/keropyan/data/IEMOCAP_full_release/"
enterfece_data_path = "/home/student/keropyan/data/enterface_database/"
ravdess_data_path = '/home/student/keropyan/data/RAVDESS'


# preprocessed_data_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/preprocessed_data/'
# iemocap_data_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/IEMOCAP_full_release/"
# enterfece_data_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/enterface_database"
# ravdess_data_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/RAVDESS'

get_enterface_paths(enterfece_data_path, preprocessed_data_path)

get_ravdess_paths(ravdess_data_path, preprocessed_data_path)

iemocap_divide_videos_to_clips(iemocap_data_path, preprocessed_data_path)

