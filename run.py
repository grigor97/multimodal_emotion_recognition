from preprocessing import *

# enterfece_data_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/enterface_database"
# enterface_pre_processed_data_path = '/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/pre_processed_data/enterface'
enterfece_data_path = "/home/student/keropyan/data/enterface_database/"
enterface_pre_processed_data_path = "/home/student/keropyan/data/pre_processed_data/iemocap/"


# iemocap = IemocapData()
# iemocap.divide_videos_to_clips()
# iemocap.extract_video_frames(29)

get_enterface_paths(enterfece_data_path, enterface_pre_processed_data_path)
