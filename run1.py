from preprocessing import *

save_test_data_path = "/home/student/keropyan/data/preprocessed_data/test_data/"

test_paths = "/home/student/keropyan/data/preprocessed_data/filtered_emotions_paths/test_paths.csv"

test_df_paths = pd.read_csv(test_paths)

prepare_whole_data(test_df_paths, save_test_data_path, 'final_test_paths.csv')
