from preprocessing import *

save_train_data_path = "/home/student/keropyan/data/preprocessed_data/train_data/"
save_test_data_path = "/home/student/keropyan/data/preprocessed_data/test_data/"
save_name2 = 'final_train_path2.csv'
train_paths = "/home/student/keropyan/data/preprocessed_data/filtered_emotions_paths/train_paths.csv"

train_df_paths = pd.read_csv(train_paths)

prepare_whole_data(train_df_paths.iloc[3000:6000], save_train_data_path, save_name2)
