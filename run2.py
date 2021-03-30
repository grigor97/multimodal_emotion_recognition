from preprocessing import *

save_train_data_path = "../data/preprocessed_data/train_data/"
save_test_data_path = "../data/preprocessed_data/test_data/"
save_name1 = 'final_train_paths1.csv'
train_paths = "../data/preprocessed_data/filtered_emotions_paths/train_paths.csv"

train_df_paths = pd.read_csv(train_paths)

prepare_whole_data(train_df_paths.iloc[:3000], save_train_data_path, save_name1)
