from preprocessing import *

save_train_data_path = "/home/student/keropyan/data/preprocessed_data/train_data/"
save_name1 = 'final_train_paths1.csv'
save_name2 = 'final_train_paths2.csv'
save_name = 'final_train_paths2.csv'

train_paths = "/home/student/keropyan/data/preprocessed_data/filtered_emotions_paths/train_paths.csv"

train_df_paths = pd.read_csv(train_paths)

prepare_whole_data(train_df_paths.iloc[5000:], save_train_data_path, save_name2)

train1 = pd.read_csv(save_train_data_path + save_name1)
train2 = pd.read_csv(save_train_data_path + save_name2)
train = pd.concat([train1, train2])
train.to_csv(save_train_data_path + save_name)

