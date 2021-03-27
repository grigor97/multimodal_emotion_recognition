import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, anneal, Trials
from sklearn.model_selection import KFold, cross_val_score

final_emos = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3, 'fru': 4, 'exc': 5, 'oth': 6}

train_paths = "/home/student/keropyan/data/preprocessed_data/filtered_emotions_paths/train_paths.csv"
test_paths = "/home/student/keropyan/data/preprocessed_data/filtered_emotions_paths/test_paths.csv"

train_ps = pd.read_csv(train_paths)
test_ps = pd.read_csv(test_paths)

train_x = np.array([]).reshape((0, 162))
train_y = []
for i, row in train_ps.iterrows():
    npy_path = row[1]
    lb = final_emos[row[2]]
    label = [lb, lb, lb, lb]
    features = np.load(npy_path)

    train_x = np.vstack([train_x, features])
    train_y.extend(label)

train_y = np.array(train_y)

print("shape of train_x is {} and shape of train_y is {}".format(train_x.shape, train_y.shape))

