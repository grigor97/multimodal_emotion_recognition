import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, anneal, Trials
from sklearn.model_selection import KFold, cross_val_score

final_emos = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3, 'fru': 4, 'exc': 5, 'oth': 6}

train_paths = "/home/student/keropyan/data/preprocessed_data/train_data/final_train_paths.csv"
test_paths = "/home/student/keropyan/data/preprocessed_data/test_data/final_test_paths.csv"

train_ps = pd.read_csv(train_paths)
test_ps = pd.read_csv(test_paths)
train_ps.dropna(inplace=True)
test_ps.dropna(inplace=True)


def get_audio_features(df_paths):
    x = np.array([]).reshape((0, 162))
    y = []
    for i, row in df_paths.iterrows():
        # print(row[1])
        npy_path = row[1]
        lb = final_emos[row[2]]
        label = [lb, lb, lb, lb]
        features = np.load(npy_path)

        x = np.vstack([x, features])
        y.extend(label)

    y = np.array(y)
    return x, y


train_x, train_y = get_audio_features(train_ps)
test_x, test_y = get_audio_features(test_ps)


print("shape of train_x is {} and shape of train_y is {}".format(train_x.shape, train_y.shape))
print("shape of test_x is {} and shape of test_y is {}".format(test_x.shape, test_y.shape))

# This is my DL course project params:)
clf = LGBMClassifier(boosting_type='gbdt', objective='multiclass',
                     learning_rate=0.42352561266844885,
                     max_depth=70,
                     n_estimators=80)

clf.fit(train_x, train_y)
preds = clf.predict(train_y)
acc = (train_y == preds).sum()/preds.size
print("train accuracy is   ", acc)

preds = clf.predict(test_x)
acc = (test_y == preds).sum()/preds.size
print("test accuracy is   ", acc)


