import numpy as np
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import KFold, cross_val_score

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def save_cm(cm, path):
    fig = plt.figure()
    plt.matshow(cm)
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(path)

# final_emos = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3, 'fru': 4, 'exc': 5, 'oth': 6}
#
# train_paths = "/home/student/keropyan/data/preprocessed_data/train_data/final_train_paths.csv"
# test_paths = "/home/student/keropyan/data/preprocessed_data/test_data/final_test_paths.csv"
#
# train_ps = pd.read_csv(train_paths)
# test_ps = pd.read_csv(test_paths)
# train_ps.dropna(inplace=True)
# test_ps.dropna(inplace=True)
#
#
# def get_audio_features(df_paths):
#     x = np.array([]).reshape((0, 162))
#     y = []
#     for i, row in df_paths.iterrows():
#         # print(row[1])
#         npy_path = row[1]
#         lb = final_emos[row[2]]
#         label = [lb, lb, lb, lb]
#         features = np.load(npy_path)
#
#         x = np.vstack([x, features])
#         y.extend(label)
#
#     y = np.array(y)
#     return x, y
#
#
# train_x, train_y = get_audio_features(train_ps)
# test_x, test_y = get_audio_features(test_ps)

train_xp = "/home/student/keropyan/data/preprocessed_data/train_data/train_x.npy"
train_yp = "/home/student/keropyan/data/preprocessed_data/train_data/train_y.npy"
test_xp = "/home/student/keropyan/data/preprocessed_data/train_data/test_x.npy"
test_yp = "/home/student/keropyan/data/preprocessed_data/train_data/test_y.npy"
# np.save(train_xp, train_x)
# np.save(train_yp, train_y)
# np.save(test_xp, test_x)
# np.save(test_yp, test_y)


# loading datasets
train_x = np.load(train_xp)
train_y = np.load(train_yp)
test_x = np.load(test_xp)
test_y = np.load(test_yp)

# normalizing datasets
train_mean = train_x.mean(axis=0)
train_x -= train_mean
test_x -= train_mean
train_x/train_x.sum(axis=1).reshape((train_x.shape[0], 1))
test_x/test_x.sum(axis=1).reshape((test_x.shape[0], 1))

print("shape of train_x is {} and shape of train_y is {}".format(train_x.shape, train_y.shape))
print("shape of test_x is {} and shape of test_y is {}".format(test_x.shape, test_y.shape))

random_state = 22
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)


def gb_mse_cv(params, random_state=random_state, cv=kf, X=train_x, y=train_y):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']),
              'learning_rate': params['learning_rate'],
              'max_depth': int(params['max_depth']),
              'boosting_type': params['boosting_type'],
              'objective': params['objective'], }

    clf = LGBMClassifier(**params)

    score = -cross_val_score(clf, X, y, scoring='accuracy', cv=cv).mean()

    return score


n_iter = 100
# possible values of parameters
space = {'n_estimators': hp.quniform('n_estimators', 20, 5000, 40),
         'max_depth' : hp.quniform('max_depth', 10, 310, 20),
         'learning_rate': hp.loguniform('learning_rate', -5, 0),
         'boosting_type': 'gbdt', # GradientBoostingDecisionTree
         'objective': 'multiclass', # Multi-class target feature
         }

# trials will contain logging information
trials = Trials()

best = fmin(fn=gb_mse_cv,  # function to optimize
            space=space,
            algo=tpe.suggest,  # optimization algorithm, hyperopt will select its parameters automatically
            max_evals=n_iter,  # maximum number of iterations
            trials=trials,  # logging
            rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility
            )


clf = LGBMClassifier(boosting_type='gbdt', objective='multiclass',
                     learning_rate=best['learning_rate'],
                     max_depth=int(best['max_depth']),
                     n_estimators=int(best['n_estimators']))


clf.fit(train_x, train_y)
# preds = clf.predict(train_x)
# train_acc = (train_y.reshape(1, -1) == preds.reshape(1, -1)).sum()/preds.size
train_acc = clf.score(train_x, train_y)
print("train accuracy is   ", train_acc)

# preds = clf.predict(test_x)
# test_acc = (test_y.reshape(1, -1) == preds.reshape(1, -1)).sum()/preds.size
test_acc = clf.score(test_x, test_y)
print("test accuracy is   ", test_acc)

print("best params are {}".format(best))

clf.booster_.save_model('logs/audio_lgb.txt')

conf_matrix = plot_confusion_matrix(clf, test_x, test_y)
save_cm(conf_matrix.confusion_matrix, 'logs/audio_lgb_confusion_matrix.jpg')

with open('logs/lightgbm_res_for_random_split.txt', 'w') as f:
    f.write("train accuracy is ")
    f.write(train_acc)
    f.write('\n')
    f.write("test accuracy is ")
    f.write(test_acc)
    f.write('\n')
    f.write("best params are ")
    f.write(best)
    f.write('\n')
