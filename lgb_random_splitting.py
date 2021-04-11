import random
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import KFold, cross_val_score
from utils.nn_utils import *


config = load_cfg('configs/config_paths.yml')

logs_path = config['logs']['logs_path']
# train_data, test_data = load_video_data(config)
train_data, val_data, test_data = load_subset_labels_data(config)

audio_train, pic_train, labels_train = train_data
audio_test, pic_test, labels_test = test_data

data_x = np.vstack([audio_train, audio_test])
data_y = np.hstack([labels_train, labels_test])
data = np.hstack([data_x, data_y.reshape(-1, 1)])
print("shape of data_x is {} and shape of data_y is {}, data shape {}".format(data_x.shape, data_y.shape, data.shape))

n = data.shape[0]
tr_s = n*80//100

random.seed(14)
pop = range(n)
train_ind = np.array(random.sample(pop, tr_s))
test_ind = np.array(list(set(pop).difference(set(train_ind))))

train = data[train_ind]
test = data[test_ind]
train_x = train[:, :-1]
train_y = train[:, -1]

test_x = test[:, :-1]
test_y = test[:, -1]


random_state = 22
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)


def gb_mse_cv(params, cv=kf, tr_x=train_x, tr_y=train_y):
    # the function gets a set of variable parameters in "params"
    params = {'n_estimators': int(params['n_estimators']),
              'learning_rate': params['learning_rate'],
              'max_depth': int(params['max_depth']),
              'boosting_type': params['boosting_type'],
              'objective': params['objective'], }

    classifier = LGBMClassifier(**params)

    score = -cross_val_score(classifier, tr_x, tr_y, scoring='accuracy', cv=cv).mean()

    return score


n_iter = 10
# possible values of parameters
space = {'n_estimators': hp.quniform('n_estimators', 20, 5000, 400),
         'max_depth': hp.quniform('max_depth', 10, 150, 20),
         'learning_rate': hp.loguniform('learning_rate', -5, 0),
         'boosting_type': 'gbdt',  # GradientBoostingDecisionTree
         'objective': 'multiclass',  # Multi-class target feature
         }

# trials will contain logging information
trials = Trials()

best = fmin(fn=gb_mse_cv,  # function to optimize
            space=space,
            algo=tpe.suggest,  # optimization algorithm, hyperopt will select its parameters automatically
            max_evals=n_iter,  # maximum number of iterations
            trials=trials,  # logging
            rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
            )


clf = LGBMClassifier(boosting_type='gbdt', objective='multiclass',
                     learning_rate=best['learning_rate'],
                     max_depth=int(best['max_depth']),
                     n_estimators=int(best['n_estimators']))

clf.fit(train_x, train_y)
train_acc = clf.score(train_x, train_y)
print("train accuracy is   ", train_acc)

test_acc = clf.score(test_x, test_y)
print("test accuracy is   ", test_acc)

print("best params are {}".format(best))

clf.booster_.save_model('logs/audio_lgb_random_split.txt')

with open('logs/lightgbm_res_for_random_split.txt', 'w') as f:
    f.write("train accuracy is ")
    f.write(str(train_acc))
    f.write('\n')
    f.write("test accuracy is ")
    f.write(str(test_acc))
    f.write('\n')
    f.write("best params are ")
    f.write(str(best))
    f.write('\n')
