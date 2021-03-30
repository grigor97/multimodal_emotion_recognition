import numpy as np
import random
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import KFold, cross_val_score


train_xp = "../data/preprocessed_data/train_data/train_x.npy"
train_yp = "../data/preprocessed_data/train_data/train_y.npy"
test_xp = "../data/preprocessed_data/train_data/test_x.npy"
test_yp = "../data/preprocessed_data/train_data/test_y.npy"

train_x = np.load(train_xp)
train_y = np.load(train_yp)
test_x = np.load(test_xp)
test_y = np.load(test_yp)

data_x = np.vstack([train_x, test_x])
data_y = np.hstack([train_y, test_y])
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


def gb_mse_cv(params, random_state=random_state, cv=kf, X=train_x, y=train_y):
    # the function gets a set of variable parameters in "params"
    params = {'n_estimators': int(params['n_estimators']),
              'learning_rate': params['learning_rate'],
              'max_depth': int(params['max_depth']),
              'boosting_type': params['boosting_type'],
              'objective': params['objective'], }

    clf = LGBMClassifier(**params)

    score = -cross_val_score(clf, X, y, scoring='accuracy', cv=cv).mean()

    return score


n_iter = 10
# possible values of parameters
space = {'n_estimators': hp.quniform('n_estimators', 20, 5000, 400),
         'max_depth' : hp.quniform('max_depth', 10, 150, 20),
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
            rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility
            )


clf = LGBMClassifier(boosting_type='gbdt', objective='multiclass',
                     learning_rate=best['learning_rate'],
                     max_depth=int(best['max_depth']),
                     n_estimators=int(best['n_estimators']))


clf.fit(train_x, train_y)
preds = clf.predict(train_x)
train_acc = (train_y.reshape(1, -1) == preds.reshape(1, -1)).sum()/preds.size
print("train accuracy is   ", train_acc)

preds = clf.predict(test_x)
test_acc = (test_y.reshape(1, -1) == preds.reshape(1, -1)).sum()/preds.size
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
