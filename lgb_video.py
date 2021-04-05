from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import plot_confusion_matrix

from utils.nn_utils import *


def save_cm(cm, path):
    plt.figure()
    plt.matshow(cm)
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(path)


config = load_cfg('configs/config_paths.yml')

logs_path = config['logs']['logs_path']
# train_data, test_data = load_video_data(config)
train_data, test_data = load_subset_labels_data(config)

train_x, pic_train, train_y = train_data
test_x, pic_test, test_y = test_data

pic_tr = pic_train.max(axis=3).mean(axis=2).reshape((pic_train.shape[0], 50, 1))
pic_te = pic_test.max(axis=3).mean(axis=2).reshape((pic_test.shape[0], 50, 1))

train_x = np.vstack([train_x, pic_tr])
test_x = np.vstack([test_x, pic_te])

print("shape of train_x is {} and shape of train_y is {}".format(train_x.shape, train_y.shape))
print("shape of test_x is {} and shape of test_y is {}".format(test_x.shape, test_y.shape))

random_state = 22
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)


def gb_mse_cv(params, cv=kf, tr_x=train_x, tr_y=train_y):
    # the function gets a set of variable parameters in "param"
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

clf.booster_.save_model('logs/audio_lgb.txt')

conf_matrix = plot_confusion_matrix(clf, test_x, test_y)
save_cm(conf_matrix.confusion_matrix, 'logs/audio_lgb_confusion_matrix.jpg')

with open('logs/lightgbm_res.txt', 'w') as f:
    f.write("train accuracy is ")
    f.write(str(train_acc))
    f.write('\n')
    f.write("test accuracy is ")
    f.write(str(test_acc))
    f.write('\n')
    f.write("best params are ")
    f.write(str(best))
    f.write('\n')
