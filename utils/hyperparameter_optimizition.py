import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import *
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# Define the LSTM model
def create_audio_lstm_model(optimizer, train_dim=162, output_dim=7, lstm_length=250):
    """
    Creates lstm model for audio data
    :param optimizer: optimizer algorithm
    :param train_dim: data simension
    :param output_dim: output classes dimension
    :param lstm_length: lstm length. For 550 params count is 1.2M and for the 250 is 0.27M
    :return: returns created model
    """
    model = Sequential()
    model.add(LSTM(lstm_length, return_sequences=False, input_shape=(train_dim, 1)))
    # model.add(LSTM(1280, return_sequences=False, input_shape=(128, 1)))

    model.add(Dense(64))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    # print(model.summary())

    # Configures the model for training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def grid_search_lstm(cfg):
    logs_path = cfg['logs']['logs_path']

    # loading datasets
    train_x = np.load(cfg['data']['train_xp'])
    train_y = np.load(cfg['data']['train_yp'])
    test_x = np.load(cfg['data']['test_xp'])
    test_y = np.load(cfg['data']['test_yp'])

    # normalizing datasets
    train_mean = train_x.mean(axis=0)
    train_x -= train_mean
    test_x -= train_mean
    train_x / train_x.sum(axis=1).reshape((train_x.shape[0], 1))
    test_x / test_x.sum(axis=1).reshape((test_x.shape[0], 1))

    print("shape of train_x is {} and shape of train_y is {}".format(train_x.shape, train_y.shape))
    print("shape of test_x is {} and shape of test_y is {}".format(test_x.shape, test_y.shape))

    labels_train_y = to_categorical(train_y)
    labels_test_y = to_categorical(test_y)

    # create model
    model = KerasClassifier(build_fn=create_audio_lstm_model, epochs=150, batch_size=64, verbose=0)
    # define the grid search parameters
    optimizers = ['SGD', 'RMSprop', 'Adam']
    param_grid = dict(optimizer=optimizers)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
    grid_result = grid.fit(train_x, labels_train_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # Evaluate the model
    loss, acc = model.score(np.expand_dims(test_x, -1), labels_test_y)
    print("model test accuracy: {:5.2f}%".format(100 * acc))
    print("model test loss: {:5.2f}%".format(loss))

    with open(logs_path + '/' + 'audio_lstm_opt/' + 'res.txt', 'w') as f:
        f.write("test accuracy and loss are ")
        f.write(str(acc) + ' ' + str(loss))
        f.write('\n')
        f.write("Best: {} using {}".format(str(grid_result.best_score_), str(grid_result.best_params_)))
