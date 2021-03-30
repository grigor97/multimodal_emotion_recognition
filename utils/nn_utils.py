import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


def nn_save_model_plots(model_history, save_path):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(save_path + "/loss.png")

    plt.clf()

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path + "/accuracy.png")


def create_audio_cnn_model(train_dim, output_dim):
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same', input_shape=(train_dim, 1)))  # X_train.shape[1] = No. of Columns
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(output_dim))  # Target class number
    model.add(Activation('softmax'))
    # opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    # opt = keras.optimizers.Adam(lr=0.0001)
    opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


# Define the LSTM model
def create_audio_lstm_model(train_dim, output_dim):
    model = Sequential()
    model.add(LSTM(550, return_sequences=False, input_shape=(train_dim, 1)))
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
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# Define the BLSTM model
def create_audio_blstm_model(train_dim, output_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(390, return_sequences=False), input_shape=(train_dim, 1)))
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
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# Define the BLSTM model
def create_audio_stacked_lstm_model(train_dim, output_dim):
    model = Sequential()
    model.add(LSTM(300, return_sequences=True, input_shape=(train_dim, 1)))
    model.add(LSTM(300, return_sequences=False, input_shape=(300, 1)))

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
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


def run_model(model_name, cfg):
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

    print("train and test shapes are {} {}".format(train_x.shape, test_x.shape))
    if model_name == 'audio_cnn':
        model = create_audio_cnn_model(train_x.shape[1], 7)
    elif model_name == 'audio_lstm':
        model = create_audio_lstm_model(train_x.shape[1], 7)
    elif model_name == 'audio_blstm':
        model = create_audio_blstm_model(train_x.shape[1], 7)
    elif model_name == 'audio_stacked_lstm':
        model = create_audio_stacked_lstm_model(train_x.shape[1], 7)
    else:
        print("sorry you do not have such a {} model".format(model_name))
        return

    print("{} model summary is \n".format(model_name, model.summary()))

    checkpoint_path = logs_path + model_name + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model_history = model.fit(np.expand_dims(train_x, -1),
                              labels_train_y,
                              batch_size=16,
                              epochs=150,
                              validation_split=0.15,
                              callbacks=[cp_callback])

    # Evaluate the model
    loss, acc = model.evaluate(np.expand_dims(test_x, -1), labels_test_y)
    print("{} model test accuracy: {:5.2f}%".format(model_name, 100 * acc))
    print("{} model test loss: {:5.2f}%".format(model_name, loss))

    model.save(checkpoint_dir + '/model.h5')
    nn_save_model_plots(model_history, checkpoint_dir)

    with open(checkpoint_dir + '/' + model_name + '_res.txt', 'w') as f:
        f.write("train accuracy and loss are ")
        f.write(str(acc) + ' ' + str(loss))
        f.write('\n')
