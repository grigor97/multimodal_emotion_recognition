import os
from utils.nn_utils import *

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, tol=15, patience=5):
        super(CustomEarlyStopping, self).__init__()
        self.tol = tol
        self.patience = patience
        # self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        v_acc = logs.get('val_accuracy')
        t_acc = logs.get('accuracy')

        if epoch < 15:
            return

        if t_acc - v_acc < self.tol:
            self.wait += 1
        else:
            self.wait = 0

        if self.wait > self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def run_model(model_name,
              train_data,
              val_data,
              test_data,
              logs_path,
              restore=False,
              continue_at=1,
              optimizer='Adam',
              lr=1e-3,
              batch_size=64,
              num_epochs=100):
    audio_train, pic_train, labels_train = train_data
    audio_val, pic_val, labels_val = val_data
    audio_test, pic_test, labels_test = test_data
    audio_train = np.expand_dims(audio_train, -1)
    audio_val = np.expand_dims(audio_val, -1)
    audio_test = np.expand_dims(audio_test, -1)
    labels_train_y = to_categorical(labels_train)
    labels_val_y = to_categorical(labels_val)
    labels_test_y = to_categorical(labels_test)

    audio_train_dim = audio_train.shape[1]
    output_dim = labels_train_y.shape[1]

    if optimizer == 'Adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    elif optimizer == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(lr=lr)
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(lr=lr)
    else:
        print("sorry your optimizer is not correct!")
        return

    if model_name == 'audio_cnn':
        model = create_audio_cnn_model(opt, audio_train_dim, output_dim)
    elif model_name == 'audio_lstm':
        model = create_audio_lstm_model(opt, audio_train_dim, output_dim)
    elif model_name == 'audio_blstm':
        model = create_audio_blstm_model(opt, audio_train_dim, output_dim)
    elif model_name == 'audio_stacked_lstm':
        model = create_audio_stacked_lstm_model(opt, audio_train_dim, output_dim)
    else:
        print("sorry you do not have such a {} model".format(model_name))
        return

    print("{} model summary is \n {}".format(model_name, model.summary()))

    checkpoint_path = logs_path + model_name + str(num_epochs) + optimizer + str(lr) + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=20, mode='max')

    if restore:
        model.load_weights(checkpoint_path)
        num_epochs = num_epochs - continue_at + 1

    # tr_audio_x, tr_pic_x, tr_y, val_audio_x, val_pic_x, val_y = random_split(audio_train, pic_train, labels_train_y)

    # print("train, val and test shapes are {} {} {}, {} {} {}, {} {} {}".
    #       format(tr_audio_x.shape, tr_pic_x.shape, tr_y.shape,
    #              val_audio_x.shape, val_pic_x.shape, val_y.shape,
    #              audio_test.shape, pic_test.shape, labels_test_y.shape))

    model_history = model.fit(audio_train,
                              labels_train_y,
                              batch_size=batch_size,
                              epochs=num_epochs,
                              validation_data=(audio_val, labels_val_y),
                              callbacks=[cp_callback, CustomEarlyStopping(), es_callback])

    # Evaluate the validation
    val_loss, val_acc = model.evaluate(audio_val, labels_val_y)
    print("{} model val accuracy: {:5.2f}%".format(model_name, 100 * val_acc))
    print("{} model val loss: {:5.2f}".format(model_name, val_loss))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(audio_test, labels_test_y)
    print("{} model test accuracy: {:5.2f}%".format(model_name, 100 * test_acc))
    print("{} model test loss: {:5.2f}".format(model_name, test_loss))

    model.save(checkpoint_dir + '/model.h5')
    nn_save_model_plots(model_history, checkpoint_dir)

    train_acc = model_history.history['accuracy'][-1]
    # val_acc = model_history.history['val_accuracy'][-1]

    with open(checkpoint_dir + '/' + model_name + '_res.txt', 'w') as f:
        f.write("test accuracy and loss are ")
        f.write(str(test_acc) + ' ' + str(test_loss))
        f.write('\n')
        f.write("val accuracy and loss are ")
        f.write(str(val_acc))
        f.write('\n')
        f.write("train accuracy and loss are ")
        f.write(str(train_acc))
        f.write('\n')

    return test_acc


def create_audio_cnn_model(optimizer, train_dim, output_dim):
    """
    Creates cnn model for audio data
    :param optimizer: nn optimizer
    :param train_dim: data dimension
    :param output_dim: output classes dimension
    :return: created model
    """
    model = Sequential()
    model.add(Conv1D(128, 8, padding='same', input_shape=(train_dim, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(64, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=8))

    model.add(Conv1D(32, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(32, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(output_dim))  # Target class number
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# def create_audio_cnn_model(optimizer, train_dim, output_dim=7):
#     """
#     Creates cnn model for audio data
#     :param optimizer: nn optimizer
#     :param train_dim: data dimension
#     :param output_dim: output classes dimension
#     :return: created model
#     """
#     model = Sequential()
#     model.add(Conv1D(256, 8, padding='same', input_shape=(train_dim, 1)))  # X_train.shape[1] = No. of Columns
#     model.add(Activation('relu'))
#     model.add(Conv1D(256, 8, padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))
#     model.add(MaxPooling1D(pool_size=(8)))
#     model.add(Conv1D(128, 8, padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv1D(128, 8, padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv1D(128, 8, padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv1D(128, 8, padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))
#     model.add(MaxPooling1D(pool_size=(8)))
#     model.add(Conv1D(64, 8, padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv1D(64, 8, padding='same'))
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     model.add(Dense(output_dim))  # Target class number
#     model.add(Activation('softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#
#     return model


# Define the LSTM model
def create_audio_lstm_model(optimizer, train_dim, output_dim, lstm_length=250):
    """
    Creates lstm model for audio data
    :param optimizer: nn optimizer
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


# Define the BLSTM model
def create_audio_blstm_model(optimizer, train_dim, output_dim, blstm_length=180):
    """
    Creates blstm model for audio data
    :param optimizer: nn optimizer
    :param train_dim: data dimension
    :param output_dim: output classes dimension
    :param blstm_length: blstm length. For 390 params count is 1.2M and for the 180 is 0.28M
    :return: returns created model
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(blstm_length, return_sequences=False), input_shape=(train_dim, 1)))
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


# Define the stacked lstm model
def create_audio_stacked_lstm_model(optimizer, train_dim, output_dim, stacked_lstm_length=150):
    """
    Creates stacked lstm model for audio data
    :param optimizer: nn optimizer
    :param train_dim: data simension
    :param output_dim: output classes dimension
    :param stacked_lstm_length: stacked lstm length. For 300 params count is 1.2M and for the 150 is 0.28M
    :return: returns created model
    """
    model = Sequential()
    model.add(LSTM(stacked_lstm_length, return_sequences=True, input_shape=(train_dim, 1)))
    model.add(LSTM(stacked_lstm_length, return_sequences=False, input_shape=(stacked_lstm_length, 1)))

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
