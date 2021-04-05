import os

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model

from utils.nn_utils import *


def run_video_model(model_name,
                    train_data,
                    test_data,
                    logs_path,
                    restore,
                    continue_at,
                    optimizer,
                    lr,
                    batch_size,
                    num_epochs):
    """
    Runs model for video data
    :param model_name: name of the model
    :param train_data: train data
    :param test_data: test data
    :param logs_path: path of logs
    :param restore: restore training  or not
    :param continue_at: if restore then in which point it should start
    :param optimizer: optimizer for a model
    :param lr: learning rate
    :param batch_size: batch size
    :param num_epochs: number of epochs
    :return: accuracy
    """
    audio_train, pic_train, labels_train = train_data
    audio_test, pic_test, labels_test = test_data
    labels_train_y = to_categorical(labels_train)
    labels_test_y = to_categorical(labels_test)

    audio_train_dim = audio_train.shape[1]
    output_dim = labels_train_y.shape[1]

    # opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    # opt = keras.optimizers.Adam(lr=0.0001)
    # opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    if optimizer == 'Adam':
        opt = tf.keras.optimizers.Adam(lr=lr)
    elif optimizer == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(lr=lr)
    elif optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(lr=lr)
    else:
        print("sorry your optimizer is not correct!")
        return

    if model_name == 'video_cnn':
        model = create_video_cnn_model(opt, audio_train_dim, pic_train[0].shape, output_dim)
    elif model_name == 'video_batchnorm_cnn':
        model = create_video_batchnorm_cnn_model(opt, audio_train_dim, pic_train[0].shape, output_dim)
    elif model_name == 'video_big_cnn':
        model = create_video_cnn_model(opt, audio_train_dim, pic_train[0].shape, output_dim)
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

    if restore:
        model.load_weights(checkpoint_path)
        num_epochs = num_epochs - continue_at + 1

    model_history = model.fit(
        {'audio_input': audio_train, 'pic_input': pic_train},
        labels_train_y,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[cp_callback]
    )

    # Evaluate the model
    loss, acc = model.evaluate({'audio_input': audio_test, 'pic_input': pic_test}, labels_test_y)
    print("{} model test accuracy: {:5.2f}%".format(model_name, 100 * acc))
    print("{} model test loss: {:5.2f}".format(model_name, loss))

    model.save(checkpoint_dir + '/model.h5')
    nn_save_model_plots(model_history, checkpoint_dir)

    train_acc = model_history.history['accuracy'][-1]
    val_acc = model_history.history['val_accuracy'][-1]

    with open(checkpoint_dir + '/' + model_name + '_res.txt', 'w') as f:
        f.write("test accuracy and loss are ")
        f.write(str(acc) + ' ' + str(loss))
        f.write('\n')
        f.write("val accuracy and loss are ")
        f.write(str(val_acc))
        f.write('\n')
        f.write("train accuracy and loss are ")
        f.write(str(train_acc))
        f.write('\n')

    return acc


def create_video_cnn_model(optimizer, audio_dim, pic_shape, output_dim):
    """
    Creates cnn model for video data
    :param optimizer: optimizer for a cnn
    :param audio_dim: dimension of audio data
    :param pic_shape: shape of the pictures
    :param output_dim: dimension of output
    :return: compiled cnn model
    """
    # audio network part
    audio_input = Input(shape=(audio_dim, 1), name='audio_input')
    audio_x = Conv1D(128, 8, padding='same', activation=activations.relu)(audio_input)

    audio_x = Conv1D(128, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    # audio_x = Dropout(0.25)(audio_x)
    audio_x = MaxPooling1D(pool_size=8)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    # audio_x = Dropout(0.25)(audio_x)
    audio_x = MaxPooling1D(pool_size=8)(audio_x)

    audio_x = Conv1D(32, 8, padding='same')(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(32, 8, padding='same')(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Flatten()(audio_x)
    audio_x = Dense(32)(audio_x)
    # end of audio network part

    # pictures network part
    pic_input = Input(shape=pic_shape, name='pic_input')

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="same")(pic_input)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(32, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(32, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Flatten()(pic_x)
    pic_x = Dense(64, activation='relu')(pic_x)
    pic_x = Dropout(0.25)(pic_x)
    pic_x = Dense(32, activation='relu')(pic_x)
    # end of pictures network part

    # concatenation of two networks
    x = concatenate([audio_x, pic_x])

    x = Dense(32, activation='relu')(x)

    out = Dense(output_dim, activation='relu')(x)

    model = Model(
        inputs=[audio_input, pic_input],
        outputs=[out]
    )

    # tf.keras.utils.plot_model(model, show_shapes=True)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def create_video_batchnorm_cnn_model(optimizer, audio_dim, pic_shape, output_dim):
    """
    Creates cnn model for video data
    :param optimizer: optimizer for a cnn
    :param audio_dim: dimension of audio data
    :param pic_shape: shape of the pictures
    :param output_dim: dimension of output
    :return: compiled cnn model
    """
    # audio network part
    audio_input = Input(shape=(audio_dim, 1), name='audio_input')
    audio_x = Conv1D(128, 8, padding='same', activation=activations.relu)(audio_input)

    audio_x = Conv1D(128, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    # audio_x = Dropout(0.25)(audio_x)
    audio_x = MaxPooling1D(pool_size=8)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    # audio_x = Dropout(0.25)(audio_x)
    audio_x = MaxPooling1D(pool_size=8)(audio_x)

    audio_x = Conv1D(32, 8, padding='same')(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(32, 8, padding='same')(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Flatten()(audio_x)
    audio_x = Dense(32)(audio_x)
    # end of audio network part

    # pictures network part
    pic_input = Input(shape=pic_shape, name='pic_input')

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="same")(pic_input)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(32, kernel_size=(3, 3), padding="same")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(32, kernel_size=(3, 3), padding="same")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="same")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Flatten()(pic_x)
    pic_x = Dense(64, activation='relu')(pic_x)
    pic_x = Dropout(0.25)(pic_x)
    pic_x = Dense(32, activation='relu')(pic_x)
    # end of pictures network part

    # concatenation of two networks
    x = concatenate([audio_x, pic_x])

    x = Dense(32, activation='relu')(x)

    out = Dense(output_dim, activation='relu')(x)

    model = Model(
        inputs=[audio_input, pic_input],
        outputs=[out]
    )

    # tf.keras.utils.plot_model(model, show_shapes=True)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def create_video_big_cnn_model(optimizer, audio_dim, pic_shape, output_dim):
    """
    Creates cnn model for video data
    :param optimizer: optimizer for a cnn
    :param audio_dim: dimension of audio data
    :param pic_shape: shape of the pictures
    :param output_dim: dimension of output
    :return: compiled cnn model
    """
    # audio network part
    audio_input = Input(shape=(audio_dim, 1), name='audio_input')
    audio_x = Conv1D(128, 8, padding='same', activation=activations.relu)(audio_input)

    audio_x = Conv1D(128, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    # audio_x = Dropout(0.25)(audio_x)
    audio_x = MaxPooling1D(pool_size=8)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    # audio_x = Dropout(0.25)(audio_x)
    audio_x = MaxPooling1D(pool_size=8)(audio_x)

    audio_x = Conv1D(32, 8, padding='same')(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(32, 8, padding='same')(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Flatten()(audio_x)
    audio_x = Dense(32)(audio_x)
    # end of audio network part

    # pictures network part
    pic_input = Input(shape=pic_shape, name='pic_input')

    pic_x = Conv2D(64, kernel_size=(3, 3), padding="same")(pic_input)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(64, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(128, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(128, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(64, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(64, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Flatten()(pic_x)
    pic_x = Dense(64, activation='relu')(pic_x)
    pic_x = Dropout(0.25)(pic_x)
    pic_x = Dense(32, activation='relu')(pic_x)
    # end of pictures network part

    # concatenation of two networks
    x = concatenate([audio_x, pic_x])

    x = Dense(32, activation='relu')(x)

    out = Dense(output_dim, activation='relu')(x)

    model = Model(
        inputs=[audio_input, pic_input],
        outputs=[out]
    )

    # tf.keras.utils.plot_model(model, show_shapes=True)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model
