import os

from tensorflow.keras import activations
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import losses_utils
from itertools import product
from utils.nn_utils import *

import numpy as np
from sklearn.metrics import classification_report


class WeightedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):

    def __init__(
        self,
        weights,
        from_logits=False,
        label_smoothing=0,
        reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
        name='categorical_crossentropy',
    ):
        super().__init__(
            from_logits, label_smoothing, reduction, name=f"weighted_{name}"
        )
        self.weights = weights

    def call(self, y_true, y_pred):
        weights = self.weights
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.reshape(
            y_pred_max, (K.shape(y_pred)[0], 1))
        y_pred_max_mat = K.cast(
            K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (
                weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return super().call(y_true, y_pred) * final_mask


def decay_schedule(epoch, lr):
    if epoch < 5:
        lr *= 1.02
    else:
        lr *= 0.98
    return lr


def run_video_model(model_name,
                    train_data,
                    val_data,
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
    :param val_data: validation data
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
    audio_val, pic_val, labels_val = val_data
    audio_test, pic_test, labels_test = test_data
    labels_train_y = to_categorical(labels_train)
    labels_val_y = to_categorical(labels_val)
    labels_test_y = to_categorical(labels_test)

    audio_train_dim = audio_train.shape[1]
    output_dim = labels_train_y.shape[1]

    test = ({'audio_input': audio_test, 'pic_input': pic_test}, labels_test_y)
    lr_scheduler = LearningRateScheduler(decay_schedule)
    # val = ({'audio_input': audio_val, 'pic_input': pic_val}, labels_val_y)
    audio_train = np.vstack((audio_train, audio_val))
    pic_train = np.vstack((pic_train, pic_val))
    labels_train_y = np.vstack((labels_train_y, labels_val_y))

    counts = np.sum(labels_train_y, axis=0)
    counts1 = np.sum(labels_test_y, axis=0)
    # weights = np.array(counts/np.sum(counts))
    weights = counts/np.sum(counts)
    weights1 = counts1/np.sum(counts1)
    cls_weights = {0: 10/weights[0],
                   1: 1.05*10/weights[1],
                   2: 1.1*10/weights[2],
                   3: 10/weights[3]}
    print("labels percantages -->> ", weights)
    print("labels percantages on test -->> ", weights1)

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
    elif model_name == 'testing':
        model = create_video_testing_model(opt, audio_train_dim, pic_train[0].shape, output_dim)
    elif model_name == 'video_batchnorm_cnn':
        model = create_video_batchnorm_cnn_model(opt, audio_train_dim, pic_train[0].shape, output_dim, weights)
    else:
        print("sorry you do not have such a {} model".format(model_name))
        return

    print("{} model summary is \n {}".format(model_name, model.summary()))

    checkpoint_path = logs_path + model_name + str(num_epochs) + optimizer + str(lr) + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)

    if restore:
        model.load_weights(checkpoint_path)
        num_epochs = num_epochs - continue_at + 1

    # tr_audio_x, tr_pic_x, tr_y, val_audio_x, val_pic_x, val_y = random_split(audio_train, pic_train, labels_train_y)

    # print("train, val and test shapes are {} {} {}, {} {} {}, {} {} {}".
    #       format(tr_audio_x.shape, tr_pic_x.shape, tr_y.shape,
    #              val_audio_x.shape, val_pic_x.shape, val_y.shape,
    #              audio_test.shape, pic_test.shape, labels_test_y.shape))
    # model = create_video_batchnorm_cnn_model(opt, audio_train_dim, pic_train[0].shape, output_dim)

    # es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=50, mode='max')
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=120, verbose=0, mode='max')
    mcp_save = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
    # reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    train = ({'audio_input': audio_train, 'pic_input': pic_train}, labels_train_y)
    model_history = model.fit(train[0],
                              train[1],
                              batch_size=batch_size,
                              epochs=num_epochs,
                              validation_data=test,
                              class_weight=cls_weights,
                              callbacks=[lr_scheduler, earlyStopping, mcp_save, CustomEarlyStopping()])

    model.load_weights(filepath=checkpoint_dir + '/mdl_wts.hdf5')

    # Evaluate the validation
    # val_loss, val_acc = model.evaluate(val[0], val[1])
    # print("{} model val accuracy: {:5.2f}%".format(model_name, 100 * val_acc))
    # print("{} model val loss: {:5.2f}".format(model_name, val_loss))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test[0], test[1])
    print("{} model test accuracy: {:5.2f}%".format(model_name, 100 * test_acc))
    print("{} model test loss: {:5.2f}".format(model_name, test_loss))

    # model.save(checkpoint_dir + '/model.h5')
    nn_save_model_plots(model_history, checkpoint_dir)

    y_pred = model.predict(test[0])
    cm_analysis(test[1].argmax(axis=1), y_pred.argmax(axis=1),
                checkpoint_dir + '/' + model_name + '_cm.png',
                ['sad', 'neu', 'hap', 'ang'])
    np.save("logs/test_pred.npy", y_pred.argmax(axis=1))
    np.save("logs/test_actual.npy", test[1].argmax(axis=1))
    print("on test data")
    print(classification_report(test[1].argmax(axis=1),
                                y_pred.argmax(axis=1),
                                target_names=['sad', 'neu', 'hap', 'ang']))

    y_pred = model.predict(train[0])
    cm_analysis(train[1].argmax(axis=1), y_pred.argmax(axis=1),
                checkpoint_dir + '/' + model_name + '_cm_train.png',
                ['sad', 'neu', 'hap', 'ang'])

    np.save("logs/train_pred.npy", y_pred.argmax(axis=1))
    np.save("logs/train_actual.npy", train[1].argmax(axis=1))
    print("on train data")
    print(classification_report(train[1].argmax(axis=1),
                                y_pred.argmax(axis=1),
                                target_names=['sad', 'neu', 'hap', 'ang']))

    train_acc = model_history.history['accuracy'][-1]
    # val_acc = model_history.history['val_accuracy'][-1]

    with open(checkpoint_dir + '/' + model_name + '_res.txt', 'w') as f:
        f.write("test accuracy and loss are ")
        f.write(str(test_acc) + ' ' + str(test_loss))
        f.write('\n')
        f.write("train accuracy and loss are ")
        f.write(str(train_acc))
        f.write('\n')

    return test_acc


def create_video_batchnorm_cnn_model(optimizer, audio_dim, pic_shape, output_dim, weights):
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
    # audio_x = Conv1D(128, 8, padding='same', activation=activations.relu)(audio_input)

    audio_x = Conv1D(64, 8, padding='valid')(audio_input)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    # audio_x = Dropout(0.25)(audio_x)
    audio_x = MaxPooling1D(pool_size=8)(audio_x)

    audio_x = Conv1D(32, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    # audio_x = Conv1D(64, 8, padding='same')(audio_x)
    # audio_x = BatchNormalization()(audio_x)
    # audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(32, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)

    audio_x = Conv1D(64, 8, padding='same')(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    # audio_x = Dropout(0.25)(audio_x)
    audio_x = MaxPooling1D(pool_size=8)(audio_x)

    # audio_x = Conv1D(32, 8, padding='same')(audio_x)
    # audio_x = Activation(activations.relu)(audio_x)

    # audio_x = Conv1D(32, 8, padding='same')(audio_x)
    # audio_x = Activation(activations.relu)(audio_x)

    audio_x = Flatten()(audio_x)
    audio_x = Dense(32)(audio_x)
    audio_x = Activation(activations.relu)(audio_x)
    audio_x = Dropout(0.5)(audio_x)
    # end of audio network part

    # pictures network part
    pic_input = Input(shape=pic_shape, name='pic_input')

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="valid")(pic_input)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    # pic_x = Dropout(0.25)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="valid")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    # pic_x = Dropout(0.25)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    # pic_x = Conv2D(16, kernel_size=(3, 3), padding="valid")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    # pic_x = Activation(activations.relu)(pic_x)
    # pic_x = Dropout(0.25)(pic_x)

    pic_x = Conv2D(32, kernel_size=(3, 3), padding="valid")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    # pic_x = Dropout(0.25)(pic_x)
    # pic_x = MaxPool2D()(pic_x)

    # pic_x = Conv2D(16, kernel_size=(3, 3), padding="valid")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    # pic_x = Activation(activations.relu)(pic_x)

    # pic_x = Conv2D(64, kernel_size=(3, 3), padding="same")(pic_x)
    # # pic_x = BatchNormalization()(pic_x)
    # pic_x = Activation(activations.relu)(pic_x)
    # pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="valid")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    # pic_x = Dropout(0.3)(pic_x)
    # pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="valid")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)

    pic_x = Flatten()(pic_x)
    # pic_x = Dense(64, activation='relu')(pic_x)
    # pic_x = Dropout(0.25)(pic_x)
    pic_x = Dense(6, activation='relu')(pic_x)
    pic_x = Dropout(0.75)(pic_x)
    # end of pictures network part

    # concatenation of two networks
    x = concatenate([audio_x, pic_x])

    x = Dense(32)(x)
    # x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = Dropout(0.5)(x)

    # x = Dense(16, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # TODO improve
    # BATCHNORM_CNN
    out = Dense(output_dim, activation='softmax')(x)

    model = Model(
        inputs=[audio_input, pic_input],
        outputs=[out]
    )

    # tf.keras.utils.plot_model(model, show_shapes=True)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # loss=WeightedCategoricalCrossentropy(weights),
        metrics=['accuracy']
    )

    return model


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

    # pic_x = Conv2D(64, kernel_size=(3, 3), padding="same")(pic_x)
    # # pic_x = BatchNormalization()(pic_x)
    # pic_x = Activation(activations.relu)(pic_x)
    # pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(32, kernel_size=(3, 3), padding="same")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Flatten()(pic_x)
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


def create_video_testing_model(optimizer, audio_dim, pic_shape, output_dim, lstm_length=250):
    """
    Creates cnn model for video data
    :param lstm_length: lstm size
    :param optimizer: optimizer for a cnn
    :param audio_dim: dimension of audio data
    :param pic_shape: shape of the pictures
    :param output_dim: dimension of output
    :return: compiled cnn model
    """
    # audio network part
    audio_input = Input(shape=(audio_dim, 1), name='audio_input')
    audio_x = LSTM(lstm_length, return_sequences=False, activation='relu')(audio_input)

    audio_x = Dense(64)(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation('relu')(audio_x)

    # audio_x = Dense(32)(audio_x)
    # audio_x = BatchNormalization()(audio_x)
    # audio_x = Activation('relu')(audio_x)

    audio_x = Dense(32)(audio_x)
    audio_x = BatchNormalization()(audio_x)
    audio_x = Activation('relu')(audio_x)
    # end of audio network part

    # pictures network part
    pic_input = Input(shape=pic_shape, name='pic_input')

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="valid")(pic_input)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    # pic_x = Dropout(0.25)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(32, kernel_size=(3, 3), padding="valid")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    # pic_x = Dropout(0.25)(pic_x)
    pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(32, kernel_size=(3, 3), padding="valid")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    # pic_x = Dropout(0.25)(pic_x)
    # pic_x = MaxPool2D()(pic_x)

    # pic_x = Conv2D(64, kernel_size=(3, 3), padding="same")(pic_x)
    # # pic_x = BatchNormalization()(pic_x)
    # pic_x = Activation(activations.relu)(pic_x)
    # pic_x = MaxPool2D()(pic_x)

    pic_x = Conv2D(16, kernel_size=(3, 3), padding="valid")(pic_x)
    pic_x = BatchNormalization()(pic_x)
    pic_x = Activation(activations.relu)(pic_x)
    pic_x = Dropout(0.3)(pic_x)
    # pic_x = MaxPool2D()(pic_x)

    # pic_x = Conv2D(, kernel_size=(3, 3), padding="valid")(pic_x)
    # pic_x = BatchNormalization()(pic_x)
    # pic_x = Activation(activations.relu)(pic_x)

    pic_x = Flatten()(pic_x)
    # pic_x = Dense(64, activation='relu')(pic_x)
    # pic_x = Dropout(0.25)(pic_x)
    pic_x = Dense(8, activation='relu')(pic_x)
    pic_x = Dropout(0.3)(pic_x)
    # end of pictures network part

    # concatenation of two networks
    x = concatenate([audio_x, pic_x])

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    # x = Dropout(0.25)(x)
    # x = Dense(16, activation='relu')(x)
    # x = Dropout(0.2)(x)
    out = Dense(output_dim, activation='softmax')(x)

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
