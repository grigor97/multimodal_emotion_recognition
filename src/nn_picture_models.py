import os

from tensorflow.keras import activations
from tensorflow.keras import Model

from utils.nn_utils import *


def run_picture_model(model_name,
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
    Runs model for picture data
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

    # audio_train_dim = audio_train.shape[1]
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

    if model_name == 'picture_cnn':
        model = create_picture_cnn_model(opt, pic_train[0].shape, output_dim)
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
    #
    # print("train, val and test shapes are {} {} {}, {} {} {}, {} {} {}".
    #       format(tr_audio_x.shape, tr_pic_x.shape, tr_y.shape,
    #              val_audio_x.shape, val_pic_x.shape, val_y.shape,
    #              audio_test.shape, pic_test.shape, labels_test_y.shape))
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=50, mode='max')

    model_history = model.fit(pic_train,
                              labels_train_y,
                              batch_size=batch_size,
                              epochs=num_epochs,
                              validation_data=(pic_val, labels_val_y),
                              callbacks=[CustomEarlyStopping(), es_callback])

    # Evaluate the validation
    val_loss, val_acc = model.evaluate(pic_val, labels_val_y)
    print("{} model val accuracy: {:5.2f}%".format(model_name, 100 * val_acc))
    print("{} model val loss: {:5.2f}".format(model_name, val_loss))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(pic_test, labels_test_y)
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


def create_picture_cnn_model(optimizer, pic_shape, output_dim):
    """
    Creates cnn model for video data
    :param optimizer: optimizer for a cnn
    :param pic_shape: shape of the pictures
    :param output_dim: dimension of output
    :return: compiled cnn model
    """

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

    pic_x = Dense(32, activation='relu')(pic_x)

    out = Dense(output_dim, activation='relu')(pic_x)

    model = Model(
        inputs=[pic_input],
        outputs=[out]
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model
