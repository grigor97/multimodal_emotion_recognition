import numpy as np
import os
from utils.utils import *
from utils.nn_utils import *

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

path_cfg = 'configs/config_paths.yml'
cfg = load_cfg(path_cfg)

logs_path = cfg['logs']['logs_path']


# loading datasets
train_x = np.load(cfg['data']['train_xp'])
train_y = np.load(cfg['data']['train_yp'])
test_x = np.load(cfg['data']['test_xp'])
test_y = np.load(cfg['data']['test_xp'])

# normalizing datasets
train_mean = train_x.mean(axis=0)
train_x -= train_mean
test_x -= train_mean
train_x/train_x.sum(axis=1).reshape((train_x.shape[0], 1))
test_x/test_x.sum(axis=1).reshape((test_x.shape[0], 1))

print("shape of train_x is {} and shape of train_y is {}".format(train_x.shape, train_y.shape))
print("shape of test_x is {} and shape of test_y is {}".format(test_x.shape, test_y.shape))

labels_train_y = to_categorical(train_y)
labels_test_y = to_categorical(test_y)

print("train and test shapes are {} {}".format(train_x.shape, test_x.shape))


def create_audio_cnn_model():
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same', input_shape=(train_x.shape[1], 1)))  # X_train.shape[1] = No. of Columns
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
    model.add(Dense(8))  # Target class number
    model.add(Activation('softmax'))
    # opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    # opt = keras.optimizers.Adam(lr=0.0001)
    opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


model = create_audio_cnn_model()
print("audio cnn model summary is \n".format(model.summary()))

checkpoint_path = logs_path + "training_audio_cnn/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model_history = model.fit(np.expand_dims(train_x, -1),
                          labels_train_y,
                          batch_size=16,
                          epochs=5,
                          validation_split=0.2,
                          callbacks=[cp_callback])

# Evaluate the model
loss, acc = model.evaluate(np.expand_dims(test_x, -1), labels_test_y)
print("audio cnn model accuracy: {:5.2f}%".format(100 * acc))
print("audio cnn model loss: {:5.2f}%".format(loss))

model.save(checkpoint_dir + '/model.h5')
nn_save_model_plots(model_history, checkpoint_dir)

with open(checkpoint_dir + '/cnn_audio_res.txt', 'w') as f:
    f.write("train accuracy and loss are ")
    f.write(str(acc) + ' ' + str(loss))
    f.write('\n')


