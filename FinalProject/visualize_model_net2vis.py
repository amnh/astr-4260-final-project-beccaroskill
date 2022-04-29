'''
Visualize model using https://viscom.net2vis.uni-ulm.de/
'''

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Conv1D, MaxPooling1D,\
                                    Activation, Dropout, Dense, Flatten, \
                                    Input, concatenate

def get_cnn(width, height, pool_size=5, filters=(16, 32, 64), batch_norm=False, regress=False):
  inputShape = (height, width)
  chanDim = -1
  inputs = Input(shape=inputShape)
  for (i, f) in enumerate(filters):
    if i == 0:
      x = inputs
    x = Conv1D(filters=f, 
               kernel_size=5,
               padding="same",
               activation=tf.nn.relu)(x)
    x = Conv1D(filters=f, 
            kernel_size=5,
            padding="same",
            activation=tf.nn.relu)(x)
    if batch_norm:
      x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling1D(pool_size=pool_size, 
                     strides=2)(x)
  x = Flatten()(x)
  if regress:
    x = Dense(1, activation="linear")(x)
  model = Model(inputs, x)
  return model

def get_model():
  batch_norm = True
  cnn_global = get_cnn(1, 2001, 5, (16, 32, 64, 128, 256), batch_norm=batch_norm)
  cnn_local = get_cnn(1, 201, 7, (16, 32), batch_norm=batch_norm)
  combined_input = concatenate([cnn_global.output, cnn_local.output])
  N_CLASSES = 3

  droupout_rate = 0.03
  x = Dense(512, activation="relu")(combined_input)
  x = Dropout(droupout_rate)(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(droupout_rate)(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(droupout_rate)(x)
  x = Dense(512, activation="relu")(x)
  x = Dense(N_CLASSES, activation="softmax")(x)

  model = Model(inputs=[cnn_global.input, cnn_local.input], outputs=x)

  return model