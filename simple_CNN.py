#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 05:02:03 2018

@author: farhan
"""

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Input images have same width and height
ImgSizeX=28   
ImgSizeY=28
epoch = 30
batchsize=128

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][channels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, ImgSizeX, ImgSizeY).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, ImgSizeX, ImgSizeY).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
# create model
    model = Sequential()
    model.add(Convolution2D(8, (3, 3), border_mode='valid', 
                            input_shape=(1, ImgSizeX, ImgSizeY),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    nb_epoch=epoch, batch_size=batchsize, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

# Print final test loss and accuracy

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Plot train vs test accuracy and loss (In IDE)

print(history.history.keys())
# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Plot model summary
model.summary()