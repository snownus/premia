import keras
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
from keras.datasets   import mnist
import pandas     as pd
import numpy      as np

# -----------------------------------------------------------
# read in the data from files
# -----------------------------------------------------------
# download the data file from: https://s3.amazonaws.com/img-datasets/mnist.npz
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
# -----------------------------------------------------------
# build the network
# -----------------------------------------------------------

# -----------------------------------------------------------
# loss functions and optimizer
# -----------------------------------------------------------

# -----------------------------------------------------------
# do the training
# -----------------------------------------------------------
# number of epochs - one epoch is one run through the data set
