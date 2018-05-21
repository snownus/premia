import keras
from keras import metrics
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

import pandas     as pd
import numpy      as np

# -----------------------------------------------------------
# read in the data from files
# -----------------------------------------------------------
pdtrainx = pd.read_csv('x_train.csv')
pdtestx = pd.read_csv('x_test.csv')
pdtrainy = pd.read_csv('y_train.csv')
pdtesty = pd.read_csv('y_test.csv')

trainx = np.array(pdtrainx[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14']])
trainy_onehot = np.array(pdtrainy[['y']])
testx = np.array(pdtestx[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14']])
testy_onehot = np.array(pdtesty[['y']])
trainy = to_categorical(trainy_onehot, num_classes=4)
testy = to_categorical(testy_onehot, num_classes=4)

# -----------------------------------------------------------
# build the network
# network - 14 -> 100 -> 40 -> 4
# -----------------------------------------------------------
inputnodes = 14
hiddennodes1 = 100
hiddennodes2 = 40
outputnodes = 4
model = Sequential()
# Dense layer performs: output = activation(dot(input, kernel) + bias)
# Source: https://keras.io/layers/core/
model.add(Dense(hiddennodes1, activation='sigmoid', input_dim=inputnodes))
model.add(Dense(hiddennodes2, activation='sigmoid'))
model.add(Dense(outputnodes, activation='sigmoid'))

# -----------------------------------------------------------
# loss functions
# options for loss are:
# 'categorical_crossentropy(y_true, y_pred)'
# 'mean_squared_error(y_true, y_pred)'
# many more at: https://keras.io/losses/
# options for optimizer are:
# keras.optimizers.SGD(lr=0.01, clipnorm=1.)
# keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# many more at: https://keras.io/optimizers/
# -----------------------------------------------------------
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=[metrics.categorical_accuracy])

# -----------------------------------------------------------
# do the training
# -----------------------------------------------------------
# number of epochs - one epoch is one run through the data set
Nepochs=100

# this line optimize and train the neural network
model.fit(trainx, trainy, epochs=Nepochs, batch_size=128)

# now evaluate the test loss
testloss = model.evaluate(testx, testy, batch_size=128)[0]
print(testloss)

# now evaluate the test accuracy
testacc = model.evaluate(testx, testy, batch_size=128)[1]
print(testacc)

# -----------------------------------------------------------
# obtain predictions for individual data
# -----------------------------------------------------------
test_predictions = model.predict(testx, batch_size=128)

print(test_predictions)
print(testy)
