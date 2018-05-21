import keras
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
from keras import metrics
# from keras.utils.np_utils import to_categorical
import pandas     as pd
import numpy      as np

# -----------------------------------------------------------
# read in the data from files
# -----------------------------------------------------------
pdtrain = pd.read_csv('train.dat')
pdtest = pd.read_csv('test.dat')

trainx = np.array(pdtrain[['x1', 'x2']])
trainy = np.array(pdtrain[['y1', 'y2']])
testx = np.array(pdtest[['x1', 'x2']])
testy = np.array(pdtest[['y1', 'y2']])
# trainy = to_categorical(trainy, num_classes=None)

# -----------------------------------------------------------
# build the network
# network - 2 -> 2 -> 2
# -----------------------------------------------------------
inputnodes = 2
hiddennodes = 2
outputnodes = 2
model = Sequential()
# Dense layer performs: output = activation(dot(input, kernel) + bias)
# Source: https://keras.io/layers/core/
model.add(Dense(hiddennodes, activation='sigmoid', input_dim=inputnodes))
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
