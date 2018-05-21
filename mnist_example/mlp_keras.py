import keras
from keras import metrics
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
from keras.datasets   import mnist
import numpy      as np

num_classes = 10
# -----------------------------------------------------------
# read in the data from files
# -----------------------------------------------------------
# download the data file from: https://s3.amazonaws.com/img-datasets/mnist.npz
(x_train0, y_train), (x_test0, y_test) = mnist.load_data(path="mnist.npz")

x_train = np.reshape(x_train0,(60000,28*28))
x_test  = np.reshape(x_test0,(10000,28*28))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# -----------------------------------------------------------
# build the network
# -----------------------------------------------------------
inputnodes   = 28*28
hiddennodes1 = 20*20
hiddennodes2 = 15*15
hiddennodes3 =  7*7
hiddennodes4 = num_classes
outputnodes  = num_classes
model = Sequential()
# Dense layer performs: output = activation(dot(input, kernel) + bias)
# Source: https://keras.io/layers/core/
model.add(Dense(hiddennodes1, activation='sigmoid', input_dim=inputnodes))
model.add(Dense(hiddennodes2, activation='sigmoid'))
model.add(Dense(hiddennodes3, activation='sigmoid'))
model.add(Dense(hiddennodes4, activation='sigmoid'))
model.add(Dense(outputnodes,  activation='softmax')) 
# do we need to add both output and softmax layers or just softmax is enough?

# -----------------------------------------------------------
# loss functions
# -----------------------------------------------------------
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=[metrics.categorical_accuracy])

# -----------------------------------------------------------
# do the training
# -----------------------------------------------------------
# number of epochs - one epoch is one run through the data set
Nepochs=32

# this line optimize and train the neural network
model.fit(x_train, y_train, epochs=Nepochs, batch_size=128)

# now evaluate the test loss and accuarcy
test_result = model.evaluate(x_test, y_test, batch_size=128)
print("")
print("test loss ")
print(test_result[0])
print("test acc ")
print(testloss[1])

# -----------------------------------------------------------
# obtain predictions for individual data
# -----------------------------------------------------------
test_predictions = model.predict(x_test)
