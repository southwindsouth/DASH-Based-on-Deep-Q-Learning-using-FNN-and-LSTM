# This code is based on the nice sample code from:
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.insert(0, '.') # for modules in the current directory
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from utils import create_dataset, apply_transform


# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataset = np.load('bandwidths.npy')
dataset = dataset.astype('float32')
dataset = dataset.reshape((len(dataset), 1)) # [n_samples, n_features] for scaler

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
joblib.dump(scaler, 'lstm_scaler.joblib')  # save for later use

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 5                # n_past_segments
look_forward = 2             # n_future_segments + 1(for current)
trainX, trainY = create_dataset(train, look_back, look_forward)
testX, testY = create_dataset(test, look_back, look_forward)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(look_forward))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2) # for debugging
model.save('lstm_model.h5')     # save for later use

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = apply_transform(trainPredict, scaler.inverse_transform)
trainY = apply_transform(trainY, scaler.inverse_transform)
testPredict = apply_transform(testPredict, scaler.inverse_transform)
testY = apply_transform(testY, scaler.inverse_transform)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
tmp = trainPredict[:,0].reshape((len(trainPredict), 1)) # only the first predictions
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = tmp

# # shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
tmp = testPredict[:,0].reshape((len(testPredict), 1)) # only the first predictions
testPredictPlot[len(trainPredict)+(look_back*2):-look_forward, :] = tmp

# plot baseline and predictions
plt.close('all')
plt.plot(scaler.inverse_transform(dataset), label='True')
plt.plot(trainPredictPlot, label='Predicted (Train)')
plt.plot(testPredictPlot, label='Predicted (Test)')
plt.xlabel('Segment Number')
plt.ylabel('Bandwidth [kbps]')
plt.legend(loc=1)
plt.show()
