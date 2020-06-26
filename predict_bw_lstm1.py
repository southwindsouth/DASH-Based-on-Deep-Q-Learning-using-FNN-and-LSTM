# This code is based on the nice sample code from:
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from skimage.util import view_as_windows

# define dataset
bws = np.load('bandwidths.npy')
X = view_as_windows(bws, 3, step=1)[:-1] # 3-sample sliding window over bws (except the last one, i.e., '[:-1]')
y = bws[3:]

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
for i in range(10):
    x_input = X[i]
    x_input = x_input.reshape((1, 3, 1))
    yhat = model.predict(x_input, verbose=0)
    print('{0} -> {1:.2e} (true value: {2:d})'.format( ','.join([str(int(i)) for i in x_input.flatten()]), yhat.flatten()[0], int(y[i])))
