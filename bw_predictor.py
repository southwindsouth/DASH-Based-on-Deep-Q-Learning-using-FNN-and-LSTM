#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     bw_predictor.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2019-04-22
#
# @brief    A class for channel bandwidth prediction based on past observations.
#
# @remarks  The current implementation is based on the trained model from
#           'predict_bw_lstm2.py'.

### import modules
import numpy as np
import sys
from keras.models import load_model
from sklearn.externals import joblib
from utils import create_dataset, apply_transform


class bw_predictor(object):
    """
    Predict future bandwidths up to n_future_segments based on past observations.
    N.B.: if n_future_segments is greater than one, you need to modify
    'predict_bw_lstm2.py' accordingly.
    """
    def __init__(self,
                 model_fname='lstm_model.h5',
                 scaler_fname='lstm_scaler.joblib',
                 n_past_segments=1,
                 n_future_segments=1):
        
        # load saved model and scaler from 'predict_bw_lstm2.py'
        self.model = load_model(model_fname)
        self.scaler = joblib.load(scaler_fname)
        self.nps = n_past_segments
        self.nfs = n_future_segments
        
    def predict(self, history):
        # scale input data
        history = apply_transform(history, self.scaler.transform)
        # # history = history.reshape((self.nps, 1)) # [n_samples, n_features] for scaler
        # history = history.reshape((1, self.nps)) # [n_samples, n_features] for scaler

        # predict
        history = np.reshape(history, (history.shape[0], 1, self.nps)) # [samples, time steps, features] for model
        bws = self.model.predict(history)         

        # inverse scale predicted data
        bws = apply_transform(bws, self.scaler.inverse_transform)

        return bws.flatten()


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from skimage.util import view_as_windows
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-P",
        "--n_past_segments",
        help=
        "number of past segments used for bandwidth prediction; default is 1",
        default=2,
        type=int)
    parser.add_argument(
        "-F",
        "--n_future_segments",
        help=
        "number of future segments to predict bandwidths for; default is 2",
        default=2,
        type=int)
    args = parser.parse_args()
    n_past_segments = args.n_past_segments
    n_future_segments = args.n_future_segments

    # create bw_predictor object
    bp = bw_predictor(n_past_segments=n_past_segments, n_future_segments=n_future_segments)

    bws = np.load('bandwidths.npy')

    # initialize variables
    look_back = n_past_segments
    look_forward = n_future_segments + 1
    history, _ = create_dataset(bws, look_back=look_back, look_forward=look_forward)

    # reshape input to be [samples, time steps, features]
    history = np.reshape(history, (history.shape[0], 1, history.shape[1]))
    
    predicted = []
    for i in range(len(history)):
        predicted.append(bp.predict(history[i]))

    predicted = np.array(predicted)[:, 0]  # first predictions only
    idx = np.arange(len(history)) + look_back
    plt.close('all')
    plt.plot(bws, label='True')
    plt.plot(idx, predicted, label='Predicted')
    plt.xlabel('Segment Number')
    plt.ylabel('Bandwidth [kbps]')
    plt.legend(loc=1)
    plt.show()
