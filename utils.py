#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     utils.py
# @author   Kyeong Soo (Joseph) Kim <Kyeongsoo.Kim@xjtlu.edu.cn>
# @date     2019-04-24
#
# @brief    Utility functions for Simulate DASH video streaming.
#
# @remarks


### import modules
import numpy as np
from skimage.util import view_as_windows


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=2, look_forward=1):
    dataset = dataset.flatten()
    dataX = view_as_windows(dataset, look_back, step=1)[:-look_forward] # sliding window (except the last one)
    dataY = view_as_windows(dataset, look_forward, step=1)[look_back:]
    return dataX, dataY


# apply scaler transform/inverse transform with shape pre/postprocessing
def apply_transform(data, transform):
    tmp = data.flatten()
    tmp = np.reshape(tmp, (len(tmp), 1))
    tmp = transform(tmp)
    return np.reshape(tmp, data.shape)


# # apply scaler inverse transform with shape pre and postprocessing
# def inverse_transform(data, scaler):
#     tmp = data.flatten()
#     tmp = np.reshape(tmp, (len(tmp), 1))
#     tmp = scaler.inverse_transform(tmp)
#     return np.reshape(tmp, data.shape)
