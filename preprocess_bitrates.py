#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     preprocess_bitrates.py
# @author   Kyeong Soo (Joseph) Kim <Kyeongsoo.Kim@xjtlu.edu.cn>
# @date     2019-04-22
#
# @brief    Preprocess the bitrates of DASH video streams.
#
# @remarks  This is for the data from Tiantian's master thesis project.

### import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# define constants
videos =  ['bear', 'bigbuckbunny', 'test']


# read from Excel files and write back to pickle and numpy files.
for video in videos:
        df = pd.read_excel(video+'.xls', sheet_name='sheet1', header=None)
        df.to_pickle(video+'.pkl')  # as dataframe
        np.save(video+'.npy', df.values)  # as numpy array
        
# read from pickle files and display
n_videos = len(videos)
fig, axs = plt.subplots(n_videos, 1)
for i in range(n_videos):
        df = pd.read_pickle(videos[i]+'.pkl')
        print(df.head())
        print("{0}: {1} rows and {2} columns".format(videos[i], df.shape[0], df.shape[1]))
        axs[i].plot(df)
        axs[i].set_title(videos[i])
plt.tight_layout()
plt.show()
