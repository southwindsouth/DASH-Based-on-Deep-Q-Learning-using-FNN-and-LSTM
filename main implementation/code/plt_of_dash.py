import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import torch
from dataclasses import dataclass
import time



fnn_with_target = np.load('./data/dash_FNN_with_target_300.npz')
fnn_no_target= np.load('./data/dash_FNN_without_target_300OK.npz')
lstm_no_target = np.load('./data/dash_LSTM_without_target_300ok.npz')
lstm_with_target = np.load('./data/dash_LSTM_with_target_300ok.npz')


fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(fnn_with_target['data_rewards'])
axs[0].plot(fnn_no_target['data_rewards'])
axs[0].set_ylabel("Reward")
axs[0].vlines(len(fnn_with_target['data_sqs']), *axs[0].get_ylim(), colors='red', linestyles='dotted')
axs[1].plot(fnn_with_target['data_sqs'])
axs[1].plot(fnn_no_target['data_sqs'])
axs[1].set_ylabel("Video Quality")
axs[1].set_xlabel("Video Episode")
axs[1].vlines(len(fnn_with_target['data_sqs']), *axs[1].get_ylim(), colors='red', linestyles='dotted')
plt.legend(['fnn_with_target','fnn_no_target'], loc='upper right')
plt.show()
plt.close('all')

fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(fnn_no_target['data_rewards'])
axs[0].plot(lstm_no_target['data_rewards'])
axs[0].set_ylabel("Reward")
axs[0].vlines(len(fnn_with_target['data_sqs']), *axs[0].get_ylim(), colors='red', linestyles='dotted')
axs[1].plot(fnn_no_target['data_sqs'])
axs[1].plot(lstm_no_target['data_sqs'])
axs[1].set_ylabel("Video Quality")
axs[1].set_xlabel("Video Episode")
axs[1].vlines(len(fnn_with_target['data_sqs']), *axs[1].get_ylim(), colors='red', linestyles='dotted')
plt.legend(['fnn_no_target','lstm_no_target'], loc='upper right')
plt.show()
plt.close('all')

fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(lstm_with_target['data_rewards'])
axs[0].plot(lstm_no_target['data_rewards'])
axs[0].set_ylabel("Reward")
axs[0].vlines(len(fnn_with_target['data_sqs']), *axs[0].get_ylim(), colors='red', linestyles='dotted')
axs[1].plot(lstm_with_target['data_sqs'])
axs[1].plot(lstm_no_target['data_sqs'])
axs[1].set_ylabel("Video Quality")
axs[1].set_xlabel("Video Episode")
axs[1].vlines(len(fnn_with_target['data_sqs']), *axs[1].get_ylim(), colors='red', linestyles='dotted')
plt.legend(['lstm_with_target','lstm_no_target'], loc='upper right')
plt.show()
plt.close('all')
