#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     base.py
# @authors  Tiantian Guo
#           Kyeong Soo (Joseph) Kim <Kyeongsoo.Kim@xjtlu.edu.cn>
# @date     2019-04-22
#
# @brief    Simulate DASH video streaming.
#
# @remarks  It is part of Tiantian's master thesis project and
#           modified by Kyeong Soo (Joseph) Kim for EEE415 Labs.

### import modules
import sys
sys.path.insert(0, '.')         # for modules in the current directory
import matplotlib.pyplot as plt
import numpy as np
from itertools import *
from bw_predictor import bw_predictor  # to be implemented by user


def simulate_dash(br, bw, bl, sd, nps, nfs, ws):
    # set parameters
    nq = br.shape[1]            # number of quality levels
    ns = len(br)                # number of segments
    phi = np.array(list(product(range(1,nq+1), repeat=nfs+1)))  # quality patterns over current and future segments
    w1, w2, w3 = ws             # weights of QoE

    # create a bw_predictor object for bandwidth prediction
    bp = bw_predictor(
                 model_fname='lstm_model.h5',
                 scaler_fname='lstm_scaler.joblib',
                 n_past_segments=nps,
                 n_future_segments=nfs)

    t = np.zeros(nfs+2)
    ts = np.zeros(nfs+1)
    qoe = np.zeros(len(phi))
    Q = np.zeros(ns, dtype=int)  # to be used as index
    T = np.zeros(ns+1)
    Ts = np.zeros(ns)

    # buffer update during the intial period w/o adaptation
    Q[:nps] = 1
    T[0]=5
    for i in range(nps):
        T[i+1] = max(T[i]-br[i][Q[i]-1]*sd/bw[i]+sd, 0)
        Ts[i] = max(br[i][Q[i]-1]*sd/bw[i]-T[i], 0)
    
    # main simulation loop for adaptation
    idxs = np.arange(nps, ns-nfs)
    for i in idxs:
        pbws = bp.predict(bw[i-nps:i].reshape((1, nps)))  # prediced bandwdiths
        for j in range(len(phi)): # optimization over quality patterns
            q = phi[j][:]
            e = np.mean(q)
            v = abs(q[1:]-q[:-1]).mean()

            # buffer updating for the currnet and future segments
            # N.B.: be careful about the indexes.
            t[0] = T[i]
            for k in range(nfs+1):
                t[k+1] = max(t[k]-br[i+k][q[k]-1]*sd/pbws[k]+sd,0)
                ts[k] = max(br[i+k][q[k]-1]*sd/pbws[k]-t[k], 0)

            tst = np.sum(ts)
            ttt = sd*(nfs+1)+tst
            ps = tst/ttt
            delta = (t[nfs+1]-t[0])/(nfs+1)
            qoe[j] = e-w1*v-w2*ps+w3*delta
        qindex = np.argmax(qoe, axis=0)
        Q[i] = phi[qindex][0]

        # update buffer level for the current segment
        T[i+1] = max(T[i]-br[i][Q[i]-1]*sd/bw[i]+sd, 0)
        Ts[i] = max(br[i][Q[i]-1]*sd/bw[i]-T[i], 0)

    # limit performance metrics to the adaptation period    
    Q = Q[idxs]
    Ts = Ts[idxs]
    T = T[idxs+1]
    
    E = Q.mean()                 # average requested media quality
    V = abs(Q[1:]-Q[:-1]).mean()  # quality switching frequency
    TSS = Ts.sum()
    PSS = TSS/(ns*sd + TSS)     # ratio of starvation event in time domain
    QoE = E - w1*V - w2*PSS
    return QoE, Q, T


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-B",
        "--dash_bitrates",
        help=
        "name of a DASH bitrates file name; default is 'bigbuckbunny.npy' (for 'big buck bunny')",
        default='bigbuckbunny.npy',
        type=str)
    parser.add_argument(
        "-C",
        "--channel_bandwidths",
        help=
        "name of a channel bandwidths file name; default is 'bandwidths.npy'",
        default='bandwidths.npy',
        type=str)
    parser.add_argument(
        "-L",
        "--bandwidth_levels",
        help="comma-separated numbers for bandwidth levels to generate; default is '20,60,100,500,1000'",
        default='20,60,100,500,1000',
        type=str)
    parser.add_argument(
        "-D",
        "--segment_duration",
        help=
        "duration of each segment in a DASH video stream; default is 2 [s]",
        default=2,
        type=float)
    parser.add_argument(
        "-P",
        "--n_past_segments",
        help=
        "number of past segments used for bandwidth prediction; default is 5",
        default=5,
        type=int)
    parser.add_argument(
        "-F",
        "--n_future_segments",
        help=
        "number of future segments to predict bandwidths for; default is 1",
        default=1,
        type=int)
    parser.add_argument(
        "-W",
        "--qoe_weights",
        help="comma-separated numbers for QoE weights (i.e., w1, w2, w3(->lambda)); default is '0.3333,2,0.9'",
        default='0.3333,2,0.9',
        type=str)
    args = parser.parse_args()
    dash_bitrates = args.dash_bitrates
    channel_bandwidths = args.channel_bandwidths
    bl = np.sort(list(map(int, args.bandwidth_levels.split(','))))  # sorted bandwidth levels from a string into a list
    sd = args.segment_duration
    nps = args.n_past_segments
    nfs = args.n_future_segments
    qw = list(map(float, args.qoe_weights.split(',')))  # w3 -> lambda; labmda is a Python keyword

    # read data
    br = np.load(dash_bitrates)      # dash bitrates
    bw = np.load(channel_bandwidths) # channel bandwdiths

    # simulate DASH video streaming
    QoE, Q, T = simulate_dash(br, bw, bl, sd, nps, nfs, qw)

    # print QoE
    print("QoE: {0:.4e}".format(QoE))

    # plot the figures
    ns = len(br)                # number of segments
    x = np.arange(ns)
    bit = np.empty(ns)
    Tplot = np.empty(ns)        # T for plotting
    bit[:] = np.nan             # ignore values outside the adaptation period
    Tplot[:] = np.nan           # "
    for i in range(nps, ns-nfs):
        bit[i] = br[i][Q[i-nps]-1]
        Tplot[i] = T[i-nps]

    plt.close('all')
    fig, axs = plt.subplots(2, 1)

    # 1st subplot
    axs[0].plot(x, bw, color='blue', label='Bandwidth')
    axs[0].plot(x, bit, color='red', label='Video Bitrate')
    axs[0].set_xlabel('Segment Index')
    axs[0].set_ylabel('Bitrate [kbs]')
    axs[0].legend()

    # 2nd subplot
    r_ax = axs[1].twinx()
    p1 = axs[1].plot(x, bw, color='blue', label='Bandwidth')
    p2 = r_ax.plot(x, Tplot, color='red', label='Buffer Reservation')
    axs[1].set_xlabel('Segment Index')
    axs[1].set_ylabel('Bitrate [kbps]')
    r_ax.set_ylabel('Buffer Reservation [kb]')
    ps = p1+p2
    labels = [p.get_label() for p in ps]  # combine legends
    axs[1].legend(ps, labels, loc='upper left')

    plt.tight_layout()
    plt.show()
