#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     generate bandwidths.py
# @authors  Tiantian Guo
#           Kyeong Soo (Joseph) Kim <Kyeongsoo.Kim@xjtlu.edu.cn>
# @date     2019-04-22
#
# @brief    Generate a series of channel bandwidths for using a Markov chain
#           for segments of a DASH video stream.
#
# @remarks  The original code is from Tiantian's master thesis project and
#           modified by Kyeong Soo (Joseph) Kim for EEE415 Labs.
#
#           The following transition matrix is used for the Markov chain:
#           [[0.4, 0.6, 0,   0,   0],
#            [0.2, 0.4, 0.4, 0,   0],
#            [0,   0.4, 0.4, 0.2, 0],
#            [0,   0,   0.5, 0.3, 0.2],
#            [0,   0,   0,   0.5, 0.5]]

### import modules
import numpy as np
import matplotlib.pyplot as plt


def generate_bandwidths(num_segments, bandwidth_levels):

    # initialize
    n_levels = len(bandwidth_levels)
    level = 0
    bandwidths = np.zeros((num_segments))
    bandwidths[0] = bandwidth_levels[level]
    A = np.array([[0.4, 0.6, 0,   0,   0],
                  [0.2, 0.4, 0.4, 0,   0],
                  [0,   0.4, 0.4, 0.2, 0],
                  [0,   0,   0.5, 0.3, 0.2],
                  [0,   0,   0,   0.5, 0.5]])

    # generate bandwidths using the Markov chain
    for i in range(1, num_segments):
        level = np.random.choice(n_levels, size=None, p=A[level]) # update bandwidth level
        bandwidths[i] = bandwidth_levels[level]

    return bandwidths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N",
        "--num_segments",
        help=
        "number of segments in a DASH video stream; default is 318 (for 'big buck bunny')",
        default=318,
        type=int)
    parser.add_argument(
        "-L",
        "--bandwidth_levels",
        help="comma-separated numbers for bandwidth levels to generate; default is '20,60,100,500,1000'",
        default='20,60,100,500,1000',
        type=str)
    args = parser.parse_args()
    num_segments = args.num_segments
    bandwidth_levels = np.sort(list(map(int, args.bandwidth_levels.split(',')))) # sorted bandwidth levels from a string into a list

    # generate bandwidths
    bandwidths = generate_bandwidths(num_segments, bandwidth_levels)
            
    # save bandwidths as numpy format (.npy)
    # - you can load it back as follows:
    # - bandwidths = np.load('bandwidths.npy')
    np.save('bandwidths.npy', bandwidths)

    # plot bandwidths
    plt.figure()
    plt.plot(bandwidths)
    plt.xlabel('Segment Index')
    plt.ylabel('Bandwidth [kbps]')
    plt.show()
