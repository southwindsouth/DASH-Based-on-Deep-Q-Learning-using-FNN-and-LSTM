import matplotlib.pyplot as plt
import numpy as np


def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


if __name__ == "__main__":
    bws = np.load('bandwidths.npy')
    b_max = bws.max()
    b_min = bws.min()

    plt.close('all')
    plt.figure()
    plt.plot(bws, label='original')
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    for w in windows[1:]:
        s_bws = smooth(bws, window=w)
        s_bws = np.clip(s_bws, b_min, b_max)[:len(bws)] # same lenth as bws
        np.save(w+'_bws.npy', s_bws)
        plt.plot(s_bws, label=w)
    plt.xlabel('Segment Index')
    plt.ylabel('Bandwidth [kbps]')
    plt.legend(loc=1)
    plt.show()
