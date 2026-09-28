import numpy as np


def fft_smooth(x, win_samples):
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    w = int(win_samples)
    if w < 1 or n == 0:
        return x.copy()
    w = min(w, n)

    window = np.zeros(n)
    lo = max((n - w + 1) // 2 - 1, 0)
    hi = min((n + w) // 2, n)
    window[lo:hi] = 1.0

    fx = np.fft.fft(x)
    fw = np.fft.fft(window)
    tmp = np.fft.ifft(fx * fw / w)
    return np.real(np.fft.ifft(np.fft.fft(tmp) * fw / w))
