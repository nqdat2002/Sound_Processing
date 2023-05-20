import numpy as np
import matplotlib.pyplot as plt

def rectangle(n = 1001):
    w = np.ones(n)
    return np.array(w)

def triangle(n = 1001):
    w = np.zeros(n)
    for i in range(0, int((n - 1) / 2) + 1):
        w[i] = (2 * i) / (n - 1)
    for i in range(int((n - 1) / 2) + 1, n):
        w[i] = 2 - ((2 * i) / (n - 1))
    return np.array(w)

def hanning(n = 1001):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    return np.array(w)

def hamming(n = 1001):
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    return np.array(w)

def blackman(n = 1001):
    w = 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1)) +  0.08 * np.cos(4 * np.pi * np.arange(n) / (n - 1))
    return np.array(w)

def plotWindow(window, n):
    n = np.arange(0, n)
    plt.plot(n, window)
    plt.subplots_adjust(hspace=0.35)
    plt.grid(True)
    plt.show()

def main():
    plotWindow(rectangle(10), 10)

if __name__ == '__main__':
    main()