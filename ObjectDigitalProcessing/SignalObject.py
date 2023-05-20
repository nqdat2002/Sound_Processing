import math
import io, os, sys, time
import array as arr
import numpy as np
import matplotlib.pyplot as plt
import cmath
import pandas as pd
import soundfile as sf
import librosa as lb
import wave
from sympy import *

class Signal:
    def __init__(self, amplitude, startPos, fs, info, length):
        self.__amplitude = amplitude
        self.__startPos = startPos
        self.__fs = fs
        self.__info = info
        self.__length = length

    def getFs(self):
        return  self.__fs

    def setFs(self, fs):
        self.__fs = fs

    def getStartPos(self):
        return self.__startPos

    def setStartPos(self, startPos):
        self.__startPos = startPos

    def getAmplitude(self):
        return self.__amplitude

    def setAmplitude(self, amplitude):
        self.__amplitude = amplitude

    def getLength(self):
        return self.__length

    def setLength(self, length):
        self.__length = length

    def getInfo(self):
        return self.__info

    def setInfor(self, info):
        self.__info = info

    def getRandSignal(self, length, amplitude, startPos = 0):
        self.__length = length
        self.__amplitude = amplitude
        self.__startPos = startPos

    def showInfo(self):
        print('Info: ', self.getInfo(), 'Fs: ', self.getFs(), 'Amplitude: ' ,self.getAmplitude(), 'StartPos: ', self.getStartPos())

    def plotSignal(self):
        indices = np.linspace(0, self.__length - 1, num=self.__length)
        indices += self.getStartPos()
        # print(self.__amplitude)
        # print(indices)
        plt.ylabel('Signal Value')
        plt.xlabel('Time (s)')
        plt.stem(indices, self.getAmplitude())
        # plt.plot(indices, self.__amplitude)
        plt.grid(True)
        plt.show()

    def stemSignal(self):
        n_samples = self.getLength()
        sample_freq = self.getFs()
        t_audio = n_samples / sample_freq
        times = np.linspace(0, t_audio, num=n_samples)
        plt.figure(figsize=(15, 5))
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.stem(times, self.getAmplitude())
        plt.xlim(0, t_audio)
        plt.grid(True)
        plt.show()

    def eraseZero(self):
        # erase 0 if which in begin and end of the amplitude
        x = self.getAmplitude()
        y = []

    def addZeroFront(self):
        # add 0 to begin of the amplitude if it has pos > 0
        x = self.getAmplitude()
        pos = self.getStartPos()
        if pos > 0:
            for _ in range(self.getStartPos() - 0): x = np.insert(x, 0, 0)
        self.setAmplitude(x)
        self.setLength(len(x))
        self.setStartPos(min(pos, 0))

    def sameLength(self, other):
        # add 0 to begin and end of the amplitude which has different length
        am1, am2, pos1, pos2 = self.getAmplitude(), other.getAmplitude(), self.getStartPos(), other.getStartPos()
        if len(am1) == len(am2): return
        n = max(len(am1), len(am2))
        if len(am1) < len(am2):
            if pos1 > pos2:
                for _ in range(n - len(am1)): am1 = np.insert(am1, 0, 0)
                self.__startPos = pos2
            elif pos1 < pos2:
                for _ in range(n - len(am1)): am1 = np.insert(am1, len(am1), 0)
        elif len(am1) > len(am2):
            if pos1 < pos2:
                for _ in range(n - len(am2)): am2 = np.insert(am2, 0, 0)
                other.__startPos = pos1
            elif pos1 > pos2:
                for _ in range(n - len(am2)): am2 = np.insert(am2, len(am2), 0)
        self.__amplitude, other.__amplitude = am1, am2

    def mulAlpha(self, n):
        a = []
        for i in self.__amplitude:
            a.append(i * n)
        self.setAmplitude(np.array(a))

    def add(self, other):
        pos1 = self.getStartPos()
        pos2 = other.getStartPos()
        pos = min(pos1, pos2)
        self.sameLength(other)
        if pos1 > pos2:
            left_over = pos1 - pos2
            for _ in range(left_over):
                self.__amplitude = np.insert(self.__amplitude, 0, 0)
                other.__amplitude = np.insert(other.__amplitude, len(other.__amplitude), 0)
        elif pos1 < pos2:
            left_over = pos2 - pos1
            for _ in range(left_over):
                other.__amplitude = np.insert(other.__amplitude, 0, 0)
                self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
        sum = self.__amplitude + other.__amplitude
        return Signal(sum, pos, self.getFs(), self.getInfo(), len(sum))

    def sub(self, other):
        pos1 = self.getStartPos()
        pos2 = other.getStartPos()
        pos = min(pos1, pos2)
        self.sameLength(other)
        if pos1 > pos2:
            left_over = pos1 - pos2
            for _ in range(left_over):
                self.__amplitude = np.insert(self.__amplitude, 0, 0)
                other.__amplitude = np.insert(other.__amplitude, len(other.__amplitude), 0)
        elif pos1 < pos2:
            left_over = pos2 - pos1
            for _ in range(left_over):
                other.__amplitude = np.insert(other.__amplitude, 0, 0)
                self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
        sum = self.__amplitude - other.__amplitude
        return Signal(sum, pos, self.getFs(), self.getInfo(), len(sum))

    def mul(self, other):
        pos1 = self.getStartPos()
        pos2 = other.getStartPos()
        pos = min(pos1, pos2)
        self.sameLength(other)
        if pos1 > pos2:
            left_over = pos1 - pos2
            for _ in range(left_over):
                self.__amplitude = np.insert(self.__amplitude, 0, 0)
                other.__amplitude = np.insert(other.__amplitude, len(other.__amplitude), 0)
        elif pos1 < pos2:
            left_over = pos2 - pos1
            for _ in range(left_over):
                other.__amplitude = np.insert(other.__amplitude, 0, 0)
                self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
        sum = self.__amplitude * other.__amplitude
        return Signal(sum, pos, self.getFs(), self.getInfo(), len(sum))

    def inverse(self):
        y = np.array([0])
        self.addZeroFront()
        pos = self.getStartPos()
        x = self.getAmplitude()
        if pos < 0:
            a = x[:abs(pos) + 1]
            # print(a)
            b = x[abs(pos) + 1:]
            # print(b)
            y = np.concatenate((b[::-1], a[::-1]))
            # self.__startPos = -len(b) - 1
            self.setStartPos(-len(b))
        else:
            a = x[pos:]
            # print(a)
            y = a[::-1]
            self.setStartPos(-len(a) + 1)
        self.setAmplitude(np.array(y))
        # print(res)
        
    def timeShifting(self, n0 = 0):
        x = self.getAmplitude()
        if n0 < 0:
            y = np.concatenate([np.zeros(-n0), x[:len(x)]])
            return Signal(y, self.getStartPos(), self.getFs(), self.getInfo(), len(y))
        if n0 > 0:
            y = np.concatenate([x, np.zeros(n0)])
            return Signal(y, self.getStartPos() - n0, self.getFs(), self.getInfo(), len(y))

    def delay(self, k):
        self.__startPos += k
        print(self.__startPos)
        ampli = self.__amplitude
        if self.__startPos > 0:
            for _ in range(self.__startPos): self.__amplitude = np.insert(self.__amplitude, 0, 0)
        elif self.__startPos < 0:
            for _ in range(abs(self.__startPos) - len(self.__amplitude)): self.__amplitude = np.insert(self.__amplitude,
                                                                                                       len(self.__amplitude),
                                                                                                       0)
        # self.__amplitude = ampli
        self.getRandSignal(len(self.__amplitude), self.__amplitude, min(0, self.__startPos))
        print(len(self.__amplitude))
        print(self.__amplitude)
        self.plotSignal()

    def early(self, k):
        self.__startPos -= k
        print(self.__startPos)
        ampli = self.__amplitude
        if abs(self.__startPos) > len(self.__amplitude):
            for _ in range(abs(self.__startPos) - len(self.__amplitude)): self.__amplitude = np.insert(self.__amplitude,
                                                                                                       len(self.__amplitude),
                                                                                                       0)
        # self.__amplitude = ampli
        elif self.__startPos > 0:
            for _ in range(self.__startPos): self.__amplitude = np.insert(self.__amplitude, 0, 0)
        self.getRandSignal(len(self.__amplitude), self.__amplitude, min(0, self.__startPos))
        print(len(self.__amplitude))
        print(self.__amplitude)
        self.plotSignal()

    def isCausalSignal(self):
        y = self.getAmplitude()
        pos = self.getStartPos()
        for i in range(0, abs(pos)):
            if y[i] != 0:
                return False
        return True

    def isAntiCausalSignal(self):
        y = self.getAmplitude()
        pos = self.getStartPos()
        for i in range(abs(pos), len(y)):
            if y[i] != 0:
                return False
        return True

    def isNonCausalSignal(self):
        return not self.isCausalSignal()

    def convolve(self, other):
        x = self.getAmplitude()
        h = other.getAmplitude()
        m, n = len(x), len(h)
        y = [0] * (m + n - 1)
        for i in range(m + n - 1):
            for j in range(max(0, i - m + 1), min(i + 1, n)):
                y[i] += x[i - j] * h[j]

        pos1 = self.getStartPos()
        pos2 = other.getStartPos()
        if pos1 * pos2 > 0:
            return Signal(np.array(y), pos1 + pos2, self.getFs(), self.getInfo(), len(y))
        return Signal(np.array(y), min(pos1, pos2), self.getFs(), self.getInfo(), len(y))

    def convolveUsingDefineMethod(self, other):
        x = self.getAmplitude()
        h = other.getAmplitude()
        y = np.convolve(x, h, 'full')
        # the method convolve has 3: x, h, and mode: ‘full’:
        # By default, mode is ‘full’. This returns the convolution at each point of overlap,
        # with an output shape of (N+M-1,). At the end-points of the convolution, the signals
        # do not overlap completely, and boundary effects may be seen.
        # ‘same’:
        # Mode ‘same’ returns output of length max(M, N). Boundary effects are still visible.
        # ‘valid’:
        # Mode ‘valid’ returns output of length max(M, N) - min(M, N) + 1. The convolution product is only given for points where the signals overlap completely. Values outside the signal boundary have no effect.
        pos1 = self.getStartPos()
        pos2 = other.getStartPos()
        if pos1 * pos2 > 0:
            return Signal(np.array(y), pos1 + pos2, self.getFs(), self.getInfo(), len(y))
        return Signal(np.array(y), min(pos1, pos2), self.getFs(), self.getInfo(), len(y))

    def crossCorrelation(self, other):
        other.plotSignal()
        # self.inverse()
        other.inverse()
        x = self.getAmplitude()
        h = other.getAmplitude()
        self.plotSignal()
        other.plotSignal()
        print(x)
        print(h)
        m, n = len(x), len(h)
        y = [0] * (m + n - 1)
        for i in range(len(y)):
            for j in range(max(0, i - m + 1), min(i + 1, n)):
                y[i] += x[i - j] * h[j]

        # for l in range(len(y)):
        #     for i in range(len(h)):
        #         y[l] += x[i + l] * h[i]
        #     # y[l] = sum([x[i + l] * h[i] for i in range(len(h))])

        pos1 = self.getStartPos()
        pos2 = other.getStartPos()
        if pos1 * pos2 > 0:
            return Signal(np.array(y), pos1 + pos2, self.getFs(), self.getInfo(), len(y))
        return Signal(np.array(y), min(pos1, pos2), self.getFs(), self.getInfo(), len(y))
        pass

    def crossCorrelationUsingDefineMethod(self, other):
        other.addZeroFront()
        x = self.getAmplitude()
        h = other.getAmplitude()
        print(x)
        print(h)
        res = np.correlate(x, h, 'full')
        other.inverse()
        pos1 = self.getStartPos()
        pos2 = other.getStartPos()
        if pos1 * pos2 > 0:
            return Signal(np.array(res), pos1 + pos2, self.getFs(), self.getInfo(), len(res))
        return Signal(np.array(res), min(pos1, pos2), self.getFs(), self.getInfo(), len(res))

    def dft(self):
        x = self.getAmplitude()
        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        e = np.exp(-2j * np.pi * k * n / N)
        X = np.dot(e, x)
        return X

    def showDft(self):
        X = self.dft()
        print(X)
        print(np.angle(X, deg=True))
        N = len(X)
        plt.ylabel('Signal Value')
        plt.xlabel('Frequency')
        plt.stem(np.arange(N), abs(X))
        # plt.plot(np.arange(N), abs(X))
        plt.grid(True)
        plt.show()

    def idft(self, X):
        N = len(X)
        n = np.arange(N)
        k = n.reshape((N, 1))
        e = np.exp((-2j * np.pi / N) * (-k * n))
        x = (1 / N) * np.dot(e, X)
        return x

    def showIdft(self, X):
        x = self.idft(X)
        x = np.real(x) # real number
        print(x)

    def fft(self, x):
        n = len(x)
        if n == 1:
            return x
        even = self.fft(x[0::2])
        odd = self.fft(x[1::2])
        T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
        return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

    def getFft(self):
        x = self.getAmplitude()
        return self.fft(x)

    def showFft(self):
        x = self.getAmplitude()
        X = np.array(self.fft(x))
        print(X)
        print(np.angle(X, deg=True))
        N = len(X)
        plt.ylabel('Signal Value')
        plt.xlabel('Frequency')
        plt.plot(np.arange(N), abs(X))
        plt.grid()
        plt.show()

    def fftUsingDefineMethod(self):
        return np.fft.fft(self.getAmplitude())

    def showFftUsingDefineMethod(self):
        X = np.array(self.fftUsingDefineMethod())
        print(X)
        fs = self.getFs()
        freqs = np.fft.fftfreq(len(self.getAmplitude()), 1 / fs)
        print(np.angle(X, deg=True))
        plt.plot(freqs, abs(X))
        plt.subplots_adjust(hspace=0.35)
        plt.grid(True)
        plt.show()

    def ifft(self, X):
        n = len(X)
        if n == 1:
            return X
        even = self.fft(X[0::2])
        odd = self.fft(X[1::2])
        T = [cmath.exp(2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
        return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

    def showIfft(self, X):
        x = np.array(self.ifft(X))
        N = len(x)
        print((1 / N) * (np.real(x)))

    def getEnergy(self):
        return np.sum(np.power(self.getAmplitude(), 2))

    def getWattage(self):
        E = self.getEnergy()
        n = symbols('n')
        return limit(E / (2 * n + 1), n, oo)

    def readSignal(filepath):
        wav_obj = wave.open(str(filepath), 'rb')
        sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
        n_samples = wav_obj.getnframes()
        t_audio = n_samples / sample_freq
        n_channels = wav_obj.getnchannels()
        signal_wave = wav_obj.readframes(n_samples)
        signal_array = np.frombuffer(signal_wave, dtype=np.int16)
        times = np.linspace(0, t_audio, num=n_samples)
        return Signal(signal_array, 0, sample_freq, 'Signal from' + str(filepath), len(times))
