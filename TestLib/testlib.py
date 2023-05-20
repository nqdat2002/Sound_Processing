import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io.wavfile import read, write;
from IPython.display import Audio;
from numpy.fft import fft, ifft;

Fs, data = read('../ObjectDigitalProcessing/sang-amthanh.wav')
data = data[:, 0]
print('Sampling Frequency is', Fs)

Audio(data, rate = Fs)

plt.figure()
plt.plot(data)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Waveform of Test Audio')
plt.show()

write('output.wav', Fs, data)