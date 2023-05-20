import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from scipy import signal
import timeit
import time
import wave
from scipy.io.wavfile import read
from SignalObject import *

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def run():
    # Filter requirements.
    order = 6
    fs = 30.0  # sample rate, Hz
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)
    print(b)
    print(a)
    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    print(w)
    print(h)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    #
    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 5.0  # seconds
    n = int(T * fs)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t) \
           + 0.5 * np.sin(12.0 * 2 * np.pi * t)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)

    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()


def LowPassFillter(filepathSrc, filepathRes, cutoff_freq = 4000, filter_size = 10001):
    wav_obj = wave.open(str(filepathSrc), 'rb')
    sample_freq = wav_obj.getframerate() # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)


    # signalbefore = Signal.readSignal(filepathSrc)
    # signalbefore.plotSignal()
    sampling_freq = n_samples
    filter_coeffs = signal.firwin(filter_size, cutoff_freq / (sampling_freq / 2))
    filtered_audio_signal = signal.convolve(signal_array, filter_coeffs, mode='same')
    filtered_audio_frames = filtered_audio_signal.astype(np.int16).tobytes()
    filtered_audio_file = wave.open(filepathRes, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

    # signalafter = Signal.readSignal(filepathRes)
    # signalafter.plotSignal()
    # signalafter.showFftUsingDefineMethod()

def main():
    LowPassFillter()
if __name__ == '__main__':
    # while True:
    #     opt = int(input("Your Option is: "))
    #     if opt == 0:
    #         break
    #     elif opt == 1:
    #         main()
    #     else:
    #         run()
    # LowPassFillter('sang-amthanh2.wav', 'res2.wav', 4000, 10000)
    run()