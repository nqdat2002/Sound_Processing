import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from scipy import signal
import window_function
from scipy.io.wavfile import read
import wave
import math

# I O(w) = -w*alpha, alpha = (N - 1) / 2, N lẻ, h(n) = h(N - n - 1)
# II O(w) = -w*alpha, alpha = (N - 1) / 2, N chẵn, h(n) = h(N - n - 1)

# III O(w) = +-pi/2 -w*alpha, alpha = (N - 1) / 2, N lẻ, h(n) = -h(N - n - 1)
# IV O(w) = +-pi/2 -w*alpha, alpha = (N - 1) / 2, N chẵn, h(n) = -h(N - n - 1)

# N là bậc của bộ lọc
# fc là tần số cắt
# window_function là loại hàm cửa sổ

def showGraph(filePath, fc1 = 0, fc2 = 0):
    file_name = filePath
    with wave.open(file_name, "rb") as wave_file:
        sample_rate = wave_file.getframerate()
        num_frames = wave_file.getnframes()
        num_channels = wave_file.getnchannels()
        sample_width = wave_file.getsampwidth()
        raw_data = wave_file.readframes(num_frames)

    # Biên độ theo thời gian
    data = np.frombuffer(raw_data, dtype=np.int16)

    times = np.linspace(0, len(data) / sample_rate, num=len(data))
    plt.subplot(4, 1, 1)
    plt.plot(times, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time domain waveform")
    plt.grid()

    # Năng lượng
    plt.subplot(4, 1, 2)
    plt.psd(data, Fs=sample_rate)
    if fc1 != 0 and fc2 == 0:
        plt.axvline(fc1, color='k')
    elif fc1 != 0 and fc2 != 0:
        plt.axvline(fc1, color='k')
        plt.axvline(fc2, color='k')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.title("Power Spectral Density")
    plt.grid()

    # Độ lớn Biên độ phổ theo miền tần số
    # thời gian = số lượng mẫu / tần số lấy mẫu
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(num_frames, 1 / sample_rate)
    # plt.plot(xf, np.abs(yf))
    plt.subplot(4, 1, 3)
    plt.plot(xf[0:len(xf)//2], np.abs(yf[0:len(yf)//2]))
    if fc1 != 0 and fc2 == 0:
        plt.axvline(fc1, color='k')
    elif fc1 != 0 and fc2 != 0:
        plt.axvline(fc1, color='k')
        plt.axvline(fc2, color='k')
    plt.title("The frequency response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()

    # pha
    plt.subplot(4, 1, 4)
    plt.plot(xf[0:len(xf) // 2], np.angle(yf[0:len(yf) // 2]), color='b')
    # plt.phase_spectrum(data, Fs = sample_rate // 2, color='red')
    # zf = np.array(yf)
    # phase = math.atan2(np.imag(zf), np.real(zf))
    # plt.plot(xf, phase)

    if fc1 != 0 and fc2 == 0:
        plt.axvline(fc1, color='k')
    elif fc1 != 0 and fc2 != 0:
        plt.axvline(fc1, color='k')
        plt.axvline(fc2, color='k')
    plt.title("The Phase response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radian)")
    plt.grid()

    plt.subplots_adjust(hspace=0.75)
    plt.show()

def LPF(N, fc = 2000, fs = 8000, lamda = 0):
    deltaW = 4 * np.pi / N
    fc_norm = fc / fs
    wc_norm = 2 * np.pi * fc_norm
    alpha = (N - 1) // 2

    n = np.arange(N)
    h = np.zeros_like(n, dtype=float)
    h[n == 0] = 2 * fc_norm
    h[n != 0] = (2 * fc_norm) * np.sinc(2 * fc_norm * (n[n != 0] - (N - 1) / 2))

    windowx = window_function.hanning(N)
    h_d = windowx * h
    return h_d

    # yf = np.fft.fft(h_d)
    # xf = np.fft.fftfreq(len(h_d), 1 / fc_norm)
    # plt.plot(xf, np.abs(yf))
    # plt.grid()
    # plt.show()
    # window_function.plotWindow(h, len(h))
    # window_function.plotWindow(windowx, len(windowx))
    # window_function.plotWindow(h_d, len(h_d))
    # return h_d

    # # Using IFT numpy.absolute(arr, out = None, ufunc ‘absolute’)
    # w = np.linspace(-wc_norm, wc_norm, N)
    # H = np.zeros_like(w)
    # H[np.abs(w) < wc_norm] = 1
    # n = np.arange(N)
    # h = (1 / N) * np.sum(H * np.exp(2j * np.pi * w * n[:, np.newaxis]), axis=1)
    # h = np.real(h)
    # windowx = window_function.hanning(N)
    # h_d = windowx * h
    # window_function.plotWindow(h_d, len(h_d))
    # return h_d

def HPF(N, fc = 2000, fs = 8000, lamda = 0):
    deltaW = 4 * np.pi / N
    fc_norm = fc / fs
    wc_norm = 2 * np.pi * fc_norm
    alpha = (N - 1) // 2

    n = np.arange(N)
    h = np.zeros_like(n, dtype=float)
    h[n == 0] = 1 - 2 * fc_norm
    h[n != 0] = - (2 * fc_norm) * np.sinc(2 * fc_norm * (n[n != 0] - (N - 1) / 2))
    windowx = window_function.hanning(N)
    h_d = windowx * h

    # yf = np.fft.fft(h_d)
    # xf = np.fft.fftfreq(len(h_d), 1 / fc_norm)
    # plt.plot(xf, np.abs(yf))
    # plt.grid()
    # plt.show()

    # window_function.plotWindow(h, len(h))
    # window_function.plotWindow(windowx, len(windowx))
    # window_function.plotWindow(h_d, len(h_d))
    return h_d

def BPF(N, fc1 = 2000, fc2 = 5000, fs = 8000, lamda = 0):
    deltaW = 4 * np.pi / N
    fc_norm1 = fc1 / fs
    fc_norm2 = fc2 / fs
    wc_norm1 = 2 * np.pi * fc_norm1
    wc_norm2 = 2 * np.pi * fc_norm2
    alpha = (N - 1) // 2

    n = np.arange(N)
    h, h1, h2 = np.zeros_like(n, dtype=float), np.zeros_like(n, dtype=float), np.zeros_like(n, dtype=float)
    h1[n == 0] = 2 * fc_norm1
    h1[n != 0] = (2 * fc_norm1) * np.sinc(2 * fc_norm1 * (n[n != 0] - (N - 1) / 2))
    h2[n == 0] = 2 * fc_norm2
    h2[n != 0] = (2 * fc_norm2) * np.sinc(2 * fc_norm2 * (n[n != 0] - (N - 1) / 2))
    h = h2 - h1

    windowx = window_function.hanning(N)

    h_d = windowx * h
    # yf = np.fft.fft(h_d)
    # xf = np.fft.fftfreq(len(h_d), 1 / fc_norm)
    # plt.plot(xf, np.abs(yf))
    # plt.grid()
    # plt.show()

    # window_function.plotWindow(h1, len(h1))
    # window_function.plotWindow(h2, len(h2))
    # window_function.plotWindow(windowx, len(windowx))
    # window_function.plotWindow(h_d, len(h_d))

    return h_d

def BSF(N, fc1 = 2000, fc2 = 5000, fs = 8000, lamda = 0):
    deltaW = 4 * np.pi / N
    fc_norm1 = fc1 / fs
    fc_norm2 = fc2 / fs
    wc_norm1 = 2 * np.pi * fc_norm1
    wc_norm2 = 2 * np.pi * fc_norm2
    alpha = (N - 1) // 2

    n = np.arange(N)
    h, h1, h2 = np.zeros_like(n, dtype=float), np.zeros_like(n, dtype=float), np.zeros_like(n, dtype=float)
    h1[n == 0] = 1 - 2 * fc_norm1
    h1[n != 0] =  - (2 * fc_norm1) * np.sinc(2 * fc_norm1 * (n[n != 0] - (N - 1) / 2))
    h2[n == 0] = 1 - 2 * fc_norm2
    h2[n != 0] =  - (2 * fc_norm2) * np.sinc(2 * fc_norm2 * (n[n != 0] - (N - 1) / 2))
    h = h2 - h1

    windowx = window_function.hanning(N)

    h_d = windowx * h
    # yf = np.fft.fft(h_d)
    # xf = np.fft.fftfreq(len(h_d), 1 / fc_norm)
    # plt.plot(xf, np.abs(yf))
    # plt.grid()
    # plt.show()

    # window_function.plotWindow(h1, len(h1))
    # window_function.plotWindow(h2, len(h2))
    # window_function.plotWindow(windowx, len(windowx))
    # window_function.plotWindow(h_d, len(h_d))
    return h_d

def TestLPF(filepathSrc, filepathRes, fc, filter_size):
    # Đọc file ra
    wav_obj = wave.open(str(filepathSrc), 'rb')
    sample_freq = wav_obj.getframerate() # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    showGraph(filepathSrc)

    h_d = LPF(filter_size, fc, n_samples // 2, 0) # dãy đáp ứng xung của bộ lọc, fnyquist = fs = sample_freq /  2

    filtered_audio_signal = signal.convolve(signal_array, h_d, mode='valid')
    filtered_audio_frames = filtered_audio_signal.astype(np.int16).tobytes()
    filtered_audio_file = wave.open(filepathRes, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

    # ghi vào file
    wav_obj = wave.open(str(filepathRes), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    showGraph(filepathRes, fc)

def TestHPF(filepathSrc, filepathRes, fc, filter_size):
    # Đọc file ra
    wav_obj = wave.open(str(filepathSrc), 'rb')
    sample_freq = wav_obj.getframerate() # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    showGraph(filepathSrc)

    h_d = HPF(filter_size, fc, n_samples // 2, 0) # dãy đáp ứng xung của bộ lọc, fnyquist = fs = sample_freq /  2

    filtered_audio_signal = signal.convolve(signal_array, h_d, mode='same')
    filtered_audio_frames = filtered_audio_signal.astype(np.int16).tobytes()
    filtered_audio_file = wave.open(filepathRes, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

    # ghi vào file
    wav_obj = wave.open(str(filepathRes), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    showGraph(filepathRes, fc)

def TestBPF(filepathSrc, filepathRes, fc1, fc2, filter_size):
    # Đọc file ra
    wav_obj = wave.open(str(filepathSrc), 'rb')
    sample_freq = wav_obj.getframerate() # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    showGraph(filepathSrc)

    h_d = BPF(filter_size, fc1, fc2, n_samples // 2, 0) # dãy đáp ứng xung của bộ lọc, fnyquist = fs = sample_freq /  2

    filtered_audio_signal = signal.convolve(signal_array, h_d, mode='valid')
    filtered_audio_frames = filtered_audio_signal.astype(np.int16).tobytes()
    filtered_audio_file = wave.open(filepathRes, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

    # ghi vào file
    wav_obj = wave.open(str(filepathRes), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    showGraph(filepathRes, fc1, fc2)

def TestBSF(filepathSrc, filepathRes, fc1, fc2, filter_size):
    # Đọc file ra
    wav_obj = wave.open(str(filepathSrc), 'rb')
    sample_freq = wav_obj.getframerate() # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    showGraph(filepathSrc)

    h_d = BSF(filter_size, fc1, fc2, n_samples // 2, 0) # dãy đáp ứng xung của bộ lọc, fnyquist = fs = sample_freq /  2

    filtered_audio_signal = signal.convolve(signal_array, h_d, mode='valid')
    filtered_audio_frames = filtered_audio_signal.astype(np.int16).tobytes()
    filtered_audio_file = wave.open(filepathRes, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

    # ghi vào file
    wav_obj = wave.open(str(filepathRes), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    showGraph(filepathRes, fc1, fc2)

def Equalizer(filepathSrc, filepathRes, fc1, fc2, filter_size, filter_number, type = 'I'):
    pass

def IOFile(filepathSrc, filepathRes, fc1=2000, fc2=8000, filterSize = 1001):
    # Đọc file ra
    wav_obj = wave.open(str(filepathSrc), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    h_d = BSF(filterSize, fc1, fc2, n_samples // 2, 0)  # dãy đáp ứng xung của bộ lọc, fnyquist = fs = sample_freq /  2

    filtered_audio_signal = signal.convolve(signal_array, h_d, mode='same')
    filtered_audio_frames = filtered_audio_signal.astype(np.int16).tobytes()
    filtered_audio_file = wave.open(filepathRes, 'wb')
    filtered_audio_file.setnchannels(n_channels)
    filtered_audio_file.setsampwidth(wav_obj.getsampwidth())
    filtered_audio_file.setframerate(sample_freq)
    filtered_audio_file.setnframes(n_samples)
    filtered_audio_file.writeframes(filtered_audio_frames)
    filtered_audio_file.close()

    # ghi vào file
    wav_obj = wave.open(str(filepathRes), 'rb')
    sample_freq = wav_obj.getframerate()  # tần số lấy mẫu
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

def main():
    TestLPF('oto.wav', 'otolpf.wav', fc=2000, filter_size=1001)
    # TestHPF('sang-amthanh2.wav', 'resulthpf.wav', fc=14000, filter_size=41)
    # TestBPF('sang-amthanh2.wav', 'resulbpf.wav', 2000, 8000, 1001)
    # TestBSF('sang-amthanh2.wav', 'resulbsf.wav', 2000, 8000, 1001)

    # print(LPF(41, fc = 2000, fs = 22050))
if __name__ == '__main__':
    main()