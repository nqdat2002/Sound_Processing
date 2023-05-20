from SignalObject import *
import numpy as np
import wave

if __name__ == '__main__':

    wav_obj = wave.open('../SoundEnhancement/sang-amthanh3.wav', 'rb')
    sample_freq = wav_obj.getframerate()
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / sample_freq
    print(sample_freq)
    n_channels = wav_obj.getnchannels()
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    times = np.linspace(0, t_audio, num=n_samples)
    print(sample_freq, n_samples, t_audio, n_channels, len(times))
    signal = Signal(signal_array, 0, sample_freq, 'testSang am thanh ', len(times))
    # signal.plotSignal()
    # signal.stemSignal()

    start = time.time()
    signal.showFftUsingDefineMethod()
    end = time.time()
    print(end - start, 's')

    # start = time.time()
    # signal.showFft()
    # end = time.time()
    # print(end - start, 's')
