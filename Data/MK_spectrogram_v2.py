import matplotlib.pyplot as plt
import librosa, librosa.display, IPython.display as ipd
import numpy as np
import json
from mutagen.mp3 import MP3
from statistics import mean, median
import noisereduce as nr
from sklearn import preprocessing
import contextlib
import wave
from scipy.io import wavfile
import os
import pandas as pd

N_FFT = 1024  # Number of frequency bins for Fast Fourier Transform
HOP_SIZE = 1024  # Number of audio frames between STFT columns
SR = 44100  # Sampling frequency
N_MELS = 40  # Mel band parameters
WIN_SIZE = 1024  # number of samples in each STFT window
WINDOW_TYPE = 'hann'  # the windowin function
FEATURE = 'mel'  # feature representation
plt.rcParams['figure.figsize'] = (10, 4)

filePath = './xeno-canto-dataset/AcrocephalusArundinaceus/countries/AcrocephalusArundinaceus765347.mp3'
y, sr = librosa.load(filePath, duration=20, mono=True)
noise, _ = librosa.load(filePath, duration=20, mono=True)
y = nr.reduce_noise(y=y, sr=sr, y_noise=noise, stationary=False, prop_decrease=1)

plt.figure()
librosa.display.specshow(
    librosa.core.amplitude_to_db(
        librosa.feature.melspectrogram(
            y=y,
            S=np.abs(
                librosa.stft(
                    y, n_fft=N_FFT,
                    hop_length=HOP_SIZE,
                    window=WINDOW_TYPE,
                    win_length=WIN_SIZE)
            ) ** 2,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_SIZE,
            n_mels=N_MELS,
            htk=True,
            fmin=0.0,
            fmax=sr / 2.0),
        ref=1.0),
    sr=SR,
    hop_length=HOP_SIZE,
    x_axis='time',
    y_axis='mel')
plt.title('Mel spectrogram')
plt.show()

plt.figure()
spectogram = librosa.display.specshow(
    librosa.core.amplitude_to_db(
        librosa.feature.melspectrogram(
            y=y,
            sr=SR)),
    x_axis='time',
    y_axis='mel')
plt.title('Mel spectrogram - simple code')
plt.show()

plt.figure()
librosa.display.specshow(
    librosa.core.amplitude_to_db(
        librosa.feature.mfcc(
            dct_type=3,
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_SIZE,
            #    n_mels=N_MELS,
            htk=True,
            fmin=0.0,
            fmax=sr / 2.0
        ),
        ref=1.0),
    sr=SR,
    hop_length=HOP_SIZE,
    x_axis='time',
    y_axis='mel')
plt.title('Spectrogram with DCT')
plt.show()

spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)


def normalize(x, axis=0):
    return preprocessing.minmax_scale(x, axis=axis)

librosa.display.waveshow(y=y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
plt.title('Spectral Centroid')
plt.show()

o_env = librosa.onset.onset_strength(y=y, sr=sr,
                                     S=None, lag=1, max_size=1,
                                     detrend=False, center=True,
                                     feature=None, aggregate=None)
times = librosa.times_like(o_env, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
D = np.abs(librosa.stft(y))
plt.figure()
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         x_axis='time', y_axis='log')
plt.title('Power spectrogram')
plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(times, o_env, label='Onset strength')
plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
           linestyle='--', label='Onsets')
plt.axis('tight')
plt.legend(frameon=True, framealpha=0.75)
plt.show()


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return mfccsscaled


features = []
class_label = 'Acroc'
data = extract_features(filePath)
features.append([data, class_label])
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
print('Finished feature extraction from ', len(featuresdf), ' files')

print(featuresdf['feature'][0])
