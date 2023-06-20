import json
from AM_downloadDataset import read_data

countries = ['Poland', 'Germany', 'Slovakia']

# initialize dictionary of a bird
bird = {'gen': 'Emberiza', 'spec': 'Citrinella', 'country': countries,
        'number of files': {'total': 0, 'quality': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, }},
        'length': {'min': 0, 'max': 0, 'mean': 0, 'median': 0}}

from mutagen.mp3 import MP3
from statistics import mean, median
import os

audioLengths = list()
audioLengthTemp = []
for country in countries:
    # find the driectory with recordings
    path = './xeno-canto-dataset/' + bird['gen'] + bird['spec'] + '/' + country
    print('Loading data from folder ' + path)

    # load info about the quality of the recording from json file
    qualityData = read_data('q', path)
    bird['number of files']['total'] = bird['number of files']['total'] + len(qualityData)
    for quality in bird['number of files']['quality']:
        bird['number of files']['quality'][quality] = bird['number of files']['quality'][quality] + qualityData.count(
            quality)

    # load MP3 file of every recording and check the length of a file
    idData = read_data('id', path)
    dataCounter = 0
    for audioFile in range(len(idData)):
        if os.path.exists(path + '/' + bird['gen'] + bird['spec'] + idData[audioFile] + '.mp3'):
            audioLengthTemp.append(
                MP3(path + '/' + bird['gen'] + bird['spec'] + idData[audioFile] + '.mp3').info.length)
            dataCounter = dataCounter + 1

    audioLengths = list(audioLengthTemp) + list(audioLengths)
    print("Loaded ", dataCounter, " out of ", len(idData), " files")

bird['length']['max'] = max(audioLengths)
bird['length']['min'] = min(audioLengths)
bird['length']['mean'] = mean(audioLengths)
bird['length']['median'] = median(audioLengths)

from random import sample

# find the driectory with recordings
path = './xeno-canto-dataset/' + bird['gen'] + bird['spec'] + '/' + 'Poland'
print('Loading data from folder ' + path)

# load json file:  read all id numbers
idData = read_data('id', path)
qualityData = read_data('q', path)

# select random 5 recordings from Poland
randFiles = sample(range(len(idData)), 3)
print('Selected random files number:', randFiles)

import IPython.display as ipd

for audioFile in randFiles:
    # path of random file
    filePath = path + '/' + bird['gen'] + bird['spec'] + idData[audioFile] + '.mp3'
    print('Play the file number ' + str(audioFile) + ', quality: ' + qualityData[audioFile])
    # show the recording and allow to play it
    ipd.display(ipd.Audio(filePath))

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

for audioFile in randFiles:
    # path of random file
    filePath = path + '/' + bird['gen'] + bird['spec'] + idData[audioFile] + '.mp3'

    # plot recording signal
    y, sr = librosa.load(filePath, duration=10)
    plt.figure(figsize=(10, 4))
    librosa.display.waveplot(y, sr=sr)
    plt.title('Monophonic - file number ' + str(audioFile))
    plt.show()

    # plot spectogram
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram - file number ' + str(audioFile))
    plt.show()
