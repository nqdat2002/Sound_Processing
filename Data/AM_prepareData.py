import matplotlib

matplotlib.interactive(False)
matplotlib.use('Agg')

# find all of the files in the directory
import os
import gc

basePath = "./xeno-canto-dataset/"
melsPath = "./mels-5class/"

birds = []  # list of all birds
for root, dirs, files in os.walk(basePath):
    if root == basePath:
        birds = dirs
birds50 = []
flist = []  # list of all files
blist = []  # list of files for one bird
i50 = 0
for i, bird in enumerate(birds):
    for root, dirs, files in os.walk(basePath + bird):
        for file in files:
            if file.endswith(".mp3"):
                blist.append(os.path.join(root, file))
        # print(root)
    if len(blist) > 50:
        i50 = i50 + 1
        print(i50, ". Found ", len(blist), ' files for ', bird, '(', i + 1, ')')
        birds50.append(bird)
        flist.append(blist)
    blist = []

print(birds50)

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


def saveMel(signal, directory):
    gc.enable()
    # MK_spectrogram modified
    N_FFT = 1024  #
    HOP_SIZE = 1024  #
    N_MELS = 128  # Higher
    WIN_SIZE = 1024  #
    WINDOW_TYPE = 'hann'  #
    FEATURE = 'mel'  #
    FMIN = 1400

    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(6, 6)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    S = librosa.feature.melspectrogram(y=signal, sr=sr,
                                       n_fft=N_FFT,
                                       hop_length=HOP_SIZE,
                                       n_mels=N_MELS,
                                       htk=True,
                                       fmin=FMIN,  # higher limit ##high-pass filter freq.
                                       fmax=sr / 2)  # AMPLITUDE
    librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN)  # power = S**2

    fig.savefig(directory)
    plt.ioff()
    # plt.show(block=False)
    fig.clf()
    ax.cla()
    plt.clf()
    plt.close('all')


import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm_notebook as tqdm

size = {'desired': 5,  # [seconds]
        'minimum': 4,  # [seconds]
        'stride': 0,  # [seconds]
        'name': 5  # [number of letters]
        }  # stride should not be bigger than desired length

print('Number of directories to check and cut: ', len(flist))

# step = (size['desired']-size['stride'])*sr # length of step between two cuts in seconds
step = 1
if step > 0:
    for bird, birdList in enumerate(flist):
        print("Processing ", bird, '. ', birds50[bird], "...")
        for birdnr, path in tqdm(enumerate(birdList)):
            path = path.replace("\\", '/')
            # print(path)
            # load the mp3 file
            directory = melsPath + str(bird) + birds50[bird][:size['name']] + "/"
            # print(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)

            if not os.path.exists(directory + path.rsplit('/', 1)[1].replace(' ', '')[:-4] + "1_1.png"):
                # print(directory + path.rsplit('/', 1)[1].replace(' ', '')[:-4] + "1_1.png")
                signal, sr = librosa.load(path)  # sr = sampling rate
                step = (size['desired'] - size['stride']) * sr  # length of step between two cuts in seconds

                nr = 0
                for start, end in zip(range(0, len(signal), step), range(size['desired'] * sr, len(signal), step)):
                    # cut file and save each piece
                    nr = nr + 1
                    # save the file if its length is higher than minimum
                    if end - start > size['minimum'] * sr:
                        melpath = path.rsplit('/', 1)[1]
                        melpath = directory + melpath.replace(' ', '')[:-4] + str(nr) + "_" + str(nr) + ".png"
                        saveMel(signal[start:end], melpath)
                    # print('New file...',start/sr,' - ',end/sr)
                    # print('Start: ',start,'end: ', end, 'length: ', end-start)

            pass
else:
    print("Error: Stride should be lower than desired length.")

print('Number of files after cutting: ')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ilist=[]
for root, dirs, files in os.walk(melsPath):
    print(dirs)
    for file in files:
        if file.endswith(".png"):
            ilist.append(os.path.join(root, file))
img=mpimg.imread(ilist[0])
imgplot = plt.imshow(img)
plt.show()
img=mpimg.imread(ilist[100])
imgplot1 = plt.imshow(img)
plt.show()
img=mpimg.imread(ilist[1000])
imgplot2 = plt.imshow(img)
plt.show()
img=mpimg.imread(ilist[4000])
imgplot3 = plt.imshow(img)
plt.show()
print("Found ",len(ilist)," files")
