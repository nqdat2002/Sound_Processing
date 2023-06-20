import efficientnet.tfkeras as efn
import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from efficientnet.keras import preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# SOUND_DIR = "./xeno-canto-dataset/AcrocephalusArundinaceus/countries/AcrocephalusArundinaceus765347.mp3"
# # listen to the recording
# ipd.display(ipd.Audio(SOUND_DIR))
#
# # load the mp3 file
# signal, sr = librosa.load(SOUND_DIR, duration=10)  # sr = sampling rate
#
# # plot recording signal
# plt.figure(figsize=(10, 4))
# librosa.display.waveshow(signal, sr=sr)
# plt.title("Monophonic")
# plt.show()
#
#
# # Plot spectogram
# plt.figure(figsize=(10, 4))
# D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
# librosa.display.specshow(D, y_axis="linear")
# plt.colorbar(format="%+2.0f dB")
# plt.title("Linear-frequency power spectrogram")
# plt.show()
#
# # Plot mel-spectrogram
# N_FFT = 1024
# HOP_SIZE = 1024
# N_MELS = 128
# WIN_SIZE = 1024
# WINDOW_TYPE = "hann"
# FEATURE = "mel"
# FMIN = 0
#
# S = librosa.feature.melspectrogram(
#     y=signal,
#     sr=sr,
#     n_fft=N_FFT,
#     hop_length=HOP_SIZE,
#     n_mels=N_MELS,
#     htk=True,
#     fmin=FMIN,
#     fmax=sr / 2,
# )
#
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(
#     librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN, y_axis="linear"
# )
# # plt.colorbar(format="%+2.0f dB")
# plt.title("Mel-scaled spectrogram")
# plt.show()
#
# # Plot mel-spectrogram with high-pass filter
# N_FFT = 1024
# HOP_SIZE = 1024
# N_MELS = 128
# # WIN_SIZE = 1024
# # WINDOW_TYPE = "hann"
# # FEATURE = "mel"
# FMIN = 1400
#
# S = librosa.feature.melspectrogram(
#     y=signal,
#     sr=sr,
#     n_fft=N_FFT,
#     hop_length=HOP_SIZE,
#     n_mels=N_MELS,
#     htk=True,
#     fmin=FMIN,
#     fmax=sr / 2,
# )
#
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(
#     librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN, y_axis="linear"
# )
# # plt.colorbar(format="%+2.0f dB")
# plt.title("Mel-scaled spectrogram with high-pass filter - 10 seconds")
# plt.show()

IM_SIZE = (224, 224)
BIRDS = [
    "0Acroc",
    "1Ember",
    "2Parus",
    "3Phyll",
    "4Sylvi"
]
DATA_PATH = "./5_class/"
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    fill_mode="nearest",
)
train_batches = train_datagen.flow_from_directory(
    DATA_PATH + "train",
    classes=BIRDS,
    target_size=IM_SIZE,
    class_mode="categorical",
    shuffle=True,
    batch_size=BATCH_SIZE,
)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(
    DATA_PATH + "val",
    classes=BIRDS,
    target_size=IM_SIZE,
    class_mode="categorical",
    shuffle=False,
    batch_size=BATCH_SIZE,
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_batches = test_datagen.flow_from_directory(
    DATA_PATH + "test",
    classes=BIRDS,
    target_size=IM_SIZE,
    class_mode="categorical",
    shuffle=False,
    batch_size=BATCH_SIZE,
)

# Define CNN's architecture
net = efn.EfficientNetB3(
    include_top=False, weights="imagenet", input_tensor=None, input_shape=(224, 224, 3)
)
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(BIRDS), activation="softmax", name="softmax")(x)
net_final = Model(inputs=net.input, outputs=output_layer)
net_final.compile(
    optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
)

print(net_final.summary())

# Estimate class weights for unbalanced dataset
class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(train_batches.classes), y=train_batches.classes
)
class_weights_dic = {}
for i in range(len(class_weights)):
    class_weights_dic[i] = class_weights[i]

# Define callbacks
ModelCheck = ModelCheckpoint(
    "models/efficientnet_checkpoint.h5",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq=1,
)

ReduceLR = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=3e-4)

# Train the model
net_final.fit(
    train_batches,
    validation_data=valid_batches,
    epochs=20,
    steps_per_epoch=400,
    class_weight=class_weights_dic,
    callbacks=[ModelCheck, ReduceLR],
)