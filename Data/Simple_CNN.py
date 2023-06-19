import sys
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=5, activation='sigmoid'))
# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('./5_class/train',
                                                 target_size=(64, 64),
                                                 batch_size=16,
                                                 class_mode='categorical', shuffle=True)
valid_set = test_datagen.flow_from_directory('./5_class/val',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='categorical')

test_set = test_datagen.flow_from_directory('./5_class/test',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='categorical')
import numpy as np

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
                                                    class_weight='balanced',
                                                    classes=np.unique(training_set.classes),
                                                    y=training_set.classes)
class_weights_dic = {}
for i in range(len(class_weights)):
    class_weights_dic[i] = class_weights[i]
print(class_weights_dic)

classifier.fit(training_set,
                         epochs=20,
                         validation_data=valid_set,
                         steps_per_epoch=400,
                         class_weight=class_weights_dic)
print(classifier.summary())

classifier.save("./models/AM_SimpleCNN.h5")




