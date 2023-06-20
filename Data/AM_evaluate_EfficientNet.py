from tensorflow.python.keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import efficientnet.tfkeras as efn

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_batches = test_datagen.flow_from_directory('./5_class/val',
                                                target_size=(224, 224),
                                                class_mode='categorical', shuffle=False,
                                                batch_size=16)
# show class indices
print('****************')
for cls, idx in test_batches.class_indices.items():
    print('Class nr ', idx, ' -> ', cls)
print('****************')

net = efn.EfficientNetB4(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(224,224,3))
x = net.output
x = Flatten()(x)
x = Dropout(0.4)(x)
output_layer = Dense(5, activation='softmax', name='softmax')(x)
loaded_model = Model(inputs=net.input, outputs=output_layer)

# load weights into new models
loaded_model.load_weights("./models/AM_efficientnet_5classes_callbackb4.h5")
print("Loaded models from disk")

# evaluate loaded models on test data
loaded_model.compile(optimizer=Adam(learning_rate=5e-5),
                     loss='categorical_crossentropy', metrics=['accuracy'])

score = loaded_model.evaluate(test_batches, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

from keras import metrics

Y_pred = loaded_model.predict(test_batches, 4565 // 4 + 1)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_batches.classes, y_pred))
print('Classification Report')
target_names = ['0Acroc', '1Ember', '2Parus', '3Phyll', '4Sylvi']
print(classification_report(test_batches.classes, y_pred, target_names=target_names))
print(classification_report(test_batches.classes, y_pred, target_names=target_names))


