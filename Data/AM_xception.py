from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.xception import Xception, preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

IM_SIZE = (299, 299)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.1,
                                   fill_mode='nearest')

train_batches = train_datagen.flow_from_directory('./5_class/train',
                                                  classes=['0Acroc', '1Ember', '2Parus', '3Phyll', '4Sylvi'],
                                                  target_size=IM_SIZE,
                                                  class_mode='categorical', shuffle=True,
                                                  batch_size=16)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory('./5_class/val',
                                                  classes=['0Acroc', '1Ember', '2Parus', '3Phyll', '4Sylvi'],
                                                  target_size=IM_SIZE,
                                                  class_mode='categorical', shuffle=False,
                                                  batch_size=16)

# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class nr ', idx, ' -> ', cls)
print('****************')

net = Xception(include_top=False,
               weights='imagenet',
               input_tensor=None,
               input_shape=(299, 299, 3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(5, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:20]:
    layer.trainable = False
for layer in net_final.layers[20:]:
    layer.trainable = True
net_final.compile(optimizer=Adam(learning_rate=5e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
print(net_final.summary())

import numpy as np

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
                                                    class_weight='balanced',
                                                    classes=np.unique(train_batches.classes),
                                                    y=train_batches.classes)
class_weights_dic = {}
for i in range(len(class_weights)):
    class_weights_dic[i] = class_weights[i]
print(class_weights_dic)
# train the models
net_final.fit(train_batches,
                        validation_data=valid_batches,
                        validation_steps=40,
                        epochs=20,
                        steps_per_epoch=400,
                        class_weight=class_weights_dic)

# save trained weights
net_final.save('./models/AM_xception_5classes.h5')


# net_final.compile(optimizer=Adam(learning_rate=1e-5),
#                   loss='categorical_crossentropy', metrics=['accuracy'])

# # train the models
# net_final.fit(train_batches,
#                         validation_data=valid_batches,
#                         epochs=20)
#
# net_final.compile(optimizer=Adam(lr=3e-6),
#                   loss='categorical_crossentropy', metrics=['accuracy'])
# # train the models
# net_final.fit(train_batches,
#                         steps_per_epoch=8 * 20,
#                         validation_data=valid_batches,
#                         epochs=20)
