from tensorflow import keras
model = keras.models.load_model('model/model.h5')
print(model)