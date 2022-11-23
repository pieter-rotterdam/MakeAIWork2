import os
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
import numpy as np

usePath = './data/Use/'

useBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=usePath, target_size=(224,224))

model = tf.keras.models.load_model(
       ('./models/appleclassifier73transfer.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer})

yhat = model.predict(useBatches)
pred = (np.argmax(yhat, axis=-1))

print (pred)
