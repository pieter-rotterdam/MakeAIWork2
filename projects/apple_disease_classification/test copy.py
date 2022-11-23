import os
import random
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
import numpy as np

usePath = './data/Use/'

useBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=usePath, target_size=(224,224))

#create sample numbers 1-317
def display():
    samples = []
    for i in range(318): # This is just to tell you how to create a list.
        samples.append(i)
    return samples

batchSampleNbrs=display()
removedSampleNbrs = batchSampleNbrs.pop(0)

# pick 10 random samples in the list of 317 and create variable sample1-10
i = 1
for n in range(10):
    globals()["sample" + str(i)] = random.sample(batchSampleNbrs, k=80)
    i += 1

# print (sample9) #test to see how random sample looks

#load model
model = tf.keras.models.load_model(
       ('./models/appleclassifier73transfer.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer})

batch2= model.predict(useBatches)[(sample2)]
pred2 = (np.argmax(batch2, axis=-1))

print (pred2)
# batch1 = model.predict(useBatches)[:80]
# pred1 = (np.argmax(batch1, axis=-1))

# print (pred1)
# use_samples = [5, 38, 3939, 27389]
# samples_to_predict = []
# []








# sample1=
# sample2=random.sample(batchSampleNbrs, k=80)
# sample3=random.sample(batchSampleNbrs, k=80)
# sample4=random.sample(batchSampleNbrs, k=80)