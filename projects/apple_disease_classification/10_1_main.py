#!/usr/bin/env python

import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pylab as plt
from jproperties import Properties
from matplotlib import pyplot as plt
import numpy as np
import os
import PIL.Image as Image
import random as python_random
from sklearn import metrics
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

# seed setting for repeated results in a a comparable way
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)


####### placeholder for variables.


# configs = Properties()

# with open('./apple.properties', 'rb') as read_prop:
# 	configs.load(read_prop)
	
# prop_view = configs.items()

# for item in prop_view:
#     use_dir =(item[1].data)
use_path = './transferData/Use/'

Use_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=use_path, target_size=(224,224), class_mode='None', shuffle=True)

imageCountTest = len(test_batches) # 4x32 images is 128
print (imageCountTest)



# #normalise dataset
# useData = useData.map(lambda x,y: (x/255, y))

# batch = tf.data.useData.random(seed=4).take(80)







# def chatbot()
#     open JSON
#     import class and stats
#     chat()

# def classify (predictions)
#     Class = argmax predictions
#     class = count argmax
#     stats = create stats
#     return (class, stats)

# def model_predict(batch)
#     load model
#     predict (batch)
#     return (predictions)

# def load_data_directory()
#     data = "directory" #as done in notebooks
#     scale = "scaling operations"
#     dataset = tf from directory
#     batches =
#     batch = element from batches
#     return (batch)

# def main()
#     batch = load_data_directory()
#     predictions = model_predict(batch)
#     classify(predictions)
#     chatbot()

#     1. random batches, 80 appels zonder terugleggen
#     2. prediction over batch


# pop = 

# 80 appels mag random met terug leggen

