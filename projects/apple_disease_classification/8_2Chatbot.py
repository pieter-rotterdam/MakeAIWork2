import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random

import json
with open('../../jsonfile/intents.json') as file:
    data = json.load(file)

print (data["intents"])

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"] #turn everything in lower case
words = sorted(list(set(words))) #remove duplicates

labels = sorted(labels)

training = []
output = [] #this is the text in json file that also needs one hot encoding

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words: #if word exists add to bag, else 0
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:] #copy of out empty
    output_row[labels.index(docs_y[x])] = 1 #look through labels list, see where tag is and set value to 1 in output row

    training.append(bag)
    output.append(output_row)


training = numpy.array(training) #confersion to array for tflearn
output = numpy.array(output)

# tf.reset_default_graph() #see if this is replaced by something else?

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")