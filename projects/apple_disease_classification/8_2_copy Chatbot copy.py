# start this file from folder projects/apple_disease_classification for json file to open correctly
import nltk
# nltk.download('all') # first use only
# nltk.download('punkt') #first use only
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy
import tflearn
import tensorflow
import random

import json
import pickle
from matplotlib import pyplot as plt

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
numpy.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tensorflow.random.set_seed(1234)


with open('../../jsonfile/intents.json') as file:
    data = json.load(file)


nbrBlotch = 4 # deze later nog vervangen door import
nbrNormal = 72
nbrRot = 1
nbrScab = 3
nbrRej = int(nbrBlotch)+int(nbrRot)+int(nbrScab)
nbrInBatch = 80

perBlotch = int(nbrBlotch)/int(nbrInBatch)*100
perNormal = int(nbrNormal)/int(nbrInBatch)*100
perRot = int(nbrRot)/int(nbrInBatch)*100
perScab = int(nbrScab)/int(nbrInBatch)*100
perRej = int(nbrRej)/int(nbrInBatch)*100

if nbrInBatch != 80:
    print ("Please offer a batch of 80 apples for a correct quality control")
else:
    if nbrNormal >= 79: #79:                                          
        batchStatus = f'The batch has been qualified as: Class 1\n , this is suitable for supermarkets and greengrocers.'
    elif nbrNormal >= 75: #75:
        batchStatus = f'The batch has been qualified as: Class 2\n , this is suitable to be used in apple sauce.'
    elif nbrNormal >= 73: #73:
         f'The batch has been qualified as: Class 3\n, this is suitable to be used in apple syrup.'
    else:
        batchStatus = f'The batch has been rejected\n, this is too bad for you.'
    

try:
    x
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern) #tokenize the patterns
            words.extend(wrds) #extend the tokens
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels: #add unexisting tags to their labels
            labels.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# tensorflow.reset_default_graph()

# net = tflearn.input_data(shape=[None, len(training[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# net = tflearn.regression(net)

# model = tflearn.DNN(net)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
model.save('model10.tflearn')

# dit werkt om niet steeds opnieuw te trainen, als dit wel nodig is, try en except weghalen en ook de indent voor model.fit en model.save

# try:
#     model.load("modelDO.tflearn")
# except:

model.fit(training, output, n_epoch=1500, batch_size=8, show_metric=True)
model.evaluate(batch_size=8)
model.save("modelcopy.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def responseReplacer(response):
    response = response.replace("{nbrInBatch}", str(nbrInBatch))
    response = response.replace("{nbrBlotch}", str(nbrBlotch))
    response = response.replace("{perBlotch}", str(perBlotch))
    response = response.replace("{nbrNormal}", str(nbrNormal))
    response = response.replace("{perNormal}", str(perNormal))
    response = response.replace("{nbrRot}", str(nbrRot))
    response = response.replace("{perRot}", str(perRot))
    response = response.replace("{nbrScab}", str(nbrScab))
    response = response.replace("{perScab}", str(perScab))
    response = response.replace("{nbrRej}", str(nbrRej))
    response = response.replace("{perRej}", str(perRej))
    response = response.replace("{batchStatus}", str(batchStatus))

    return response

def chat():
    print("Start talking with Tim Apple to get stats on the apple quality control AQL (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:

            if tg['tag'] == tag:
                # The status of the batch is {batchStatus} 
                responses = [responseReplacer(response) for response in tg['responses']]             
                # modified responses = tg ['replace']

        print(random.choice(list(responses)))

# chat()



# random.choice(list(dict1))


        # for tg in data['intents']:
        #     if tg['tag'] == tag:
        #         responses = tg['responses']
