import colorama 
colorama.init()
from colorama import Fore, Style
import json 
import os
import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
import numpy as np

with open('../../jsonfile/intents.json') as file:
    data = json.load(file)
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"

#load data
usePath = './data/Use/'
useBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=usePath, target_size=(224,224))

#create sample numbers 1-317
def display():
    samples = []
    for i in range(317): # This is just to tell you how to create a list.
        samples.append(i)
    return samples
batchSampleNbrs=display()
# removedSampleNbrs = batchSampleNbrs.pop(0)

# create 10 random samples of 80 in the list of 317 and create variable sample1-10
list_samples = []
for k in range(11):
    listSamples = f'batch{k}'
    list_samples.append(listSamples)
# print(list_samples)

#load model
model = tf.keras.models.load_model(
       ('./models/appleclassifier73transfer.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer})

print ("Model Loaded, please wait for the predictions")

#make predictions and stats
predList = []
statsList= []
nbrInBatchList = []
nbrBlotchList = []
nbrNormalList = []
nbrRotList = []
nbrScabList = []
nbrRejList = []
percBlotchList = []
percNormalList = []
percRotList = []
percScabList = []
percRejList = []
batchStatusList=[]

for x in list_samples:
    y=random.sample(batchSampleNbrs, k=80)
    batch= model.predict(useBatches)[(y)]
    pred = (np.argmax(batch, axis=-1))  
    # print(f'Sample: {x} prediction: {pred} ')
    predList.append(pred)
    print(f'predictions {x} of 10 done.')

for pred in predList:
    unique2, counts2 = np.unique(pred, return_counts=True)
    dict(zip(unique2, counts2))
    stats = dict(zip(unique2, counts2))
    nbrInBatch= (len(pred))
    nbrInBatchList.append(nbrInBatch)
    nbrBlotch = stats[0]
    nbrBlotchList.append(nbrBlotch)
    nbrNormal = stats[1]
    nbrNormalList.append(nbrNormal)
    nbrRot = stats[2]
    nbrRotList.append(nbrRot)
    nbrScab = stats[3]
    percScabList.append(nbrScab)
    nbrRej = int(nbrBlotch)+int(nbrRot)+int(nbrScab)
    nbrRejList.append(nbrRej)

    percBlotch = int(nbrBlotch)/int(nbrInBatch)*100
    percBlotchList.append(percBlotch)
    percNormal = int(nbrNormal)/int(nbrInBatch)*100
    percNormalList.append(percNormal)
    percRot = int(nbrRot)/int(nbrInBatch)*100
    percRotList.append(percRot)
    percScab = int(nbrScab)/int(nbrInBatch)*100
    percScabList.append(percScab)
    percRej = int(nbrRej)/int(nbrInBatch)*100
    percRejList.append(percRej)

    if nbrInBatch != 80:
        print ("Please offer batches of 80 apples for a correct quality control")

    else:
        if nbrNormal >= 79:                                          
            batchStatus = (f'Class 1')
        elif nbrNormal >= 72:
            batchStatus = (f'Class 2')
        elif nbrNormal >= 69:
            (f'Class 3')
        else:
            batchStatus = (f'rejected')
    batchStatusList.append(batchStatus)

print (nbrNormalList)
print (batchStatusList)

def responseReplacer(response):
    response = response.replace("{nbrInBatch}", str(nbrInBatch))
    response = response.replace("{nbrBlotch}", str(nbrBlotch))
    response = response.replace("{perBlotch}", str(percBlotch))
    response = response.replace("{nbrNormal}", str(nbrNormal))
    response = response.replace("{perNormal}", str(percNormal))
    response = response.replace("{nbrRot}", str(nbrRot))
    response = response.replace("{perRot}", str(percRot))
    response = response.replace("{nbrScab}", str(nbrScab))
    response = response.replace("{perScab}", str(percScab))
    response = response.replace("{nbrRej}", str(nbrRej))
    response = response.replace("{perRej}", str(percRej))
    response = response.replace("{batchStatus}", str(batchStatus))
    return response    

def chat():
    # load trained model
    model = keras.models.load_model('sequentialChatModel')
    print(Fore.YELLOW + "Start messaging with AQL assistant Tim Apple (type quit to stop)!" + Style.RESET_ALL)
    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 200
    inp, stopCondition = "", ""

    while not stopCondition:
        print(Fore.LIGHTMAGENTA_EX + "PinkLady User: " + Style.RESET_ALL, end="")
        inp = input()
        stopCondition = inp.lower() in ["quit", "exit", "stop"]

        if (stopCondition):
            break
        # if inp.lower() == "quit":can
        #     break
        # if inp.lower() == "exit":
        #     break
        # if inp.lower() == "stop":
        #     break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = [responseReplacer(response) for response in tg['responses']]
                print(Fore.GREEN + "AQL assistant Tim Apple:" + Style.RESET_ALL , np.random.choice(list(responses)))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))
chat()
