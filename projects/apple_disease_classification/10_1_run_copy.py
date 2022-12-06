import colorama 
colorama.init()
from colorama import Fore, Back, Style
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

os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED" # workaround for cache issue

# global variables
modelLocation = './models/appleclassifier73transfer.h5'
usePath = './data/Use/'
batchAmmount = 10 #how many batches of apples do we want to feed the model
batchNr = 0
nbrFilesInFolder = 317 #how many images are in the folder where we create the batches from
aqlSampleSize = 80 # what is the required batch size for this aql check

# required lists for predictions and stats
predList = []
statsList = []
batchStatusList = []

#nbr related lists
nbrInBatchList = []
nbrBlotchList = []
nbrNormalList = []
nbrRotList = []
nbrScabList = []
nbrRejList = []
#perentage related list
perBlotchList = []
perNormalList = []
perRotList = []
perScabList = []
perRejList = []

# TODO
# batchScoreList = dict(
#     "nbrInBatchList": list(),
#     "nbrBlotchList": list()
# )

# TODO functie maken voor alle acties op een batch 

# data pipeline
def preprocessor(usePath):
    useBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=usePath, target_size=(224,224))
    return useBatches

#create sample numbers
def samplecreator(nbrFilesInFolder):
    samples = []
    for i in range(nbrFilesInFolder): # This is just to tell you how to create a list.
        samples.append(i)
    return samples

batchSampleNbrs = samplecreator(nbrFilesInFolder)

# create ammount samples of aqlSampleSize in the list of 317 and create variable sample1-10
listSamples = []
for k in range(10):
    listSample = f'batch{k}'
listSamples.append(listSample)
      
#load model
def model(modelLocation):
    tf.keras.models.load_model((modelLocation),
    custom_objects={'KerasLayer':hub.KerasLayer})
    return model
    print (Fore.WHITE + Back.GREEN + "Model Loaded, please wait for the predictions")

# classifier
def classifier(listSamples, useBatches):
    for x in listSamples:
        y = random.sample(batchSampleNbrs, aqlSampleSize)
        batch = model.predict(useBatches)[(y)]
        pred = (np.argmax(batch, axis=-1))  
    # print(f'Sample: {x} prediction: {pred} ')
        predList.append(pred)
        print(Fore.WHITE + Back.BLUE + f'predictions {x} of 10 done.')
        return predList

nbrInBatch = 0

def statcreator(predList):
    for pred in predList:    
        nbrInBatch = (len(pred))
        unique2, counts2 = np.unique(pred, return_counts=True)
        stats = dict(zip(unique2, counts2))
    
    # ammounts
        nbrBlotch = stats[0]
        nbrNormal = stats[1]
        nbrRot = stats[2]
        nbrScab = stats[3]
        nbrRej = int(nbrBlotch)+int(nbrRot)+int(nbrScab)

     # perentages
        perBlotch = round(int(nbrBlotch)/int(nbrInBatch)*100,2)
        perNormal = round(int(nbrNormal)/int(nbrInBatch)*100,2)
        perRot = round(int(nbrRot)/int(nbrInBatch)*100,2)
        perScab = round(int(nbrScab)/int(nbrInBatch)*100,2)
        perRej = round(int(nbrRej)/int(nbrInBatch)*100,2)
     
      # list with ammounts
        nbrBlotchList.append(nbrBlotch)
        nbrNormalList.append(nbrNormal)
        nbrRotList.append(nbrRot)
        nbrScabList.append(nbrScab)
       
     # lists with rejected and number of apples in batch
        nbrRejList.append(nbrRej)
        nbrInBatchList.append(nbrInBatch)
       
     # list with perentages
        perBlotchList.append(perBlotch)
        perNormalList.append(perNormal)
        perRotList.append(perRot)
        perScabList.append(perScab)
        perRejList.append(perRej)
    return nbrInBatch, nbrBlotch, nbrNormal, nbrRot, nbrScab, nbrRej, perBlotch, perNormal, perRot, perScab, perRej

def aql(nbrInBatch,nbrNormal):

    if nbrInBatch != aqlSampleSize:
        print ("Please offer the right ammount of apples per batch for a correct AQL quality control")
            # in case bath!=aqlSampleSize continue predition
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

            return batchStatusList, batchStatus, 

# chat
def responseReplacer(response, batchStatus, nbrInBatch, nbrBlotch, nbrNormal, nbrRot, nbrScab, nbrRej, perBlotch, perNormal, perRot, perScab, perRej):
    # general numbers
    response = response.replace("{nbrInBatch}", str(nbrInBatch))
    response = response.replace("{batchStatus}", str(batchStatus))

    # fault numbers
    response = response.replace("{nbrBlotch}", str(nbrBlotch))
    response = response.replace("{nbrNormal}", str(nbrNormal))
    response = response.replace("{nbrRot}", str(nbrRot))
    response = response.replace("{nbrScab}", str(nbrScab))
    response = response.replace("{nbrRej}", str(nbrRej))

    #fault percentages
    response = response.replace("{perBlotch}", str(perBlotch))
    response = response.replace("{perNormal}", str(perNormal))
    response = response.replace("{perRot}", str(perRot))
    response = response.replace("{perScab}", str(perScab))
    response = response.replace("{perRej}", str(perRej))
 
    return response    

#load json for chatresponses
with open('../../jsonfile/intents.json') as file:
    data = json.load(file)

def chat():
    # load trained model
    model = keras.models.load_model('sequentialChatModel')
    print(Fore.WHITE + Back.RED + "Start messaging with AQL assistant Tim Apple (type quit to stop)!" + Style.RESET_ALL)
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
        print(Fore.WHITE + Back.LIGHTMAGENTA_EX + "PinkLady User: " + Style.RESET_ALL, end="")
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
                print(Fore.WHITE+ Back.LIGHTBLUE_EX + "AQL assistant Tim Apple:" + Style.RESET_ALL , np.random.choice(list(responses)))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))


def main():
    preprocessor(usePath)
    samplecreator(nbrFilesInFolder)
    model(modelLocation)
    statcreator(predList)
    aql(nbrInBatch,nbrNormal)
    responseReplacer(response, batchStatus, nbrInBatch, nbrBlotch, nbrNormal, nbrRot, nbrScab, nbrRej, perBlotch, perNormal, perRot, perScab, perRej)
    chat()
if __name__ == '__main__':
    main()
