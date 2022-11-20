import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

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
        batchStatus = f'The batch has been qualified as: Class 1\n They are suitable for supermarkets and greengrocers.'
    elif nbrNormal >= 75: #75:
        batchStatus = f'The batch has been qualified as: Class 2\n They are suitable to be used in apple sauce.'
    elif nbrNormal >= 73: #73:
         f'The batch has been qualified as: Class 3\n They are suitable to be used in apple syrup.'
    else:
        batchStatus = f'The batch has been rejected\n This is too bad for your boss.'

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
    max_len = 20
    
    while True:
        print(Fore.LIGHTMAGENTA_EX + "PinkLady User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break
        if inp.lower() == "exit":
            break
        if inp.lower() == "stop":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = [responseReplacer(response) for response in tg['responses']]
                print(Fore.GREEN + "AQL assistant Tim Apple:" + Style.RESET_ALL , np.random.choice(list(responses)))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))
chat()

#https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281

# also tried with no luck: https://www.analyticsvidhya.com/blog/2021/06/learn-to-develop-a-simple-chatbot-using-python-and-deep-learning/

# if i had all the time in the world for a chatbot project #https://towardsdatascience.com/complete-guide-to-building-a-chatbot-with-spacy-and-deep-learning-d18811465876
# gui i've seen often: https://solozano0725.medium.com/retrieval-based-chatbots-using-nltk-keras-e4f86b262b17