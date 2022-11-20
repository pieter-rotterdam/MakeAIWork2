import json 
import numpy as np
from tensorflow import keras
import colorama 
colorama.init()
from colorama import Fore, Style
import pickle

with open('../../jsonfile/intents.json') as file:
    data = json.load(file)

def chat():
    # load trained model
    model = keras.models.load_model('SequentialChatmodel')

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
        if inp.lower() == "q":
            break
        if inp.lower() == "stop":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "AQL assistant Tim Apple" + Style.RESET_ALL , np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with AQL assistant Tim Apple (type quit to stop)!" + Style.RESET_ALL)
chat()