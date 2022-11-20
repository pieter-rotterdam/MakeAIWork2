import pickle
import numpy as np
import json
from keras.models import load_model
import random
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk

intents_file = json.loads(open('../../jsonfile/intents.json').read())
lem_words = pickle.load(open('lem_words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
bot_model = load_model('chatbot_model85.h5')

def cleaning(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

def bag_ow(text, words, show_details=True):
    sentence_words = cleaning(text)
    bag_of_words = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag_of_words[i] = 1
    return (np.array(bag_of_words))

def class_prediction(sentence, model):
    p = bag_ow(sentence, lem_words,show_details=False)
    result = bot_model.predict(np.array([p]))[0]
    ER_THRESHOLD = 0.30
    f_results = [[i,r] for i,r in enumerate(result) if r > ER_THRESHOLD]
    f_results.sort(key=lambda x: x[1], reverse=True)
    intent_prob_list = []
    for i in f_results:
        intent_prob_list.append({"intent": pred_class[i[0]], "probability": str(i[1])})
    return intent_prob_list

def getbotResponse(ints, intents):
    tag = ints[0]['intent']
    intents_list = intents['intents']
    for intent in intents_list:
        if(intent['tag']== tag):
            result = random.choice(intent['responses'])
            break
    return result

def bot_response(text):
    ints = class_prediction(text, bot_model)
    response = getbotResponse(ints, intents)
    return response

