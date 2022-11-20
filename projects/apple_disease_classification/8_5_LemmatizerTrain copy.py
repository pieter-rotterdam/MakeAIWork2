import numpy as np
import nltk
import json
import pickle
import re
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import gradient_descent_v2
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

tokenized_words=[]
classes = []
doc = []
ignoring_words = ['?', '!']
data_file = open('../../jsonfile/intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern) #tokenizing
        tokenized_words.extend(w)
        doc.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(words.lower()) for words in tokenized_words if w not in ignoring_words] #lemmatization

lemmatized_words = sorted(list(set(lemmatized_words))) 
classes = sorted(list(set(classes)))

pickle.dump(lemmatized_words,open('lem_words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training_data = []

empty_array = [0] * len(classes)

for d in doc:

    bag_of_words = []

    pattern = d[0]

    pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]

    for w in lemmatized_words:

        bag_of_words.append(1) if w in pattern else bag_of_words.append(0)

    output_row = list(empty_array)

    output_row[classes.index(d[1])] = 1

    training_data.append([bag_of_words, output_row])

random.shuffle(training_data)

training = np.array(training_data)

x_train = list(training[:,0])

y_train = list(training[:,1])

bot_model = Sequential()
bot_model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
bot_model.add(Dropout(0.5))
bot_model.add(Dense(64, activation='relu'))
bot_model.add(Dropout(0.5))
bot_model.add(Dropout(0.25))
bot_model.add(Dense(len(y_train[0]), activation='softmax'))

sgd = gradient_descent_v2.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
bot_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


x_train = np.array(x_train)
y_train = np.array(y_train)
hist = bot_model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)

bot_model.save('chatbot_model85.h5', hist)