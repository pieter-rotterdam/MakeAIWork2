# Importing modules
import re
from nltk.corpus import wordnet

# Building a list of Keywords
list_words=['help', 'scab', 'blotch', 'rot', 'healthy', 'normal', 'good', 'regular', 'goodbye', 'hello', 'bad', 'sick', 'disease', 'unhealthy', 'defect', 'more', 'recommend', 'name', 'more', 'extra', 'joke', 'funny', 'thanks' ]
list_syn={}
for word in list_words:
    synonyms=[]
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            # Remove any special characters from synonym strings
            lem_name = re.sub('[^a-zA-Z0-9 \n\.]', ' ', lem.name())
            synonyms.append(lem_name)
    list_syn[word]=set(synonyms)
print (list_syn)