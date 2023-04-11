#import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
import nltk
import pyttsx3
# import speech_recognition as sr
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, LSTM ,Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
import datetime



# pred_ip = gui.input_text.get()
#import datasets
with open('content.json') as content:
    datal=json.load(content)

#getting all data on list
tags=[]
inputs=[]
responses={}
for intent in datal['intents']:
    responses[intent['tag']]=intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])

#converting to data frame
data=pd.DataFrame({"inputs":inputs,"tags":tags})
print(data)

#REMOVING PUNCTUATION
data['inputs']=data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs']=data['inputs'].apply(lambda wrd:''.join(wrd))

#tokenize data
tokenizer=Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

#apply padding
x_train = pad_sequences(train)
#encoding data
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

#input length
input_shape = x_train.shape[1]
print(input_shape)
#define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
#output length
output_length = le.classes_.shape[0]
print("output length: ",output_length)

#neural network
i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model  = Model(i,x)

#compiling model
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
#training model
train = model.fit(x_train,y_train,epochs=310)

#voice as output
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

#wishing function
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 4 and hour < 12:
        speak("Good Morning!")

    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")

    else:
        speak("Good Evening!")

    speak("Banc this side. Please tell me how may i help you!")


#chatting
import random
wishMe()
