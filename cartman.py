import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import random

df = pd.read_csv('data/cartman.csv', names = ['Document'], dtype = str)
corpus = [str(row) for row in df['Document']]
def tf_idf(corpus):
    vectorizor = TfidfVectorizer(stop_words = 'english', strip_accents = 'ascii')
    X = vectorizor.fit_transform(corpus)
    tfidf_df = pd.DataFrame(X.todense(), columns = sorted(vectorizor.vocabulary_))
    return X, tfidf_df, vectorizor

X, tfidf_df, vec = tf_idf(corpus)
word_tokens = vec.get_feature_names()
# configuration = GPT2Config()
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

def response(user_response):
    robo_response=''
    vectorizor = TfidfVectorizer(stop_words = 'english', strip_accents = 'ascii')
    X = vectorizor.fit_transform(corpus)
    vals = cosine_similarity(X[-1], X)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
            robo_response=robo_response+"I am sorry! I don't understand your terrible english"
            return robo_response
    else:
        robo_response = robo_response+corpus[idx]
        return robo_response


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

flag=True 
while flag:
    user_response = input()
    user_response=user_response.lower()
    if(user_response =='exit'):
        flag=False
        print("ROBO: Screw you guys I'm going home")
    else:
        if(greeting(user_response)!=None):
            print("ROBO: "+greeting(user_response))
        else:
            corpus.append(user_response)
            word_tokens=word_tokens+nltk.word_tokenize(user_response)
            final_words=list(set(word_tokens))
            print("ROBO: ",end="")
            print(response(user_response))
            corpus.remove(user_response)



