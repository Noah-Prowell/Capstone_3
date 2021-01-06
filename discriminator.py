import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

name = 'stan_gentext_20210106_224837.txt'
def read_in_text_file(name):
    lines_char = []
    file1 = open(name, 'r')
    Lines = file1.readlines()
    for line in Lines: 
        lines_char.append(line.strip())
    df = pd.DataFrame(lines_char, dtype=str, columns = ['Document'])
    df = df[df['Document'] != '====================']
    return df



stan_real = pd.read_csv('data/stan.csv', names = ['Document'], dtype = str)
stan_gen = read_in_text_file(name)
stan_gen_short = stan_gen[:len(stan_real)]

corpus_sr = [str(row) for row in stan_real['Document']]
corpus_sf = [str(row) for row in stan_gen['Document']]

def tf_idf(corpus):
    vectorizor = TfidfVectorizer(stop_words = 'english', strip_accents = 'ascii')
    X = vectorizor.fit_transform(corpus)
    tfidf_df = pd.DataFrame(X.todense(), columns = sorted(vectorizor.vocabulary_))
    return X, tfidf_df, vectorizor

X, stan_tfidf_df, vec = tf_idf(corpus_sr)

Y, stan_tfidf_df_fake, vec_fake = tf_idf(corpus_sf)

stan_tfidf_df['label'] = True
stan_tfidf_df_fake['label'] = False

full_df_wnans = pd.concat([stan_tfidf_df, stan_tfidf_df_fake])
full_df = full_df_wnans.fillna(0)
label = full_df.pop('label')

X_train, X_test, y_train, y_test = train_test_split(full_df, label)

print('fitting model')
model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(model.score(X_test, y_test))
