import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import time

name_stan = 'generated_data/stan_gentext_20210106_224837.txt'
name_cartman = 'generated_data/cartman_gentext_20210107_003347.txt'
name_kyle = 'generated_data/kyle_gentext_20210107_013016.txt'

def read_in_text_file_from_gpt2(name):
    """reads in the gpt2 text files

    Args:
        name (string): name of generated text file

    Returns:
        dataframe: dataframe of specified text file
    """
    lines_char = []
    file1 = open(name, 'r')
    Lines = file1.readlines()
    for line in Lines: 
        lines_char.append(line.strip())
    df = pd.DataFrame(lines_char, dtype=str, columns = ['Document'])
    df = df[df['Document'] != '====================']
    return df

def create_corpus(gen_name, r_name):
    """create a corpus list for specified charachter

    Args:
        gen_name (string): name of generated text file
        r_name (string): name of real text file

    Returns:
        tuple, lists: the real and generate corpus
    """
    real = pd.read_csv(f'data/{r_name}.csv', names = ['Document'], dtype = str)
    gen_df = read_in_text_file_from_gpt2(gen_name)
    gen_short = gen_df[:len(real)]
    corpus_r = [str(row) for row in real['Document']]
    corpus_f = [str(row) for row in gen_short['Document']]
    return corpus_f, corpus_r

def create_corpus_tf_gen(gen_name, r_name):
    """create a corpus for the tensorflow generated text

    Args:
        gen_name (string): name of generated text file
        r_name (string): name of real text file

    Returns:
        tuple of lists: corpus for real and fake text
    """
    real = pd.read_csv(f'data/{r_name}.csv', names = ['Document'], dtype = str)
    gen_df = pd.read_csv(f'generated_data/{gen_name}.csv')
    gen_df.drop('Unnamed: 0', axis = 1, inplace=True)
    gen_short = gen_df[:len(real)]
    corpus_r = [str(row) for row in real['Document']]
    corpus_f = [str(row) for row in gen_short['Document']]
    return corpus_f, corpus_r

def tf_idf(corpus):
    """vectorizes the specified corpus

    Args:
        corpus (list of strings): corpus of a character

    Returns:
        tfidf_df: dataframe of tfidf vectors
    """
    vectorizor = TfidfVectorizer(stop_words = 'english', strip_accents = 'ascii')
    X = vectorizor.fit_transform(corpus)
    tfidf_df = pd.DataFrame(X.todense(), columns = sorted(vectorizor.vocabulary_))
    return tfidf_df



corpus_kf, corpus_kr = create_corpus(name_kyle, 'kyle') 
def create_fulldf(corpus_r, corpus_f):
    tfidf_df= tf_idf(corpus_r)
    tfidf_df_fake = tf_idf(corpus_f)

    tfidf_df['label'] = True
    tfidf_df_fake['label'] = False
    full_df_wnans = pd.concat([tfidf_df, tfidf_df_fake])
    full_df = full_df_wnans.fillna(0)
    label = full_df.pop('label')
    return label, full_df

tf_gen_name_stan = 'stan_tf_gen'
tf_corpus_f, tf_corpus_r = create_corpus_tf_gen(tf_gen_name_stan, 'stan')
tf_label, tf_df = create_fulldf(tf_corpus_r, tf_corpus_f)



X_train, X_test, y_train, y_test = train_test_split(tf_df, tf_label)

print('fitting model')
model = RandomForestClassifier()
start = time.time()
model.fit(X_train, y_train)
stop = time.time()
preds = model.predict(X_test)
print(f'{tf_gen_name_stan}:{model.score(X_test, y_test)}')
print(f"Training time: {stop - start}s")
