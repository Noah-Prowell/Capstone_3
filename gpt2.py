from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
from transformers import GPT2Config, GPT2Tokenizer, TFGPT2LMHeadModel, TFGPT2Model
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np
from transformers import pipeline, set_seed
#read in text
df = pd.read_csv('data/cartman.csv', names = ['Document'], dtype = str)
corpus = [str(row) for row in df['Document']]

"""
# Tokenize the data
vec = CountVectorizer()
word_tokenizer = vec.build_tokenizer()
doc_terms_list_train = [word_tokenizer(doc) for doc in corpus]

vectorizor = TfidfVectorizer(stop_words = 'english', strip_accents = 'ascii')
X = vectorizor.fit_transform(corpus)
word_tokens = vectorizor.get_feature_names()
# tfidf_df = pd.DataFrame(X.todense(), columns = sorted(vectorizor.vocabulary_), dtype = np.int64)
"""

"""
tokenizer = GPT2Tokenizer()
#make config
config = GPT2Config(vocab_size=len(word_tokens))

#make the model
model = TFGPT2LMHeadModel(config)

optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defining our metric which we want to observe
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# compiling the model
model.compile(optimizer=optimizer, loss= loss, metrics=[metric])
num_epoch = 1
history = model.fit(, epochs=num_epoch)
"""


# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# text = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2Model.from_pretrained('gpt2')
text = corpus[0]
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)