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


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
text = corpus[2]
encoded_input = tokenizer.encode(text, return_tensors='tf')
output = model(encoded_input)
greedy_output = model.generate(encoded_input, max_length=50, top_k=50, top_p=.95, do_sample=True)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# pip install -q gpt-2-simple
# import gpt_2_simple as gpt2