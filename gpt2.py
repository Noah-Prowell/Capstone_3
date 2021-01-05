from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
from transformers import GPT2Config, GPT2Tokenizer, TFGPT2LMHeadModel, TFGPT2Model
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np
from transformers import pipeline, set_seed
from BPE_token import BPE_token
#read in text
df = pd.read_csv('data/cartman.csv', names = ['Document'], dtype = str)
corpus = [str(row) for row in df['Document']]

"""
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
text = corpus[6]
encoded_input = tokenizer.encode(text, return_tensors='tf')
output = model(encoded_input)
greedy_output = model.generate(encoded_input, max_length=50, top_k=50, top_p=.95, do_sample=True)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
"""
path = 'outfile.txt'
tokenizer = BPE_token()
# train the tokenizer model
tokenizer.bpe_train(path)
# saving the tokenized data in our specified folder 
save_path = 'tokenized_data'
tokenizer.save_tokenizer(save_path)

# loading tokenizer from the saved model path
tokenizer = GPT2Tokenizer.from_pretrained(save_path)