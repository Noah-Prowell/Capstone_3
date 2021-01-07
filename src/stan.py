import pandas as pd 
import numpy as np
import os




df = pd.read_csv('data/stan.csv', names = ['Document'], dtype = str)
corpus = [str(row) for row in df['Document']]


with open("stan.txt", "w") as outfile:
    outfile.write("\n".join(corpus))