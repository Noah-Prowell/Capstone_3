import pandas as pd 
import numpy as np
import os




df = pd.read_csv('data/kyle.csv', names = ['Document'], dtype = str)
corpus = [str(row) for row in df['Document']]


with open("kyle.txt", "w") as outfile:
    outfile.write("\n".join(corpus))