import numpy as np
import unicodedata
import pandas as pd
from collections import defaultdict
import seaborn as sns
from train import *

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

df = pd.read_csv('dataset/test-data.csv', encoding='utf-8')
text_data = df['text'].tolist()
labels = df['language'].tolist()

count = 0
for i in range(len(text_data)):
    predicted_language = identifier.predict(text_data[i])
    if(predicted_language.lower() == labels[i].lower()):
        count += 1
print(count)

