# Usage:
# Initialize NaiveBayesLanguageIdentifier
import random_data_selection
import pickle
from test import *
from model import *

model = NaiveBayesLanguageIdentifier(n=3)
epochs = 10

for i in range(epochs):
    print("Epoch: ",i+1)
    random_data_selection.random_data_selection(size=8000)
    text_data, labels = model.load_data('dataset/train-subDataSet.csv')
    model.train(text_data, labels)
    test(model)

print("\nTraining completed.\n")
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)