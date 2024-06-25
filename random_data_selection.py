##Selects 100 random rows from train-data.csv and saves them to test-data.csv
import numpy as np
import random
import pandas as pd

def random_data_selection(size=5000):
    file_path = 'dataset/training-data.csv'

    training_data = pd.read_csv(file_path, encoding='utf-8')

    if training_data.shape[0] < size:
        raise ValueError("The input file does not have at least size rows.")

    # Randomly select size rows
    random_rows = training_data.sample(n=size, axis=0, random_state=np.random.seed(random.randint(0, 10000)))

    # Save the selected rows to a new CSV file
    random_rows.to_csv('dataset/train-subDataSet.csv', index=False)

random_data_selection() 