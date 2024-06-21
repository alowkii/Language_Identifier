##Selects 1000 random rows from train-data.csv and saves them to test-data.csv
import os
import pandas as pd

file_path = 'training-data.csv'

# Check if the file exists and print its content
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        print("File Content:")
        print(content[:500])  # Print the first 500 characters of the file content
else:
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Read the original CSV file
training_data = pd.read_csv(file_path)

# Ensure there are at least 1000 rows
if training_data.shape[0] < 1000:
    raise ValueError("The input file does not have at least 1000 rows.")

# Randomly select 1000 rows
random_rows = training_data.sample(n=1000, axis=0, random_state=1)

# Save the selected rows to a new CSV file
random_rows.to_csv('test-data.csv', index=False)

# Get the indices of the selected rows
random_indices = random_rows.index

# Remove the selected rows from the original DataFrame
remaining_data = training_data.drop(random_indices)

# Save the remaining data back to the original CSV file
remaining_data.to_csv(file_path, index=False)

print("Random 1000 rows have been saved to test-data.csv and removed from training-data.csv")
