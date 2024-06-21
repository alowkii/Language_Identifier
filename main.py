from train import *

# Load a test document from file
test_document = load_text_file('input/input.txt')

# Predict the language of the test document
predicted_language = identifier.predict(test_document)
print("Predicted language:", predicted_language)