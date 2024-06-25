import pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Load a test document from file
test_document = load_text_file('input/input.txt')

# Predict the language of the test document
predicted_language = model.predict(test_document)
print("Predicted language:", predicted_language)