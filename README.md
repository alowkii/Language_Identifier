#Language Identifier using Naive Bayes Classifier

Overview
This project implements a Language Identifier that uses a Naive Bayes Classifier to predict the language of a given text. The classifier is trained on a dataset containing text samples in multiple languages, achieving an accuracy of 89.78%.

Features
High Accuracy: Achieves an accuracy of 89.78% on the test dataset.
Multilingual Support: Can identify a variety of languages.
Naive Bayes Classifier: Utilizes the simplicity and efficiency of the Naive Bayes algorithm.
Image to Text: Supports extracting text from images and identifying the language.
Easy Integration: Designed to be easily integrated into other projects or used as a standalone tool.
Installation
To install the necessary dependencies and set up the project, follow these steps:

Clone the Repository:

bash
Copy code
git clone https://github.com/alowkii/language-identifier.git
cd language-identifier
Activate Existing Environment:
The repository already includes a virtual environment called env. Activate this environment:

bash
Copy code
source env/bin/activate # On Windows, use `env\Scripts\activate`
Install Dependencies:
Ensure all dependencies are installed in the activated environment.

bash
Copy code
pip install -r requirements.txt
Tesseract Installation:
Install Tesseract OCR for text extraction from images.

Download and install from Tesseract OCR.
Set the tesseract_cmd path in image_to_text.py if needed.
Usage
Training the Model
To train the model, run the following script. This will train the model and save it to a file named model.pkl.

bash
Copy code
python train_and_test.py
Testing the Model
To test the model's performance on a test dataset:

bash
Copy code
python test.py
Predicting Language
You can use the pre-trained model to predict the language of a text file:

bash
Copy code
python main.py
Extracting Text from an Image
To extract text from an image and identify the language:

bash
Copy code
python image_to_text.py
Command Line Interface (CLI)
You can use the CLI to identify the language of a given text file:

bash
Copy code
python main.py --file input/input.txt
Project Structure
language_identifier/
**init**.py: Initializes the module.
model.py: Contains the Naive Bayes Classifier model.
train.py: Script to train the model.
test.py: Script to test the model.
identify_language.py: CLI script to identify language from a text file.
dataset/: Directory containing training and testing data.
input/input.txt: Put the required text inside or simply run the image with image_to_text.py
requirements.txt: Lists the Python dependencies.
train_and_test.py: Combines training and testing procedures.
image_to_text.py: Extracts text from images and predicts the language.
main.py: Loads the model and predicts language for a given text file.
README.md: Project documentation.
How It Works
Data Preprocessing: The text data is cleaned and tokenized to create a suitable input format for the model.
Feature Extraction: Extracts features such as character n-grams or word frequency distributions.
Model Training: The Naive Bayes Classifier is trained on the extracted features from the training dataset.
Prediction: For a given input text, the model calculates the probability of the text belonging to each language and returns the language with the highest probability.
Performance
The Naive Bayes Classifier achieves an accuracy of 89.78% on the test dataset. This performance can vary based on the quality and diversity of the training data.

Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate tests.

License
This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.

Contact
For any questions or issues, please open an issue on the GitHub repository or contact the project maintainer at @alowkii.
