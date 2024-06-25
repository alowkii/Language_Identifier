# Language Identifier using Naive Bayes Classifier

## Overview

This project implements a Language Identifier that uses a Naive Bayes Classifier to predict the language of a given text. The classifier is trained on a dataset containing text samples in multiple languages, achieving an accuracy of 89.78%.

## Features

- High Accuracy: Achieves an accuracy of 89.78% on the test dataset.
- Multilingual Support: Can identify a variety of languages.
- Naive Bayes Classifier: Utilizes the simplicity and efficiency of the Naive Bayes algorithm.
- Image to Text: Supports extracting text from images and identifying the language.
- Easy Integration: Designed to be easily integrated into other projects or used as a standalone tool.

## Installation

To install the necessary dependencies and set up the project, follow these steps:

### Clone the Repository

```bash
git clone https://github.com/alowkii/language-identifier.git
cd language-identifier
```

### Activate Existing Environment:

The repository already includes a virtual environment called env. Activate this environment:

```bash
source env/bin/activate # On Windows, use `env\Scripts\activate
```

### Install Dependencies:

Ensure all dependencies are installed in the activated environment.

```bash
pip install -r requirements.txt
```

### [Optional] Tesseract Installation:

Install Tesseract OCR for text extraction from images. This is totally optional and supports only a few languages. Only if you are going to extract data from the **input.txt** file.

#### Make sure to change the directory path in image_to_text.py

Download and install from Tesseract OCR.
Set the tesseract_cmd path in image_to_text.py if needed.

## Usage/Examples

### Training the Model

To train the model, run the following script. This will train the model and save it to a file named model.pkl.

```bash
python __init__.py
```

### Testing the Model

To test the model's performance on a test dataset:

```bash
python test.py
```

### Predicting Language

You can use the pre-trained model to predict the language of a text file:

```bash
python identify_language.py
```

### [Optional] Extracting Text from an Image

To extract text from an image and identify the language:

```bash
python image_to_text.py
```

### Run the full program

You can use the CLI to identify the language of a given text file:

```bash
python main.py
```

## Project Structure

#### language_identifier/

**\_\_init\_\_.py:** Initializes the module and trains the model.

**model.py:** Contains the Naive Bayes Classifier model.

**test.py:** Script to test the model.

**identify_language.py:** Script to identify language from a text file.

**dataset/:** Directory containing training and testing data.

**input/input.txt:** Put the required text inside or simply run the image with image_to_text.py

**requirements.txt:** Lists the Python dependencies.

**image_to_text.py:** Extracts text from images and predicts the language and stores in input/input.txt.

**main.py:** Loads the model and predicts language for a given text file or image.

## README: Project documentation.

**How It Works**

**Data Preprocessing:** The text data is cleaned and tokenized to create a suitable input format for the model and removes any stopwords in that language.

**Feature Extraction:** Extracts features such as character n-grams or word frequency distributions.

**Model Training:** The Naive Bayes Classifier is trained on the extracted features from the training dataset.

**Prediction:** For a given input text, the model calculates the probability of the text belonging to each language and returns the language with the highest probability.

## Performance

The Naive Bayes Classifier achieves an accuracy of 89.78% on the test dataset. This performance can vary based on the quality and diversity of the training data.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate tests.

## License

- [MIT](https://choosealicense.com/licenses/mit/)

- [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the project maintainer at @alowkii.
