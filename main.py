# Encoding: utf-8
import re
import numpy as np
import unicodedata
import pandas as pd
from collections import defaultdict


class NaiveBayesLanguageIdentifier:
    def __init__(self, n=2):
        self.n = n
        self.vocab = set()
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.stop_words = set()
    
    def _normalize_text(self, text):
        # Normalize text
        # We can add more normalization methods based on languages
        normalized_text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return normalized_text
    
    def _clean_text(self, text):
        # Remove non-alphabetic characters and convert to lowercase
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
        cleaned_text = cleaned_text.lower()
        return cleaned_text
    
    def _extract_ngrams(self, text):
        ngrams = []
        text = ' ' + text + ' '  # Add padding to get n-grams at the beginning and end of the text
        for i in range(len(text) - self.n + 1):
            ngrams.append(text[i:i+self.n])
        return ngrams
    
    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            normalized_doc = self._normalize_text(doc)
            cleaned_doc = self._clean_text(normalized_doc)
            ngrams = self._extract_ngrams(cleaned_doc)
            self.class_counts[label] += 1
            for ngram in ngrams:
                self.class_word_counts[label][ngram] += 1
                self.vocab.add(ngram)
    
    def _calculate_log_likelihood(self, text, label):
        log_likelihood = 0.0
        ngrams = self._extract_ngrams(text)
        for ngram in ngrams:
            if ngram not in self.stop_words:  # Check if n-gram is not a stop word
                word_count = self.class_word_counts[label].get(ngram, 0) + 1  # Add-one smoothing
                total_words = sum(self.class_word_counts[label].values()) + len(self.vocab)
                log_likelihood += np.log(word_count / total_words)
        return log_likelihood
    
    def predict(self, text):
        normalized_text = self._normalize_text(text)
        cleaned_text = self._clean_text(normalized_text)
        best_label = None
        max_log_prob = float('-inf')
        for label in self.class_counts.keys():
            log_prior = np.log(self.class_counts[label] / sum(self.class_counts.values()))
            log_likelihood = self._calculate_log_likelihood(cleaned_text, label)
            log_posterior = log_prior + log_likelihood
            if log_posterior > max_log_prob:
                max_log_prob = log_posterior
                best_label = label
        return best_label
    
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Example usage:
# Small sample dataset:
import pandas as pd

df = pd.read_csv('Language Detection.csv')

documents = df.iloc[:, 0].tolist()
labels = df.iloc[:, 1].tolist()

identifier = NaiveBayesLanguageIdentifier(n=3)

# Define some stop words for noise filtering
STOP_WORDS = {}
languages = set(labels)
print(languages)
for language in languages:
    try:
        exec(open(f'stopwords/{language}.py', encoding='utf8').read(),)
        stop_words = {
            language : STOP_WORDS
        }
    except FileNotFoundError as e:
        print(f'{language} not found')

# Union of stop words for all languages
identifier.stop_words = set().union(*stop_words.values())

identifier.train(documents, labels)

# Load a test document from file
test_document = load_text_file('input.txt')

# Predict the language of the test document
predicted_language = identifier.predict(test_document)
print("Predicted language:", predicted_language)
