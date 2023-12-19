import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def text_preprocessing(text):
    # Converting to lowercase
    text = text.lower()

    # Removing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Removing non-word and non-whitespace characters
    text = re.sub(r'[^\w\s]', '', text)

    # Removing digits
    text = re.sub(r'\d', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


# Example usage
input_text = "Hello! This is an example text with a URL https://example.com. Let's go to Room 123."
processed_text = text_preprocessing(input_text)
print(processed_text)

# output: ['hello', 'exampl', 'text', 'url', 'let', 'go', 'room']
