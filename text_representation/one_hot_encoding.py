from sklearn.feature_extraction.text import CountVectorizer

# Sample sentences
sentences = ["I love natural language processing.",
             "One hot encoding is simple but inefficient.",
             "NLP is a fascinating field of study."]

# Create an instance of CountVectorizer
vectorizer = CountVectorizer(binary=True)

# Fit and transform the sentences to obtain the one-hot encoding
one_hot_encoding = vectorizer.fit_transform(sentences).toarray()

# Get the feature names (words in the vocabulary)
vocab = vectorizer.get_feature_names_out()

# Display the one-hot encoding and vocabulary
print("One-Hot Encoding:")
print(one_hot_encoding)
print("\nVocabulary:")
print(vocab)

# Output:
# One-Hot Encoding:
# [[0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0]
#  [1 1 0 0 1 1 1 0 0 0 0 0 1 0 1 0]
#  [0 0 1 1 0 0 1 0 0 0 1 1 0 0 0 1]]

# Vocabulary:
# ['but' 'encoding' 'fascinating' 'field' 'hot' 'inefficient' 'is'
#  'language' 'love' 'natural' 'nlp' 'of' 'one' 'processing' 'simple'
#  'study']
