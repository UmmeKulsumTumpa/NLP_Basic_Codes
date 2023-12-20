import nltk
from nltk import word_tokenize
from nltk.util import ngrams

# Sample text
text = "The dog in the house"

# Tokenize the text into words
tokens = word_tokenize(text)


# Function to generate n-grams
def generate_ngrams(input_list, n):
    return list(ngrams(input_list, n))


# Uni-grams
uni_grams = generate_ngrams(tokens, 1)
print("Uni-gram:", uni_grams)

# Bi-grams
bi_grams = generate_ngrams(tokens, 2)
print("Bi-gram:", bi_grams)

# Tri-grams
tri_grams = generate_ngrams(tokens, 3)
print("Tri-gram:", tri_grams)


# Output:
# Uni-gram: [('The',), ('dog',), ('in',), ('the',), ('house',)]
# Bi-gram: [('The', 'dog'), ('dog', 'in'), ('in', 'the'), ('the', 'house')]
# Tri-gram: [('The', 'dog', 'in'), ('dog', 'in', 'the'), ('in', 'the', 'house')]
