import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Assuming you have a DataFrame 'df' with a column 'text'
# Replace this with your actual DataFrame and column names

# Example DataFrame
data = {'text': ["It is one of the most used text vectorization techniques.",
                 "Bag of words is a little bit similar to one-hot encoding.",
                 "It is mostly used in text classification tasks."]}

df = pd.DataFrame(data)

# Create a CountVectorizer instance
cv = CountVectorizer()

# Fit and transform the 'text' column
bow = cv.fit_transform(df['text'])

# Display the vocabulary and the bag-of-words matrix
print("Vocabulary:")
print(cv.get_feature_names_out())

print("\nBag-of-Words Matrix:")
print(bow.toarray())

# Output: 
# Vocabulary:
# ['bag' 'bit' 'classification' 'encoding' 'hot' 'in' 'is' 'it' 'little'
#  'most' 'mostly' 'of' 'one' 'similar' 'tasks' 'techniques' 'text' 'the'
#  'to' 'used' 'vectorization' 'words']

# Bag-of-Words Matrix:
# [[0 0 0 0 0 0 1 1 0 1 0 1 1 0 0 1 1 1 0 1 1 0]
#  [1 1 0 1 1 0 1 0 1 0 0 1 1 1 0 0 0 0 1 0 0 1]
#  [0 0 1 0 0 1 1 1 0 0 1 0 0 0 1 0 1 0 0 1 0 0]]
