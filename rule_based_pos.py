import nltk


# Rule-based POS tagging function
def rule_based_pos_tagging(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens


# Example usage
sentence = "The brown fox jumps over the lazy dog."
result = rule_based_pos_tagging(sentence)
print(result)
