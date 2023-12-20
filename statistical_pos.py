import nltk
import numpy as np

# Ensure that np.seterr is used to handle overflow issues
np.seterr(all='warn')

from nltk.corpus import treebank
from nltk.tag import hmm


# Statistical POS tagging function
def statistical_pos_tagging(sentence):
    training_data = treebank.tagged_sents()
    trainer = hmm.HiddenMarkovModelTrainer()
    pos_tagger = trainer.train(training_data, estimator=lambda fd, bins: nltk.LidstoneProbDist(fd, 0.1, bins))
    tagged_tokens = pos_tagger.tag(nltk.word_tokenize(sentence))
    return tagged_tokens


# Example usage
sentence = "The brown fox jumps over the lazy dog."
result = statistical_pos_tagging(sentence)
print(result)

# Output: [('The', 'DT'), ('brown', '$'), ('fox', 'CD'), ('jumps', 'NNS'), ('over', 'IN'),
# ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NNS'), ('.', '.')]
