import random
from math import floor

import nltk
from nltk import DecisionTreeClassifier, NaiveBayesClassifier
from nltk.corpus import names

males = [(name, 0) for name in names.words("male.txt")]
females = [(name, 1) for name in names.words("female.txt")]


def get_features(word: str):
    return {"last_letter": word[-1], "first_letter": word[0], "word_len": len(word)}


def get_fsets(dataset: list, training_rate: float = 0.8, shuffle: bool = False, k: int = 1):
    training_size = floor(len(dataset) * training_rate)
    fset = [(get_features(name), label) for name, label in dataset]
    if shuffle:
        random.shuffle(fset)
    return fset[:training_size], fset[training_size:]
    
training_male_set, test_male_set = get_fsets(males)
training_female_set, test_female_set = get_fsets(females)
training_set, test_set = training_male_set + training_female_set, test_male_set + test_female_set

model = NaiveBayesClassifier.train(training_set)

print(nltk.classify.accuracy(model, test_set))
