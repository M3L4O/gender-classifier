import nltk
from nltk.corpus import names
from data import get_fsets, get_features

nomesRot = (
    [(name,'masc') for name in names.words('male.txt')] + [(name,'fem') for name in names.words('female.txt')]
)

train, test = get_fsets(nomesRot, ["masc", "fem"],shuffle=True)

print(len(list(filter(lambda x: x[1] == "masc", train))),len(list(filter(lambda x: x[1] == "fem", train))), len(train))
print(len(list(filter(lambda x: x[1] == "masc", test))), len(list(filter(lambda x: x[1] == "fem", test))), len(test))


classif1 = nltk.NaiveBayesClassifier.train(train)

print(classif1.classify(get_features('Neo')))
print(classif1.classify(get_features('Trinity')))
print(classif1.classify(get_features('Simon')))

print(nltk.classify.accuracy(classif1, test))

classif1.show_most_informative_features(10)