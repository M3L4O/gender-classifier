import nltk
from nltk.corpus import names

from data import get_features, get_fsets

nomesRot = [(name, "masc") for name in names.words("male.txt")] + [
    (name, "fem") for name in names.words("female.txt")
]

train, test = get_fsets(nomesRot, ["masc", "fem"], shuffle=True)

print(
    len(list(filter(lambda x: x[1] == "masc", train))),
    len(list(filter(lambda x: x[1] == "fem", train))),
    len(train),
)
print(
    len(list(filter(lambda x: x[1] == "masc", test))),
    len(list(filter(lambda x: x[1] == "fem", test))),
    len(test),
)


classif1 = nltk.NaiveBayesClassifier.train(train)

print(classif1.classify(get_features("Neo")))
print(classif1.classify(get_features("Trinity")))
print(classif1.classify(get_features("Simon")))

print(nltk.classify.accuracy(classif1, test))

classif1.show_most_informative_features(10)


train_sets, test_sets = get_fsets(nomesRot, ["masc", "fem"], shuffle=True, k=5)

accuracies = []
i = 0
for train_set, test_set in zip(train_sets, test_sets):
    i += 1
    model = nltk.NaiveBayesClassifier.train(train_set)
    acc = nltk.classify.accuracy(model, test_set)
    accuracies.append(acc)
    print(f"Model {i}:\nAccuracy:{acc}")
    print("masc fem tota")
    print(
        len(list(filter(lambda x: x[1] == "masc", train_set))),
        len(list(filter(lambda x: x[1] == "fem", train_set))),
        len(train_set),
    )
    print(
        len(list(filter(lambda x: x[1] == "masc", test_set))),
        len(list(filter(lambda x: x[1] == "fem", test_set))),
        len(test_set),
    )
    model.show_most_informative_features(5)
    print("\n\n")

mean = sum(accuracies) / len(accuracies)
print(f"Accuracy mean: {mean}")
