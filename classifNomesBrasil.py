import csv

import nltk

from data import get_features, get_fsets

nomesRot = []

with open("nomes.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        names = row["alternative_names"].split("|")
        label = row["classification"]

        for name in names:
            if name:
                nomesRot.append((name, label))

train, test = get_fsets(nomesRot, ["M", "F"], shuffle=True)

print(
    len(list(filter(lambda x: x[1] == "M", train))),
    len(list(filter(lambda x: x[1] == "F", train))),
    len(train),
)
print(
    len(list(filter(lambda x: x[1] == "M", test))),
    len(list(filter(lambda x: x[1] == "F", test))),
    len(test),
)

classifNomesBr = nltk.NaiveBayesClassifier.train(train)

# alguns testes
print(classifNomesBr.classify(get_features("Cleonice")))
print(classifNomesBr.classify(get_features("Ariel")))
print(classifNomesBr.classify(get_features("Bernardo")))
print(classifNomesBr.classify(get_features("Kalina")))

# acurácia do modelo
print(nltk.classify.accuracy(classifNomesBr, test))

classifNomesBr.show_most_informative_features(5)


train_sets, test_sets = get_fsets(nomesRot, ["M", "F"], shuffle=True, k=5)

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
        len(list(filter(lambda x: x[1] == "M", train_set))),
        len(list(filter(lambda x: x[1] == "F", train_set))),
        len(train_set),
    )
    print(
        len(list(filter(lambda x: x[1] == "M", test_set))),
        len(list(filter(lambda x: x[1] == "F", test_set))),
        len(test_set),
    )
    model.show_most_informative_features(5)
    print("\n\n")

mean = sum(accuracies) / len(accuracies)
print(f"Accuracy mean: {mean}")
