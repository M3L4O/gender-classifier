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

print(len(list(filter(lambda x: x[1] == "M", train))),len(list(filter(lambda x: x[1] == "F", train))), len(train))
print(len(list(filter(lambda x: x[1] == "M", test))), len(list(filter(lambda x: x[1] == "F", test))), len(test))

classifNomesBr = nltk.NaiveBayesClassifier.train(train)

# alguns testes
print(classifNomesBr.classify(get_features("Cleonice")))
print(classifNomesBr.classify(get_features("Ariel")))
print(classifNomesBr.classify(get_features("Bernardo")))
print(classifNomesBr.classify(get_features("Kalina")))

# acurácia do modelo
print(nltk.classify.accuracy(classifNomesBr, test))

classifNomesBr.show_most_informative_features(5)
