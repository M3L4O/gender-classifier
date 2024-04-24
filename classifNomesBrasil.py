import nltk
from data import get_fsets, get_features
import csv

nomesRot = []

with open('nomes.csv','r',encoding = 'utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        names = row['alternative_names'].split('|')
        label = row['classification']
        
        for name in names:
            if name:
                nomesRot.append((name, label))

train, test = get_fsets(nomesRot, shuffle=True)

classifNomesBr = nltk.NaiveBayesClassifier.train(train)

#alguns testes
print(classifNomesBr.classify(get_features('Cleonice')))
print(classifNomesBr.classify(get_features('Ariel')))
print(classifNomesBr.classify(get_features('Bernardo')))
print(classifNomesBr.classify(get_features('Kalina')))

#acurácia do modelo
print(nltk.classify.accuracy(classifNomesBr, test))

classifNomesBr.show_most_informative_features(5)