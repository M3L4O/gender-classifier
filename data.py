import random
from math import floor

random.seed(1)

def get_features(word: str):
    return {"last_letter": word[-1], "first_letter": word[0], "word_len": len(word)}


def get_fset(
    dataset: list, training_rate: float = 0.8, shuffle: bool = False, k: int = 0
):
    dataset_size = len(dataset)
    fset = [(get_features(name), label) for name, label in dataset]
    if shuffle:
        random.shuffle(fset)
    if k != 0:
        fold_size = dataset_size // k
        training_sets = []
        test_sets = []
        for i in range(k):
            test_sets.append(fset[i * fold_size : i * fold_size + fold_size])
            initial_training_set = fset[: i * fold_size] if i > 0 else []
            final_training_set = fset[i * fold_size + fold_size :] if i != k - 1 else []
            initial_training_set.extend(final_training_set)
            training_sets.append(initial_training_set)

        return training_sets, test_sets
    else:
        training_size = floor(dataset_size * training_rate)

        return fset[:training_size], fset[training_size:]


def get_fsets(
    dataset: tuple,
    labels: list,
    training_rate: float = 0.8,
    shuffle: bool = False,
    k: int = 0,
):
    datasets = [list(filter(lambda x: x[1] == label, dataset)) for label in labels]

    training_sets, test_sets = [0, 0], [0, 0]

    training_sets[0], test_sets[0] = get_fset(
        datasets[0], training_rate=training_rate, shuffle=shuffle, k=k
    )
    training_sets[1], test_sets[1] = get_fset(
        datasets[1], training_rate=training_rate, shuffle=shuffle, k=k
    )

    if k == 0:

        training_set = training_sets[0] + training_sets[1]
        test_set = test_sets[0] + test_sets[1]

        return training_set, test_set

    training_set = [
        sets[0] + sets[1] for sets in zip(training_sets[0], training_sets[1])
    ]
    test_set = [sets[0] + sets[1] for sets in zip(test_sets[0], test_sets[1])]
    return training_set, test_set
