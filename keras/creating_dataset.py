import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler



def make_train_dataset():
    train_labels = []
    train_samples = []

    for i in range(50):
        random_younger = randint(13, 64)
        train_samples.append(random_younger)
        train_labels.append(1)

        random_older = randint(65,100)
        train_samples.append(random_older)
        train_labels.append(0)

    for i in range(1000):
        random_younger = randint(13, 64)
        train_samples.append(random_younger)
        train_labels.append(0)

        random_older = randint(65,100)
        train_samples.append(random_older)
        train_labels.append(1)

    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)
    train_labels, train_samples = shuffle(train_labels, train_samples)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

    return scaled_train_samples, train_labels


def make_validation_dataset():
    eval_labels = []
    eval_samples = []

    for i in range(5):
        random_younger = randint(13, 64)
        eval_samples.append(random_younger)
        eval_labels.append(1)

        random_older = randint(65,100)
        eval_samples.append(random_older)
        eval_labels.append(0)

    for i in range(100):
        random_younger = randint(13, 64)
        eval_samples.append(random_younger)
        eval_labels.append(0)

        random_older = randint(65,100)
        eval_samples.append(random_older)
        eval_labels.append(1)

    eval_labels = np.array(eval_labels)
    eval_samples = np.array(eval_samples)
    eval_labels, eval_samples = shuffle(eval_labels, eval_samples)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_eval_samples = scaler.fit_transform(eval_samples.reshape(-1, 1)) # 
    eval_labels = eval_labels.reshape(-1, 1)
    return (scaled_eval_samples, eval_labels)


def make_test_dataset():
    test_labels = []
    test_samples = []

    for i in range(10):
        random_younger = randint(13, 64)
        test_samples.append(random_younger)
        test_labels.append(1)

        random_older = randint(65,100)
        test_samples.append(random_older)
        test_labels.append(0)

    for i in range(200):
        random_younger = randint(13, 64)
        test_samples.append(random_younger)
        test_labels.append(0)

        random_older = randint(65,100)
        test_samples.append(random_older)
        test_labels.append(1)

    test_labels = np.array(test_labels)
    test_samples = np.array(test_samples)
    test_labels, test_samples = shuffle(test_labels, test_samples)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1)) # 
    
    return scaled_test_samples, test_labels