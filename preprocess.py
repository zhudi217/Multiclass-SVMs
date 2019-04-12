import numpy as np
import scipy.io

def load_data(file_path='MNIST_data.mat'):
    data = scipy.io.loadmat(file_path)

    X_train = np.array(data['train_samples'])
    Y_train = np.array(data['train_samples_labels']).reshape((data['train_samples_labels'].shape[0],))
    X_test = np.array(data['test_samples'])
    Y_test = np.array(data['test_samples_labels']).reshape((data['test_samples_labels'].shape[0],))

    return X_train, Y_train, X_test, Y_test


def data_clustering(X, Y):
    X_clustered = [[], [], [], [], [], [], [], [], [], []]
    for i in range(X.shape[0]):
        X_clustered[Y[i]].append(X[i])
    X_clustered_np = [np.array(X_clustered[i]) for i in range(len(X_clustered))]
    return X_clustered_np


def one_and_theRest(X_clustered_np, number):
    X_negative = []
    len_theRest = 0

    for i in range(len(X_clustered_np)):
        if i == number:
            Y_positive = np.ones(len(X_clustered_np[i]))
            X_positive = np.array(X_clustered_np[i])
        else:
            X_negative.extend(X_clustered_np[i])
            len_theRest += len(X_clustered_np[i])

    X_negative = np.vstack((X_negative))
    Y_negative = np.ones(len_theRest) * -1

    return X_positive, X_negative, Y_positive, Y_negative

def one_versus_one(X_clustered_np, pair):
    X_positive = X_clustered_np[pair[0]]
    X_negative = X_clustered_np[pair[1]]
    Y_positive = np.ones(len(X_positive))
    Y_negative = -np.ones(len(X_negative))
    return X_positive, X_negative, Y_positive, Y_negative

