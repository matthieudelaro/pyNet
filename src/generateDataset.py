"""This script transforms MNIST dataset provided
available at http://yann.lecun.com/exdb/mnist/
into a pickled version optimised for the neural
network."""

import numpy as np
import pickle
from Dataset import OriginalMNISTDataset, OptimizedDataset


def generateOptimizedDataSet():
    origin = OriginalMNISTDataset()
    print("Reshaping samples...")
    shape = origin.trainX.shape
    origin.trainX = origin.trainX.reshape(shape[0], shape[1] * shape[2])
    shape = origin.testX.shape
    origin.testX = origin.testX.reshape(shape[0], shape[1] * shape[2])
    print("Samples reshaped.")
    print()

    print("Splitting dataset into train/valid/tests, divide pixel values by 255...")
    ratio = 0.9  # percentageOfTrainingSetForValidation

    train = [()] * int(len(origin.trainX)*ratio)
    valid = [()] * int(len(origin.trainX)*(1-ratio) + 1)
    tests = [()] * len(origin.testX)

    for i in range(len(train)):
        train[i] = (np.array([origin.trainX[i]]).T.astype(dtype="double") / 255, labelAsArray(origin.trainY[i]), origin.trainY[i])

    for j in range(len(train), len(train) + len(valid)):
        i = j - len(train)
        valid[i] = (np.array([origin.trainX[j]]).T.astype(dtype="double") / 255, labelAsArray(origin.trainY[j]), origin.trainY[j])

    for i in range(len(tests)):
        tests[i] = (np.array([origin.testX[i]]).T.astype(dtype="double") / 255, labelAsArray(origin.testY[i]), origin.testY[i])
    print("Done:")
    print("\ttraining set:", len(train), "samples")
    print("\tvalidation set:", len(valid), "samples")
    print("\ttesting set:", len(tests), "samples")

    print("\nSaving entire dataset...")
    optim = OptimizedDataset()
    optim.train = train
    optim.valid = valid
    optim.tests = tests

    f_write = open('../data/pickledMNIST/data.pkl', 'bw')
    pickle.dump(optim, f_write, protocol=4, fix_imports=False)
    f_write.close()
    print("Done")

    print("\nSaving smaller dataset for debug...")
    optim = OptimizedDataset()
    optim.train = train[:180]
    optim.valid = valid[:20]
    optim.tests = tests
    f_write = open('../data/pickledSmallMNIST/data.pkl', 'bw')
    pickle.dump(optim, f_write, protocol=4, fix_imports=False)
    f_write.close()
    print("Done")

    print("\nSaving medium dataset for debug...")
    optim = OptimizedDataset()
    optim.train = train[:1800]
    optim.valid = valid[:200]
    optim.tests = tests
    f_write = open('../data/pickledMediumMNIST/data.pkl', 'bw')
    pickle.dump(optim, f_write, protocol=4, fix_imports=False)
    f_write.close()
    print("Done")

    print("\nSample of the dataset:\n", train[4])


def labelAsArray(label):
    array = np.zeros((10, 1))
    array[label] = 1
    return array


if __name__ == '__main__':
    generateOptimizedDataSet()
