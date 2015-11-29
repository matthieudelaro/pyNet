import numpy as np
import Dataset
from Dataset import OptimizedDataset, OptimizedDatabase
import unittest
import Loss
import time
import random
import Activation
import math


class Net:
    def __init__(self):
        # self.sizes = np.array([784, 784, 784, 10])  # size of each layer
        self.sizes = np.array([784, 100, 30, 10])  # size of each layer
        # self.sizes = np.array([784, 100, 10])  # size of each layer
        self.L = len(self.sizes)  # number of layers
        self.W = [0.01 * np.random.randn(sizeCurrent, sizePrevious).astype(dtype="double")
             for sizePrevious, sizeCurrent
             in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [0.01 * np.random.randn(sizeCurrent, 1).astype(dtype="double")
             for sizeCurrent in self.sizes[1:]]
        self.activationF = Activation.LeakyReLU  # determine activation function
        self.lossF = Loss.Softmax
        self.dropOut = 1  # [0-1]. 1 : no dropout
        self.epochsStats = []
        self.learningRateStep = 5

    def guessLabel(self, sample):
        """Return the class evaluated by the network for the
        given sample."""
        a = sample
        for W, b in zip(self.W, self.b):
            z = W@a + b  # @ is the dot product (python3.5)
            a = self.activationF.f(z)
        return np.argmax(a)

    def evaluate(self, dataset):
        """Returns the success rate of the network over the test set
        of the given dataset."""
        success = 0
        for sample, labelVector, label in dataset.tests:
            if self.guessLabel(sample) == label:
                success += 1
        return success / len(dataset.tests)

    def forwardBackward(self, sample, label, ass, zss, dWs, dbs):
        # forward pass: for each layer (except for the first layer
        # which is just input data)
        # compute z and activation a
        a = sample  # activation of first layer: input sample

        if self.dropOut < 1:
            dropMasks = [(np.random.rand(*aa.shape) < self.dropOut)/self.dropOut for aa in ass]

        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = W@a + b
            a = self.activationF.f(z)

            if self.dropOut < 1:
                a *= dropMasks[l]
            ass[l] = a
            zss[l] = z

        # compute loss and loss gradient
        loss = self.lossF.f(a, label)
        dError = self.lossF.fprime(ass[-1], label) * self.activationF.fprime(zss[-1])

        # propagate the error back
        for l in reversed(range(len(self.W))):
            # print()
            # retrieve required values
            if l < len(self.W)-1:
                W_lPlus1 = self.W[l+1]

                dError_lPlus1 = dError
                # print("Reading zss at index:", l)
                z = zss[l]

                # compute the error
                dError = (W_lPlus1.T @ dError_lPlus1) * self.activationF.fprime(z)

            if l != 0:
                # print("Reading ass as index:", l-1)
                a_lMinus1 = ass[l-1]
            else:
                # print("Reading sample because index", l-1)
                a_lMinus1 = sample
            # a_lMinus1 = ass[l]

            if self.dropOut < 1:
                # print(dError.shape, dropMasks[l].shape)
                dError *= dropMasks[l]

            # print("dW <= a_lMinus1 dError :", dWs[l].shape, "<=", a_lMinus1.shape, dError.shape)
            # dW = a_lMinus1 * dError
            dWs[l] += dError@a_lMinus1.T
            dbs[l] += dError
        return loss

    def batchForwardBackward(self, batch, learningRate, ass, zss, dWs, dbs):
        for dW in dWs:
            dW.fill(0)
        for db in dbs:
            db.fill(0)

        # forwardBackward on each sample, and sum up modifications suggested by the gradients
        iterations = 0
        sumLoss = 0
        for sample, label, labelScalar in batch:
            iterations += 1
            sumLoss += self.forwardBackward(sample, label, ass, zss, dWs, dbs)
        meanLoss = sumLoss/iterations

        # modify weigths and biases according to previous backpropagations
        for W, dW in zip(self.W, dWs):
            W -= dW * (learningRate / iterations)
        for b, db in zip(self.b, dbs):
            b -= db * (learningRate / iterations)

        return meanLoss

    def train(self, dataset, epochs, batchSize, learningRate):
        zss = [np.empty((layerSize, 1)).astype(dtype="double")  # for each layer, stores the value, ie z = W@a + b
               for layerSize in self.sizes[1:]]
        ass = [np.empty((layerSize, 1)).astype(dtype="double")  # for each layer, stores the activation, ie a = f(z)
               for layerSize in self.sizes[1:]]
        dWs = [np.zeros_like(W) for W in self.W]  # for each layer, stores dW, ie how much weights should be modified
        dbs = [np.zeros_like(b) for b in self.b]  # for each layer, stores db, ie how much biases should be modified

        print("\nTraining with configuration:")
        print("\tSize of layers:", self.sizes)
        print("\tActivation function:", self.activationF.name())
        print("\tLoss function:", self.lossF.name())
        print("\tLearning rate:", learningRate)
        print("\tLearning rate step:", self.learningRateStep)
        print("\tDropout percentage:", self.dropOut, "(From 0 to 1. 1 means 'no dropout')")
        print("\tBatch size:", batchSize)
        print("\tEpochs:", epochs)
        print("\tTraining size:", len(dataset.train))
        print()

        for epoch in range(1, epochs + 1):
            epochBeginTime = time.clock()
            print()
            print("Beginning of epoch", epoch, ". Time:", "{0:.2f}".format(epochBeginTime/60), "min")
            random.shuffle(dataset.train)
            batchBeginning = 0
            batchEnd = min(len(dataset.train), batchBeginning + batchSize)

            iterations = 0
            sumLoss = 0
            while (batchEnd - batchBeginning) >= 1:
                # print()
                # print("Beginning of batch ", iterations, "/", epoch, ". Timer:", time.clock())
                # print("learning rate:", learningRate)
                # print("Test success rate:", self.evaluate(dataset))
                iterations += 1
                batchLoss = self.batchForwardBackward(
                    dataset.train[batchBeginning:batchEnd],
                    learningRate,
                    ass, zss, dWs, dbs
                )
                sumLoss += batchLoss

                if iterations % 100 == 0:
                    print("Loss of batch", epoch, "-", iterations, ":", batchLoss)

                batchBeginning = batchEnd
                batchEnd = min(len(dataset.train), batchBeginning + batchSize)

            meanLoss = sumLoss / iterations
            epochEndTime = time.clock()
            epochDuration = epochEndTime - epochBeginTime
            print("End of epoch ", epoch, ". Time:", "{0:.2f}".format(epochEndTime/60), "min. Duration:", "{0:.2f}".format(epochDuration), "seconds")
            print("Mean loss :", meanLoss)
            print("learning rate:", learningRate)
            successRate = self.evaluate(dataset)
            print("Test success rate:", successRate)
            self.epochsStats.append((epochBeginTime, epochDuration, epochEndTime, meanLoss, learningRate, successRate))
            # learningRate *= 0.97
            learningRate *= 0.97**math.floor(epoch/self.learningRateStep)


class TestsSmaller(unittest.TestCase):
    def setUp(self):
        # self.data = Dataset.SmallerDataset()
        self.data = Dataset.loadSmallPickledData()
        self.net = Net()

    def test_evaluateRandom(self):
        res = self.net.guessLabel(self.data.trainX[0])
        print(res)
        pass

    def test_backprop(self):
        ass = [np.empty((layerSize, 1)).astype(dtype="double")
               for layerSize in self.net.sizes[1:]]
        zss = [np.empty((layerSize, 1)).astype(dtype="double")
               for layerSize in self.net.sizes[1:]]
        dWs = [np.zeros_like(W) for W in self.net.W]  # for each layer, stores dW, ie how much weights should be modified
        dbs = [np.zeros_like(b) for b in self.net.b]  # for each layer, stores db, ie how much biases should be modified

        self.net.forwardBackward(self.data.train[0][0], self.data.train[0][1], ass, zss, dWs, dbs)

    def test_batchBackprop(self):
        batch1 = 90
        batch2 = 90
        self.net.batchForwardBackward(self.data.trainX[:batch1], self.data.trainY[:batch1], 0.01)
        self.net.batchForwardBackward(self.data.trainX[batch1:batch1+batch2], self.data.trainY[batch1:batch1+batch2], 0.01)

    def test_train(self):
        self.net.train(self.data, 30, 50, 0.3)


def runSmall():
    data = Dataset.loadSmallPickledData()
    net = Net()
    net.train(data, 30, 50, 0.1)


def runMedium():
    data = Dataset.loadMediumPickledData()
    net = Net()
    net.train(data, 30, 100, 0.1)


def runBig():
    data = Dataset.loadPickledData()
    net = Net()
    net.train(data, 30, 100, 0.1)


if __name__ == '__main__':
    # unittest.main()
    # runSmall()
    # runMedium()
    runBig()
