import numpy as np
import Dataset
from Dataset import OptimizedDataset, OptimizedDatabase
import unittest
import Loss
import time
import random
import Activation
import math
from FullyConnectedLayer import FCLayer


class DeNet:
    """DeNet is similar to LoopyNet. It uses the same algorithms,
    but each layer hold its own data and achieves
    forward/backward propagation/update by itself. DeNet has been designed
    to minimize data allocations, and to be parallelizable."""

    def __init__(self):
        self.lossF = Loss.Softmax
        self.dropOut = 1  # [0-1]. 1 : no dropout
        self.epochsStats = []
        self.learningRateStep = 5
        self.activationF = Activation.LeakyReLU

        self.sizes = np.array([784, 100, 30, 10])  # size of each layer
        self.L = len(self.sizes)  # number of layers
        self.layers = [None]
        for sizePrevious, sizeCurrent in zip(self.sizes[:-1], self.sizes[1:]):
            self.layers.append(FCLayer(sizePrevious,
                                       sizeCurrent,
                                       self.activationF,
                                       self.dropOut))

    def guessLabel(self, sample):
        """Return the class evaluated by the network for the
        given sample."""
        a = sample
        for layer in self.layers[1:]:
            a = layer.fastForward(a)
        return np.argmax(a)

    def evaluate(self, dataset):
        """Returns the success rate of the network over the test set
        of the given dataset."""
        success = 0
        for sample, labelVector, label in dataset.tests:
            if self.guessLabel(sample) == label:
                success += 1
        return success / len(dataset.tests)

    def forwardBackward(self, sample, label, indexInBatch, ass, bundles):
        """Performs a forward pass and a backward pass over the network with
        given sample. Returns the loss relative to the given sample."""

        # forward pass
        for l, layer in enumerate(self.layers):
            if l == 0:
                ass[l] = sample
            else:
                layer.forward(l, ass, bundles[l][indexInBatch])

        # compute loss and gradient
        loss = self.lossF.f(ass[-1], label)
        dErrors = [np.empty_like(a) for a in ass]
        dErrorLeftOperand = self.lossF.fprime(ass[-1], label)

        # propagate the error back
        for l in reversed(range(1, len(self.layers))):
            dErrorLeftOperand = self.layers[l].backward(l, ass, dErrors, bundles[l][indexInBatch], dErrorLeftOperand)

        return loss

    def batchForwardBackward(self, batch, learningRate, ass, dErrors, bundles):
        """Trains the network over given batch. Called by train(). Returns the mean
        loss relative to samples of the batch."""

        # reset bundles
        for layerIndex in range(1, len(bundles)):
            for bundle in bundles[layerIndex]:
                self.layers[layerIndex].resetBundle(bundle)

        # forwardBackward on each sample, and sum up modifications suggested by the gradients
        iterations = 0
        sumLoss = 0
        for indexInBatch, (sample, label, labelScalar) in enumerate(batch):
            iterations += 1
            sumLoss += self.forwardBackward(sample, label, indexInBatch, ass, bundles)
        meanLoss = sumLoss/iterations

        # modify weigths and biases according to previous backpropagations
        for layerIndex in range(1, len(bundles)):
            self.layers[layerIndex].update(layerIndex, ass, learningRate, bundles[layerIndex])

        return meanLoss


    def train(self, dataset, epochs, batchSize, learningRate):
        """Trains the network using the training set of
        the given dataset, during given amount of epochs, using given
        batch size, and beginning with given learning rate."""

        datas = [layer.getDataStructures()
                 for layer in self.layers[1:]]
        ass = [a for a, dError in datas]
        dErrors = [dError for a, dError in datas]
        # dErrors.append(None)  # reserve a spot for

        bundles = [[layer.createBundle() for bundle in range(batchSize)]
                   for layer in self.layers[1:]]
        ass.insert(0, None)  # reserve a spot for the sample as input of the network
        dErrors.insert(0, None)  # reserve a spot for the sample as input of the network
        bundles.insert(0, None)  # reserve a spot for the sample as input of the network

        print("\nDeNet training with configuration:")
        print("\tSize of hidden layers:", [layer.getSize() for layer in self.layers[1:]])
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
                iterations += 1
                batchLoss = self.batchForwardBackward(
                    dataset.train[batchBeginning:batchEnd],
                    learningRate,
                    ass, dErrors, bundles
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
            learningRate *= 0.97**math.floor(epoch/self.learningRateStep)


class Tests(unittest.TestCase):
    def setUp(self):
        self.data = Dataset.loadSmallPickledData()
        self.net = DeNet()

    def test_evaluateRandom(self):
        res = self.net.guessLabel(self.data.train[0][0])
        print("random guess:", res)

    def test_evaluate(self):
        res = self.net.evaluate(self.data)
        print("success rate:", res)


def runSmall():
    data = Dataset.loadSmallPickledData()
    net = DeNet()
    net.train(data, 2, 50, 0.1)


def runMedium():
    data = Dataset.loadMediumPickledData()
    net = DeNet()
    net.train(data, 30, 100, 0.1)


def runBig():
    data = Dataset.loadPickledData()
    net = DeNet()
    net.train(data, 30, 100, 0.1)


if __name__ == '__main__':
    # unittest.main()
    # runSmall()
    # runMedium()
    runBig()
