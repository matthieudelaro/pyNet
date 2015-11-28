import numpy as np
import Dataset
from Dataset import OptimizedDatabase
import unittest
import Loss
import time
import random
import Activation


class Net:
    def __init__(self):
        self.sizes = np.array([784, 100, 30, 10])  # size of each layer
        self.L = len(self.sizes)  # number of layers
        # print(sizes[:-1])
        self.W = [0.01 * np.random.randn(sizeCurrent, sizePrevious).astype(dtype="double")
             for sizePrevious, sizeCurrent
             in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [0.01 * np.random.randn(sizeCurrent, 1).astype(dtype="double")
             for sizeCurrent in self.sizes[1:]]
        self.activationF = Activation.Sigmoid  # determine activation function
        self.lossF = Loss.Softmax

    def guessLabel(self, sample):
        """Return the class evaluated by the network for the
        given sample."""
        # a = np.array([sample]).T
        a = sample
        for W, b in zip(self.W, self.b):
            # aprevious = a[:]
            z = W@a + b
            # z = np.dot(W, a) + b
            a = self.activationF.f(z)  # @ is the dot product (python3.5)
            # print("self.activationF.f(z) / z: ", a.shape, z.shape)
            # print("a <= W@a + b : ", a.shape, " <= ", W.shape, aprevious.shape, " + ", b.shape)
        return np.argmax(a)

    def evaluate(self, dataset):
        """Returns the success rate of the network over the test set
        of the given dataset."""
        success = 0
        for sample, labelVector, label in dataset.tests:
            if self.guessLabel(sample) == label:
                success += 1
        return success / len(dataset.tests)

    def forwardBackward(self, sample, label):#, ass, zss):
        # sample = np.array([sample]).T  # todo: change the shape of the samples
        ass = [sample]  # stores activation (=f(z)) for each layer
        zss = []  # stores z (=W@a + b) for each layer

        # forward pass: for each layer (except for the first layer
        # which is just input data)
        # compute z and activation a
        a = sample  # activation of first layer: input sample
        for W, b in zip(self.W, self.b):
            z = W@a + b
            a = self.activationF.f(z)
            ass.append(a)
            zss.append(z)

        # compute loss and loss gradient
        loss = self.lossF.f(a, label)
        # print("loss:", loss)

        dError = self.lossF.fprime(ass[-1], label) * self.activationF.fprime(zss[-1])
        # print("dError:", dError.reshape((10)))
        # print("label:", label)

        # propagate the error back
        dWs = [np.zeros_like(W) for W in self.W]  # for each layer, stores dW, ie how much weights should be modified
        dbs = [np.zeros_like(b) for b in self.b]  # for each layer, stores db, ie how much biases should be modified

        for l in reversed(range(len(self.W))):
            # retrieve required values
            if l < len(self.W)-1:
                W_lPlus1 = self.W[l+1]

                dError_lPlus1 = dError
                z = zss[l]

                # compute the error
                dError = (W_lPlus1.T @ dError_lPlus1) * self.activationF.fprime(z)

            # if l != 0:
            #     a_lMinus1 = ass[l-1]
            # else:
            #     a_lMinus1 = sample
            a_lMinus1 = ass[l]

            # print("dW <= a_lMinus1 dError :", dWs[l].shape, "<=", a_lMinus1.shape, dError.shape)
            # dW = a_lMinus1 * dError
            dW = dError@a_lMinus1.T
            db = dError

            dWs[l] = dW
            dbs[l] = db
        return dWs, dbs, loss

    def batchForwardBackward(self, batch, learningRate):
        dWs = [np.zeros_like(W) for W in self.W]  # for each layer, stores dW, ie how much weights should be modified
        dbs = [np.zeros_like(b) for b in self.b]  # for each layer, stores db, ie how much biases should be modified

        # forwardBackward on each sample, and sum up modifications suggested by the gradients
        iterations = 0
        sumLoss = 0
        for sample, label, labelScalar in batch:
            iterations += 1
            dWsCurrent, dbsCurrent, loss = self.forwardBackward(sample, label)
            for summ, current in zip(dWs, dWsCurrent):
                summ += current
            for summ, current in zip(dbs, dbsCurrent):
                summ += current
            sumLoss += loss
        # print("len(samples)", len(samples))
        # print("len(labels)", len(labels))
        meanLoss = sumLoss/iterations
        # print("mean loss over batch of", iterations, ":", meanLoss)

        # modify weigths and biases according to previous backpropagations
        for W, dW in zip(self.W, dWs):
            W -= dW * (learningRate / iterations)
        for b, db in zip(self.b, dbs):
            b -= db * (learningRate / iterations)

        return meanLoss

    def train(self, dataset, epochs, batchSize, learningRate):
        for epoch in range(epochs):
            random.shuffle(dataset.train)
            batchBeginning = 0
            batchEnd = min(len(dataset.train), batchBeginning + batchSize)
            # batchEnd = min(len(dataset.trainX-1), batchBeginning + batchSize)

            iterations = 0
            sumLoss = 0
            while (batchEnd - batchBeginning) >= 2:
                iterations += 1
                # print("Running batch over", batchBeginning, batchEnd)
                batchLoss = self.batchForwardBackward(
                    dataset.train[batchBeginning:batchEnd],
                    # dataset.trainY[batchBeginning:batchEnd],
                    learningRate
                )
                # print("batch loss:", batchLoss)
                sumLoss += batchLoss

                batchBeginning = batchEnd
                batchEnd = min(len(dataset.train), batchBeginning + batchSize)
                # batchEnd = min(len(dataset.trainX-1), batchBeginning + batchSize)

            meanLoss = sumLoss / iterations
            print()
            print("End of epoch ", epoch, ". Timer:", time.clock())
            print("Mean loss :", meanLoss)
            print("Test success rate:", self.evaluate(dataset))




            # lastBatchOfEpoch = False

            # while !lastBatchOfEpoch:
            # batchEnd = max(len(dataSet.trainX-1), batchBeginning + batchSize)



    # data = SmallerDataset()

class TestsSmaller(unittest.TestCase):
    def setUp(self):
        # self.data = Dataset.SmallerDataset()
        self.data = Dataset.loadSmallPickledData()
        self.net = Net()

    # def test_evaluateRandom(self):
        # res = self.net.guessLabel(self.data.trainX[0])
        # print(res)
        # pass

    # def test_backprop(self):
        # self.net.forwardBackward(self.data.trainX[0], self.data.trainY[0])

    # def test_batchBackprop(self):
    #     batch1 = 90
    #     batch2 = 90
    #     self.net.batchForwardBackward(self.data.trainX[:batch1], self.data.trainY[:batch1], 0.01)
    #     self.net.batchForwardBackward(self.data.trainX[batch1:batch1+batch2], self.data.trainY[batch1:batch1+batch2], 0.01)

    def test_train(self):
        self.net.train(self.data, 30, 50, 3)

# class TestsBigger(unittest.TestCase):
#     def setUp(self):
#         self.data = Dataset.loadPickledData()
#         self.net = Net()

#     # def test_evaluateRandom(self):
#         # res = self.net.guessLabel(self.data.trainX[0])
#         # print(res)
#         # pass

#     # def test_backprop(self):
#         # self.net.forwardBackward(self.data.trainX[0], self.data.trainY[0])

#     # def test_batchBackprop(self):
#     #     batch1 = 90
#     #     batch2 = 90
#     #     self.net.batchForwardBackward(self.data.trainX[:batch1], self.data.trainY[:batch1], 0.01)
#     #     self.net.batchForwardBackward(self.data.trainX[batch1:batch1+batch2], self.data.trainY[batch1:batch1+batch2], 0.01)

#     def test_ReLU(self):
#         self.assertTrue(np.allclose(Activation.ReLU.fprime(np.array([-2, -1, 0, 1, 2]).reshape(5, 1)),
#                         np.array([[0],
#                                  [0],
#                                  [0],
#                                  [1],
#                                  [1]])))
#         self.assertTrue(np.allclose(Activation.ReLU.fprime(np.array([-2, -1, 0, 1, 2])),
#                         np.array([0,
#                                   0,
#                                   0,
#                                   1,
#                                   1])))
#         # print(Activation.ReLU.fprime(np.array([[-2, -1, 0, 1, 2]])))
#         # self.assertTrue(np.allclose(Activation.ReLU.fprime(np.array([[-2, -1, 0, 1, 2]])),
#                         # np.array([[0],
#                         #           [0],
#                         #           [0],
#                         #           [1],
#                         #           [1]])))
#         # self.assertEquals()

#     def test_train(self):
#         self.net.train(self.data, 30, 100, 3)
#         # print()
#         pass


if __name__ == '__main__':
    unittest.main()
