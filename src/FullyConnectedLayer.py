import Layer
import Activation
import unittest
import numpy as np


class FCLayer(Layer.Layer):
    """Fully connected layer."""

    def __init__(self, inputSize, neuronsQuantity,
                 activationFunction, dropOut):
        self.W = 0.01 * np.random.randn(neuronsQuantity, inputSize).astype(dtype="double")
        self.b = 0.01 * np.random.randn(neuronsQuantity, 1).astype(dtype="double")
        self.activationF = activationFunction
        self.dropOut = dropOut

    def getSize(self):
        """Returns the length of the output (ie. quantity of neurons)"""
        return self.W.shape[0]

    def fastForward(self, inputData):
        """Returns the activation of the layer."""
        return self.activationF.f(self.W@inputData + self.b)

    def forward(self, l, ass, bundle):
        """Executes a forward pass as layer l.
        Neurons activations must be stored in ass[l].
        All other data which is required
        to perform the backward pass must be stored in the bundle."""
        bundle.z = self.W@ass[l-1] + self.b
        ass[l] = self.activationF.f(bundle.z)

        if self.dropOut < 1:  # move this to resetBundle if it seems ok to update mask only once per batch
            bundle.dropMask = (np.random.rand(*self.b.shape) < self.dropOut)/self.dropOut

        if self.dropOut < 1:
            ass[l] *= bundle.dropMask

    def backward(self, l, ass, dErrors, bundle, dErrorLeftOperand):
        """Performs backward propagation as layer l. Neurons activations
        are given as in ass[l].
        dError of the top layer is accessible as dErrors[l+1]."""
        #            W_lPlus1.T@dErrors[l+1]
        dErrors[l] = dErrorLeftOperand     * self.activationF.fprime(bundle.z)

        if self.dropOut < 1:
            dErrors[l] *= bundle.dropMask

        bundle.dW = dErrors[l]@ass[l-1].T
        bundle.db = dErrors[l]

        return self.W.T@dErrors[l]  # return dErrorLeftOperand for next layer

    def update(self, l, ass, learningRate, bundles):
        """Performs updates (weights updates, bias updates, etc)
        as layer l, given neurons activations azz[l].
        Given bundles is a list of the bundles of the whole batch,
        for this layer."""
        iterations = len(bundles)
        for bundle in bundles:
            self.W -= bundle.dW * (learningRate / iterations)
            self.b -= bundle.db * (learningRate / iterations)

    def getDataStructures(self):
        """Initializes and returns data structures with proper shapes for:
        - activation of the layer (accessible later as ass[l])
        - dError of the layer (accessible later as dErrors[l])"""
        dError = np.empty_like(self.b)
        a = np.empty_like(self.b)
        return a, dError

    def createBundle(self):
        """Create, initialize and return a bundle."""
        bundle = FCLayer.Bundle()
        bundle.dW = np.zeros_like(self.W)
        bundle.db = np.zeros_like(self.b)
        bundle.z = np.empty((self.getSize(), 1)).astype(dtype="double")
        if self.dropOut < 1:
            bundle.dropMask = (np.random.rand(*self.b.shape) < self.dropOut)/self.dropOut
        return bundle

    def resetBundle(self, bundle):
        """Take whatever required action on the bundle to make
        it proper for re-use."""
        bundle.dW = np.zeros_like(self.W)
        bundle.db = np.zeros_like(self.b)
        # if self.dropOut < 1:
            # bundle.dropMask = (np.random.rand(*self.b.shape) < self.dropOut)/self.dropOut

    class Bundle(Layer.Bundle):
        pass


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.inputSize = 10
        self.neuronsQuantity = 5
        self.nextLayerSize = 3
        self.activationF = Activation.LeakyReLU
        self.dropOut = 0.7
        self.layer = FCLayer(self.inputSize,
                             self.neuronsQuantity,
                             self.activationF,
                             self.dropOut)
        self.a, self.dError = self.layer.getDataStructures()
        self.bundle = self.layer.createBundle()
        self.W_lPlus1 = 0.01 * np.random.randn(self.nextLayerSize, self.neuronsQuantity).astype(dtype="double")
        self.ass = [np.random.rand(self.inputSize, 1), self.a, np.random.rand(self.nextLayerSize, 1)]
        self.dErrors = [None, self.dError, np.random.rand(self.nextLayerSize, 1)]

    def test_getSize(self):
        self.assertEqual(self.neuronsQuantity,
                         self.layer.getSize())

    def test_fastForward(self):
        self.layer.fastForward(np.random.rand(self.inputSize, 1))

    def test_forward(self):
        self.layer.forward(1, self.ass, self.bundle)

    def test_backward(self):
        dErrorLeftOperand = self.W_lPlus1.T@self.dErrors[2]
        self.layer.backward(1, self.ass, self.dErrors, self.bundle, dErrorLeftOperand)

    def test_update(self):
        self.layer.update(1, self.ass, 0.1, [self.bundle])



if __name__ == '__main__':
    unittest.main()
