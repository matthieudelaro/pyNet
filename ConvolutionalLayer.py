import numpy as np


class ConvolutionalLayer(object):
    """Convolutional layer."""

    def getSize(self):
        """Returns the length of the output (ie. quantity of neurons)"""
        raise NotImplementedError()

    def getShape(self):
        """Returns the shape of the output. For a fully connected layer,
        the shape is a vector of (getSize() x 1)"""
        return np.shape(self.getSize(), 1)

    def fastForward(self, inputData):
        """Returns the activation of the layer."""
        raise NotImplementedError()

    def forward(self, l, ass, bundle):
        """Executes a forward pass as layer l.
        Neurons activations must be stored in ass[l].
        All other data which is required
        to perform the backward pass must be stored in the bundle."""
        raise NotImplementedError()

    def backward(self, l, ass, dErrors, bundle, W_lPlus1):
        """Performs backward propagation as layer l. Neurons activations
        are given as in ass[l].
        dError of the top layer is accessible as dErrors[l+1]."""
        raise NotImplementedError()

    def update(self, l, ass, bundles):
        """Performs updates (weights updates, bias updates, etc)
        as layer l, given neurons activations azz[l].
        Given bundles is a list of the bundles of the whole batch,
        for this layer."""
        raise NotImplementedError()

    def getDataStructures(self):
        """Initializes and returns data structures with proper shapes for:
        - dError of the layer (accessible later as dErrors[l])
        - activation of the layer (accessible later as ass[l])"""
        raise NotImplementedError()

    def createBundle(self):
        """Create, initialize and return a bundle."""
        raise NotImplementedError()

    def resetBundle(self, bundle):
        """Take whatever required action on the bundle to make
        it proper for re-use."""
        raise NotImplementedError()


class Bundle(object):
    pass
