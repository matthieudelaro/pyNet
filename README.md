# PyNet
PyNet is my own implementation of a neural network in Python to learn from MNIST dataset (handwritten digits).
It is my first step toward an implementation on GPU with CUDA. The goal is to prototype algorithms and an architecture which are close to those that will be parallelized and efficient on GPU, keeping in mind to limit data transfer between host and device.

## LoopyNet
The first network, LoopyNet, is inspired from the book by Michael Nielsen : http://neuralnetworksanddeeplearning.com/chap1.html
It is composed of fully connected layers only, represented by the weights and biases required to compute the forward pass and backward pass in a loop. Main mathematical formulas applied and refered to in the source code are the following :
![alt tag](http://neuralnetworksanddeeplearning.com/images/tikz21.png)
Optimization have been done by allocating numpy arrays only once when possible, instead of doing it for each forward/backward computation. It provides gradient descent backward propagation, batches, Softmax loss, ReLU/LeakyReLU/Sigmoid activation functions, drop out, and learning rate decay.
It achieves 97-98% accuracy with 3 input layers (140 hidden neurons : 100 + 30 + 10 neurons).

## DeNet
While LoopyNet is almost self contained and rather easy to read, it lakes modularity. Its design is not easily parallelizable, and it assumes that all layers are fully connected layers. They should all have fully connected connections, with weights and biases, the same activation function, etc.

DeNet is designed to solve this problem by implementing layers in a separate class. Each layer handles his own data (in a Bundle), computes forward and backward pass on his own, and updates configuration (weights, biases, etc) on his own. It is quite similar to the architecture of Caffe.

A FCLayer class implements fully connected layer that work the same as those implemented in LoopyNet.
Classes to implement convolutional layers, pooling layers, and ReLU layers are yet to be implemented.

## Requirements
PyNet requires Python 3.5 (@ operator for dot product), as well as numpy and idx2numpy packages.
It also requires MNIST dataset preformated for the network. You can download and
pre-process it automatically by running ./downloadAssets.sh.

## Usage
To train the network, run ./src/LoopyNet.py, or ./src/DeNet.py

##File Architecture:
### ./src/
Source files

### ./data/
Training and testing data sets reside in ./data. Those are
ignored in the repot, but can be downloaded automatically
by running ./downloadAssets.sh.


