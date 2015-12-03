# PyNet
PyNet is my own implementation of a neural network in Python to learn from MNIST dataset (handwritten digits).
It is my first step toward an implementation on GPU with CUDA. The goal is to prototype algorithms and an architecture which are close to those that will be parallelized and efficient on GPU, keeping in mind to limit data transfer between host and device.

## LoopyNet
The first network, LoopyNet, is inspired from the book by Michael Nielsen : http://neuralnetworksanddeeplearning.com/chap1.html
It is composed of fully connected layers only, represented by the weights and biases required to compute the forward pass and backward pass in a loop. Main mathematical formulas applied and refered to in the source code are the following :
![alt tag](http://neuralnetworksanddeeplearning.com/images/tikz21.png)

Optimization have been done by allocating numpy arrays only once when possible, instead of doing it for each forward/backward computation. It provides gradient descent backward propagation, batches, Softmax loss, ReLU/LeakyReLU/Sigmoid activation functions, drop out, and learning rate decay. Those parameters can be changed in the constructor of LoopyNet. 
It achieves 97-98% accuracy with 3 input layers (140 hidden neurons : 100 + 30 + 10 neurons).

As of commit a648ea9626624e4df991798126a50fe8159ea4d8, drop out does not significatively improve the performances. (Maybe because the quantity of neurons is really limited?)

## DeNet
While LoopyNet is almost self contained and rather easy to read, it lakes modularity. Its design is not easily parallelizable, and it assumes that all layers are fully connected layers. They should all have fully connected connections, with weights and biases, the same activation function, etc.

DeNet is designed to solve this problem by implementing layers in a separate class. Each layer handles his own data (in a Bundle), computes forward and backward pass on his own, and updates configuration (weights, biases, etc) on his own. It is quite similar to the architecture of Caffe.

A FCLayer class implements fully connected layer that work the same as those implemented in LoopyNet.
Classes to implement convolutional layers, pooling layers, and ReLU layers are yet to be implemented. They should inherite from the class Layer, which behaves as an interface.

As of commit a648ea9626624e4df991798126a50fe8159ea4d8, DeNet is slower than LoopyNet. It must be due to poor memory management. Also, it could be improved by using numpy implementation from PyCUDA.

### FCLayer : Fully Connected Layer
FCLayer stores its weights and biases as member variables. In fact, those are read only during a batch evaluation, so those can be accessed by several threads at the same time. However, the activation is stored in a numpy array provided to the method Layer.forward(), so that it can be passed to the next layer. Other data, such as weight updates and biases updates are stored in a bundle.
Only one instance of the layer is in memory at a time, through out the whole life time of the network. But different activations of a same layer (during a batch evaluation) are represented by its bundle.

### Bundles
Each type of layer may have different kinds of data to handle, and data computed during the forward pass (Layer.forward()) might be needed during the backward pass (Layer.backward()) and the update phase (Layer.update(), at the end of evaluation of a batch).
Each layer takes care of creating a bundle (Layer.createBundle()) and to reset it (Layer.resetBundle()) when the batch process begins.

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

## References
- Deep Learning For Computer Vision, KAIST class, SIIT laboratory
- Book by Michael Nielsen : http://neuralnetworksanddeeplearning.com/chap1.html
- Notes of the Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition : http://cs231n.github.io/
- MNIST dataset : http://yann.lecun.com/exdb/mnist/


