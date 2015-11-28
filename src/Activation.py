import numpy as np


class Sigmoid:
    def f(z):
        return 1.0/(1.0+np.exp(-z))

    def fprime(z):
        sigmoidResult = 1.0/(1.0+np.exp(-z))
        return sigmoidResult*(1-sigmoidResult)


class ReLU:
    def f(z):
        return np.maximum(0, z)

    def fprime(z):
        res = np.empty_like(z)
        for i in range(len(z)):
            if z[i] > 0:
                res[i] = 1
            else:
                res[i] = 0
        return res
