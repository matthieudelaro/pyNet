"""Activation module contains activation functions."""

import numpy as np
import unittest


class Sigmoid:
    def f(z):
        return 1.0/(1.0+np.exp(-z))

    def fprime(z):
        sigmoidResult = 1.0/(1.0+np.exp(-z))
        return sigmoidResult*(1-sigmoidResult)

    def name():
        return "Sigmoid"


class LeakyReLU:
    def f(z):
        return z * (z > 0) + 0.1 * z * (z < 0)

    def fprime(z):
        return 0.1 * (z <= 0) + 1 * (z > 0)

    def name():
        return "LeakyReLU"


class ReLU:
    def f(z):
        # return np.maximum(0, z)
        return z * (z > 0)  # faster than np.max implementation

    def fprime(z):
        return 1 * (z > 0)

    def name():
        return "ReLU"


class Copy:
    """This fake activation function has been developed for
    debug purpose only."""
    def f(z):
        return z

    def fprime(z):
        return np.ones_like(z)

    def name():
        return "Copy"


class Tests(unittest.TestCase):
    def setUp(self):
        pass

    def test_ReLU(self):
        act = ReLU
        self.assertTrue(np.allclose(act.f(np.array([-2, -1, 0, 1, 2]).reshape(5, 1)),
                        np.array([[0],
                                 [0],
                                 [0],
                                 [1],
                                 [2]])))
        self.assertTrue(np.allclose(act.fprime(np.array([-2, -1, 0, 1, 2]).reshape(5, 1)),
                        np.array([[0],
                                 [0],
                                 [0],
                                 [1],
                                 [1]])))
        self.assertTrue(np.allclose(act.fprime(np.array([-2, -1, 0, 1, 2])),
                        np.array([0,
                                  0,
                                  0,
                                  1,
                                  1])))

    def test_LeakyReLU(self):
        act = LeakyReLU
        # print(act.f(np.array([-2, -1, 0, 1, 2])))
        self.assertTrue(np.allclose(act.f(np.array([-2, -1, 0, 1, 2]).reshape(5, 1)),
                        np.array([[-0.2],
                                 [-0.1],
                                 [0],
                                 [1],
                                 [2]])))
        self.assertTrue(np.allclose(act.fprime(np.array([-2, -1, 0, 1, 2]).reshape(5, 1)),
                        np.array([[0.1],
                                 [0.1],
                                 [0.1],
                                 [1],
                                 [1]])))
        self.assertTrue(np.allclose(act.fprime(np.array([-2, -1, 0, 1, 2])),
                        np.array([0.1,
                                  0.1,
                                  0.1,
                                  1,
                                  1])))

if __name__ == '__main__':
    unittest.main()
