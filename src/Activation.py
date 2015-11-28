import numpy as np
import unittest

class Sigmoid:
    def f(z):
        return 1.0/(1.0+np.exp(-z))

    def fprime(z):
        sigmoidResult = 1.0/(1.0+np.exp(-z))
        return sigmoidResult*(1-sigmoidResult)


class LeakyReLU:
    def f(z):
        # return np.max(0.1 * z, z)
        return z * (z > 0) + 0.1 * z * (z < 0)

    def fprime(z):
        return 0.1 * (z <= 0) + 1 * (z > 0)


class ReLU:
    def f(z):
        # return np.maximum(0, z)
        return z * (z > 0)

    # def fprime(z):
        # return np.power(z, np.zeros_like(z))

    def fprime(z):
        return 1 * (z > 0)
        # res = np.empty_like(z)
        # for i in range(len(z)):
        #     if z[i] > 0:
        #         res[i] = 1
        #     else:
        #         res[i] = 0
        # return res


class Copy:
    """This fake activation function has been developed for
    debug purpose only."""
    def f(z):
        return z

    def fprime(z):
        return np.ones_like(z)


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
