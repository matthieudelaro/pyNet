"""Loss module contains definition of loss functions"""

import numpy as np
import math


class Softmax:
    def f(scores, label, margin=0):
        nomin = math.exp(scores[label.argmax()])
        denom = np.sum(np.exp(scores))
        return -math.log(nomin / denom)

    def fprime(scores, label):
        return scores - label

    def name():
        return "Softmax"
