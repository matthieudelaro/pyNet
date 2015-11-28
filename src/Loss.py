import numpy as np
import math


class Softmax:
    def f(scores, label, margin=0):
        # try:
        #     nomin = math.exp(scores[label.argmax()])
        #     denom = np.sum(np.exp(scores))
        #     res = -math.log(nomin / denom)
        # except:
        #     print("error for scores", scores)
        # return res
        nomin = math.exp(scores[label.argmax()])
        denom = np.sum(np.exp(scores))
        return -math.log(nomin / denom)

    def fprime(scores, label):
        return scores - label
