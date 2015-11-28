import idx2numpy
import unittest
import time
import pickle


class OriginalMNISTDataset:
    def __init__(self):
        print("Loading dataset from files...")
        self._load(
            idx2numpy.convert_from_file('../data/MNISTdataset/train-images-idx3-ubyte.idx'),
            idx2numpy.convert_from_file('../data/MNISTdataset/train-labels-idx1-ubyte.idx'),
            idx2numpy.convert_from_file('../data/MNISTdataset/t10k-images-idx3-ubyte.idx'),
            idx2numpy.convert_from_file('../data/MNISTdataset/t10k-labels-idx1-ubyte.idx')
        )
        print("Dataset loaded from files.")

    def _load(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY


class OptimizedDataset:
    pass


class OptimizedDatabase(object):
    """Database stored in pickle file"""
    pass


def loadPickledData():
    print("Loading dataset from file...")
    f_read = open('../data/pickledMNIST/data.pkl', 'br')
    data = pickle.load(f_read)
    f_read.close()
    print("Dataset loaded from file. Timer:", time.clock())
    return data


def loadMediumPickledData():
    print("Loading dataset from file...")
    f_read = open('../data/pickledMediumMNIST/data.pkl', 'br')
    data = pickle.load(f_read)
    f_read.close()
    print("Dataset loaded from file. Timer:", time.clock())
    return data


def loadSmallPickledData():
    print("Loading dataset from file...")
    f_read = open('../data/pickledSmallMNIST/data.pkl', 'br')
    data = pickle.load(f_read)
    f_read.close()
    print("Dataset loaded from file. Timer:", time.clock())
    return data


class TestsPickled(unittest.TestCase):
    def setUp(self):
        pass

    def test_small(self):
        self.data = loadSmallPickledData()
        print(len(self.data.train))
        print(len(self.data.valid))
        print(len(self.data.tests))
        print("shape of example input:", self.data.train[0][0].shape)
        print("average of values of training example 10", self.data.train[10][0].sum() / 784)

    def test_medium(self):
        self.data = loadMediumPickledData()
        print(len(self.data.train))
        print(len(self.data.valid))
        print(len(self.data.tests))
        print("shape of example input:", self.data.train[0][0].shape)
        print("average of values of training example 10", self.data.train[10][0].sum() / 784)

    def test_big(self):
        self.data = loadPickledData()
        print(len(self.data.train))
        print(len(self.data.valid))
        print(len(self.data.tests))
        print("average of values of training example 10", self.data.train[10][0].sum() / 784)

if __name__ == '__main__':
    unittest.main()
