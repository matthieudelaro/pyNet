import numpy as np
import idx2numpy
import unittest
import time
import pickle


class OptimizedDatabase(object):
    """Database stored in pickle file"""
    pass

def loadPickledData():
    print("Loading dataset from file...")
    f_read = open('../data/pickledMNIST/data.pkl', 'br')
    data = pickle.load(f_read)
    f_read.close()
    for i in range(len(data.train)):
        data.train[i] = data.train[i][0].astype(dtype="double") / 255, data.train[i][1], data.train[i][2]
    print("Dataset loaded from file. Timer:", time.clock())
    return data

def loadSmallPickledData():
    print("Loading dataset from file...")
    f_read = open('../data/pickledSmallMNIST/data.pkl', 'br')
    data = pickle.load(f_read)
    f_read.close()
    for i in range(len(data.train)):
        data.train[i] = data.train[i][0].astype(dtype="double") / 255, data.train[i][1], data.train[i][2]
    # print(data.train[0])
    print("Dataset loaded from file. Timer:", time.clock())
    return data

class Dataset:
    def __init__(self):
        print("Loading dataset from file...")
        self._load(
            idx2numpy.convert_from_file('../data/reshapedMNIST/train-images-idx3-ubyte.idx'),
            idx2numpy.convert_from_file('../data/reshapedMNIST/train-labels-idx1-ubyte.idx'),
            idx2numpy.convert_from_file('../data/reshapedMNIST/t10k-images-idx3-ubyte.idx'),
            idx2numpy.convert_from_file('../data/reshapedMNIST/t10k-labels-idx1-ubyte.idx')
        )
        print("Dataset loaded from file. Timer:", time.clock())
        # print("Reshaping dataset...")
        # shape = self.trainX.shape
        # self.trainX = self.trainX.reshape(shape[0], shape[1] * shape[2])
        # shape = self.testX.shape
        # self.testX = self.testX.reshape(shape[0], shape[1] * shape[2])

    def _load(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY




class NewDataCreator:
    def __init__(self):
        self.old = Dataset()
        optim = self._reshape()

    def _reshape(self):
        ratio = 0.9  # percentageOfTrainingSetForValidation

        train = [()] * int(len(self.old.trainX)*ratio)
        valid = [()] * int(len(self.old.trainX)*(1-ratio))
        tests = [()] * len(self.old.testX)

        for i in range(len(train)):
            train[i] = (np.array([self.old.trainX[i]]).T, NewDataCreator._labelAsArray(self.old.trainY[i]), self.old.trainY[i])

        for i in range(len(valid)):
            j = i + len(train) - 1
            valid[i] = (np.array([self.old.trainX[j]]).T, NewDataCreator._labelAsArray(self.old.trainY[j]), self.old.trainY[j])

        for i in range(len(tests)):
            tests[i] = (np.array([self.old.testX[i]]).T, NewDataCreator._labelAsArray(self.old.testY[i]), self.old.testY[i])

        print("training set:", len(train))
        print("validation set:", len(valid))
        print("testing set:", len(tests))
        print(train[4])
        # print("old testing set:", )
        # print(train[0])

        optim = OptimizedDatabase()
        optim.train = train
        optim.valid = valid
        optim.tests = tests

        f_write = open('../data/pickledMNIST/data.pkl', 'bw')
        pickle.dump(optim, f_write, protocol=4, fix_imports=False)

    def _labelAsArray(label):
        array = np.zeros((10, 1))
        array[label] = 1
        return array

    def _load(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY


class SmallerDataset:
    def __init__(self):
        print("Loading dataset from file...")
        self._load(
            idx2numpy.convert_from_file('../data/smallerDataSet/train-images-idx3-ubyte.idx'),
            idx2numpy.convert_from_file('../data/smallerDataSet/train-labels-idx1-ubyte.idx'),
            idx2numpy.convert_from_file('../data/smallerDataSet/t10k-images-idx3-ubyte.idx'),
            idx2numpy.convert_from_file('../data/smallerDataSet/t10k-labels-idx1-ubyte.idx')
        )
        print("Dataset loaded from file. Timer:", time.clock())

    def _load(self, trainX, trainY, testX, testY):
        self.trainX = trainX  #.astype(dtype="float") #reshape((200, 1, 784))
        self.trainY = trainY  #.astype(dtype="float")
        self.testX = testX  #.astype(dtype="float")
        self.testY = testY  #.astype(dtype="float")


# class Tests(unittest.TestCase):
#     def setUp(self):
#         self.dataset = Dataset()

#     def test_content(self):
#         # print("self.dataset.trainX.shape:", self.dataset.trainX.shape)
#         # print("Before reshape:")
#         # print("self.dataset.trainX.shape:", self.dataset.trainX.shape)
#         # print("self.dataset.trainX[0]:", self.dataset.trainX[0])
#         # shape = self.dataset.trainX.shape
#         # self.dataset.trainX = self.dataset.trainX.reshape(shape[0], shape[1] * shape[2])
#         # print("After reshape:")
#         # print("self.dataset.trainX[0]:", self.dataset.trainX[0])
#         # print()
#         print("self.dataset.trainX.shape:", self.dataset.trainX.shape)
#         print("self.dataset.trainY.shape:", self.dataset.trainY.shape)
#         print("self.dataset.testX.shape:", self.dataset.testX.shape)
#         print("self.dataset.testY.shape:", self.dataset.testY.shape)
#         pass

#     # def test_saveSmallVersion(self):
#     #     # np.save(f_write, self.dataset.trainY[0:200])
#     #     # idx2numpy.write_to_file('myfile_copy.idx', ndarr)
#     #     # idx2numpy.convert_to_file('myfile_copy.idx', ndarr) #  doesn't work, because seem to open in w mode, instead of wb mode. Report issue to https://github.com/ivanyu/idx2numpy?
#     #     f_write = open('../smallerDataSet/train-images-idx3-ubyte.idx', 'bw')
#     #     idx2numpy.convert_to_file(f_write, self.dataset.trainX[0:200])
#     #     f_write = open('../smallerDataSet/train-labels-idx1-ubyte.idx', 'bw')
#     #     idx2numpy.convert_to_file(f_write, self.dataset.trainY[0:200])
#     #     f_write = open('../smallerDataSet/t10k-images-idx3-ubyte.idx', 'bw')
#     #     idx2numpy.convert_to_file(f_write, self.dataset.testX[0:200]),
#     #     f_write = open('../smallerDataSet/t10k-labels-idx1-ubyte.idx', 'bw')
#     #     idx2numpy.convert_to_file(f_write, self.dataset.testY[0:200])

#     # def test_saveReshapedVersion(self):
#     #     # np.save(f_write, self.dataset.trainY[0:200])
#     #     # idx2numpy.write_to_file('myfile_copy.idx', ndarr)
#     #     # idx2numpy.convert_to_file('myfile_copy.idx', ndarr) #  doesn't work, because seem to open in w mode, instead of wb mode. Report issue to https://github.com/ivanyu/idx2numpy?
#     #     f_write = open('../data/reshapedMNIST/train-images-idx3-ubyte.idx', 'bw')
#     #     idx2numpy.convert_to_file(f_write, self.dataset.trainX)
#     #     f_write = open('../data/reshapedMNIST/train-labels-idx1-ubyte.idx', 'bw')
#     #     idx2numpy.convert_to_file(f_write, self.dataset.trainY)
#     #     f_write = open('../data/reshapedMNIST/t10k-images-idx3-ubyte.idx', 'bw')
#     #     idx2numpy.convert_to_file(f_write, self.dataset.testX),
#     #     f_write = open('../data/reshapedMNIST/t10k-labels-idx1-ubyte.idx', 'bw')
#     #     idx2numpy.convert_to_file(f_write, self.dataset.testY)


# class TestsSmaller(unittest.TestCase):
#     def setUp(self):
#         self.dataset = SmallerDataset()

#     def test_content(self):
#         print("self.dataset.trainX.shape:", self.dataset.trainX.shape)
#         print("self.dataset.trainY.shape:", self.dataset.trainY.shape)
#         print("self.dataset.testX.shape:", self.dataset.testX.shape)
#         print("self.dataset.testY.shape:", self.dataset.testY.shape)


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

    def test_big(self):
        self.data = loadPickledData()
        print(len(self.data.train))
        print(len(self.data.valid))
        print(len(self.data.tests))
        print("average of values of training example 10", self.data.train[10][0].sum() / 784)

if __name__ == '__main__':
    unittest.main()
