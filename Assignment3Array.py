import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Assignment2
import numpy.matlib as matlib
import math

np.random.seed(400)
trainPath = 'Datasets/cifar-10-batches-mat/data_batch_1.mat'
valPath = 'Datasets/cifar-10-batches-mat/data_batch_2.mat'
testPath = 'Datasets/cifar-10-batches-mat/test_batch.mat'

n_layers = 3
nr_node = 2


def LoadBatch(filename):
    enc = OneHotEncoder()

    Data = loadmat(filename)

    Im = Data['data'].transpose()
    scaler = MinMaxScaler()
    scaler.fit(Im)
    scaledIm = scaler.transform(Im)

    Label = Data['labels']
    fit = enc.fit(Label)
    labelsEnc = enc.transform(Label).toarray().transpose()

    mean_X = scaledIm.mean(axis=1)
    X = scaledIm - matlib.repmat(mean_X, scaledIm.shape[1], 1).transpose()

    return X, labelsEnc, Label


trainX, trainY, trainy = LoadBatch(trainPath)
valX, valY, valy = LoadBatch(valPath)
testX, testY, testy = LoadBatch(testPath)

n = trainX.shape[1]
d = trainX.shape[0]
k = trainY.shape[0]

def initVariables(n_layer, n_nodes, input_size, output_size):
    Weigths = []
    biases = []
    for i in range(n_layer):
        if (i == n_layer - 1):
            Weigths.append(np.random.normal(loc=0, scale=0.001, size=(output_size, input_size)))
            biases.append(np.zeros((output_size, 1)))
        else:
            Weigths.append(np.random.normal(loc=0, scale=0.001, size=(n_nodes, input_size)))
            biases.append(np.zeros((n_nodes, 1)))

        input_size = n_nodes

    return Weigths, biases#np.asarray(Weigths), np.asarray(biases)



#
# def EvaluateClassifier(input, Ws, bs, n_layer):
#     X = input
#     s = []#np.zeros(n_layer)
#     p = 0
#     for l in range(n_layer):
#         s.append(np.dot(Ws[l], X) + bs[l])
#         X = np.zeros(s[l].shape)
#         if (l == n_layer-1):
#             p = np.exp(s[l]) / np.sum(np.exp(s[l]))
#         else:
#             for i in range(s[l].shape[0]):
#                 for j in range(s[l].shape[1]):
#                     X[i][j] = np.maximum(0, s[l][i][j])
#
#     s.pop()
#     s.insert(0,input)
#     return np.asarray(s),p


def EvaluateClassifier1(input, weight, bias):
    X = []
    X.append(input)
    s = []
    P = 0
    for l in range(0,n_layers):
        if l == n_layers-1:
            value = np.dot(weight[l], X[l]) + bias[l]
            s.append(value)
            P = np.exp(s[l]) / np.sum(np.exp(s[l]))
        else:
            value = np.dot(weight[l], X[l]) + bias[l]
            s.append(value)
            zeroS = np.zeros(np.asarray(s)[l].shape)
            X.append(np.maximum(zeroS, s[l]))

    return P, X, s

weights, biases = initVariables(n_layers, nr_node, d, k)

def ComputeGradient(varX, varY, weight, bias, lamda=0):
    g = []
    n = valX.shape[1]
    grad_b = bias.copy()
    grad_w = weight.copy()

    P, X, S = EvaluateClassifier1(varX, weight, bias)
    g = -np.asarray(np.subtract(np.asarray(varY), P)).T

    val1 = np.sum(np.asarray(g).T, axis=1) / n
    grad_b[n_layers-1] = val1

    val2 = (np.dot(np.asarray(g).T, X[n_layers-2].T) + 2 * lamda * weights[n_layers-1]) / n
    grad_w[n_layers-1] = val2

    # print(np.asarray(grad_w)[n_layers-1].shape)
    # print(np.asarray(grad_b)[n_layers-1].shape)

    g = np.dot(g,weights[n_layers-1])
    newS1 = []
    s1 = S
    for indexi in range(s1[n_layers-1].shape[0]):
        if (s1[n_layers-1][indexi][0] > 0):
            newS1.append(1)
        else:
            newS1.append(0)

    newS1 = np.asarray(newS1)
    newS1 = np.diagflat(newS1)
    g = np.dot(g,newS1)

    for l in range(n_layers):
        val1 = np.sum(np.asarray(g).T, axis=1) / n
        grad_b[l] = (val1)
        val2 = (np.dot(np.asarray(g).T, X[n_layers - 2].T) + 2 * lamda * weights[l]) / n
        grad_w[l] = val2
        if l>1:
            g = np.dot(g, weights[n_layers - 1])
            newS1 = []
            s1 = S
            for indexi in range(s1[n_layers - 1].shape[0]):
                if (s1[n_layers - 1][indexi][0] > 0):
                    newS1.append(1)
                else:
                    newS1.append(0)
    return grad_w, grad_b

grad_w, grad_b = ComputeGradient(trainX[:,:1], trainY[:,:1], weights, biases, lamda=0)
print(grad_w)
print(grad_b)