import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import Assignment2
import numpy.matlib as matlib
import math

np.random.seed(400)
trainPath = 'Datasets/cifar-10-batches-mat/data_batch_1.mat'
valPath = 'Datasets/cifar-10-batches-mat/data_batch_2.mat'
testPath = 'Datasets/cifar-10-batches-mat/test_batch.mat'


######Change n_layers= 3
######Change n_layers = 4
n_layers = 2
nr_node = 5
dimRed = 100 #trainX.shape[0]




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

d = trainX.shape[0]
k = trainY.shape[0]
n = trainX.shape[1]
m = nr_node
nLayers = n_layers


def initVariables(n_layer, n_nodes, input_size, output_size):
    Weigths = {}
    biases = {}
    for i in range(1,n_layer+1):
        if (i == n_layer):
            Weigths.update({i: np.random.normal(loc=0, scale=0.001, size=(output_size, input_size))})
            biases.update({i: np.zeros((output_size, 1))})
        else:
            Weigths.update({i: np.random.normal(loc=0, scale=0.001, size=(n_nodes, input_size))})
            biases.update({i: np.zeros((n_nodes, 1))})

        input_size = n_nodes

    return Weigths, biases



weights, biases = initVariables(n_layers, nr_node, dimRed, trainY.shape[0])
# weights = {1: Assignment2.W1, 2: Assignment2.W2}
# biases = {1: Assignment2.b1, 2: Assignment2.b2}


def EvaluateClassifier(input, Ws, bs, n_layer):
    X = {0:input}
    s = []#np.zeros(n_layer)
    p = 0
    for l in range(1, n_layer+1):
        s.append(np.dot(Ws.get(l), X.get(l-1)) + bs.get(l))
        a = np.zeros(np.asarray(s[l-1]).shape)
        if l == n_layer:
            p = np.exp(s[l-1]) / np.sum(np.exp(s[l-1]))
        else:
            for i in range(s[l-1].shape[0]):
                for j in range(s[l-1].shape[1]):
                    a[i][j] = np.maximum(0, s[l-1][i][j])

            X.update({l: a})
    return X, p



def computeCost(X, Y, Weights, biases, n_layers, lamda):
    finCross = 0
    for i in range(X.shape[1]):
        h, P = EvaluateClassifier(np.asarray(X[:, i]).reshape(X.shape[0], 1), Weights, biases, n_layers)
        lCross = np.dot(np.asarray(Y[:, i]).T, P)
        logCross = -np.log(lCross)
        finCross += logCross
    expr1 = (1 / X.shape[1]) * finCross

    sqW = []
    for W in Weights.values():
        sqW.append(np.sum(np.square(W)))
    sqW = np.asarray(sqW).sum()

    expr2 = lamda * sqW
    J = expr1 + expr2
    return J, expr1

def initGradsDict(n_layer):
    grad_bs = {}
    for i in range(1, n_layer+1):
        grad_bs.update({i: 0})
    return grad_bs

def dictToArr(dict):
    arr = []#np.zeros(len(dict))
    for index, value in dict.items():
        arr.append(value)

    return np.asarray(arr)

# def ArrTodict(arr):



def ComputeGradient(X, Y, weights, biases, n_layer, lamda=0):
    n = X.shape[1]
    g = []
    grad_bs = weights.copy()
    grad_ws = biases.copy()


    S, P = EvaluateClassifier(np.asarray(X), weights, biases, n_layer)
    g = -np.asarray(np.subtract(np.asarray(Y), P)).T


    exb = np.zeros(np.asarray(biases.get(n_layer).T).shape)
    exw = np.zeros(np.asarray(weights.get(n_layer)).shape)

######### Equation 20 lab3
    # for i in range(n):
    #     exb += np.asarray(g)[i]
    #     ##helping variables for W grad
    #     g_trans = np.reshape(np.asarray(g)[i], (1,np.asarray(g)[i].shape[0])).T
    #     x_trans = np.reshape(S.get(n_layer-1)[i],(S.get(n_layer-1)[i].shape[0],1)).T
    #     exw = exw + np.dot(g_trans, x_trans)
    #
    #
    # exw = (exw + 2*lamda*weights.get(n_layer))/n
    #
    # exb = exb/n

    val1 = np.sum(np.asarray(g).T, axis=1)/n
    grad_bs.update({n_layer: val1})

    val2 = (np.dot(np.asarray(g).T, S.get(n_layer-1).T)+ 2*lamda*weights.get(n_layer))/n
    grad_ws.update({n_layer: val2})


    ########### Equation 21
    g = np.dot(g,weights.get(n_layer))
    newS1 = []
    s1 = S
    for indexi in range(s1.get(n_layer-1).shape[0]):
        if (s1.get(n_layer-1)[indexi][0] > 0):
            newS1.append(1)
        else:
            newS1.append(0)

    newS1 = np.asarray(newS1)
    newS1 = np.diagflat(newS1)
    g = np.dot(g,newS1)

    for l in range(n_layer-1, 0, 1):
        val1 = np.sum(np.asarray(g).T, axis=1) / n
        grad_bs.update({l: val1})

        val2 = (np.dot(np.asarray(g).T, S.get(l - 1).T) + 2 * lamda * weights.get(l)) / n
        grad_ws.update({n_layer: val2})

        if l>1:
            g = np.dot(g, weights.get(l))
            newS1 = []
            s1 = S
            for indexi in range(s1.get(l - 1).shape[0]):
                if (s1.get(l - 1)[indexi][0] > 0):
                    newS1.append(1)
                else:
                    newS1.append(0)

            newS1 = np.asarray(newS1)
            newS1 = np.diagflat(newS1)
            g = np.dot(g, newS1)

    return grad_ws, grad_bs

# ComputeGradient(trainX[:dimRed,:2], trainY[:,:2], weights, biases, n_layers, lamda=0)



def ComputeGradientsNumSlow( X, Y, Wt, bias, n_layers, lamda,h=1e-5):
    Wt = dictToArr(Wt)
    bias = dictToArr(bias)
    numgrad_W = Wt.copy()
    numgrad_b = bias.copy()


    for j in range(len(bias)):
        # numgrad_b[j] = np.zeros(b[j].shape)
        for i in range(len(bias[j])):
            b_try = bias.copy()
            b_try[j][i] -= h
            c1 = computeCost(X, Y, Wt, b_try, n_layers, lamda)

            b_try = bias.copy()
            b_try[j][i] += h
            c2 = computeCost(X, Y, Wt, b_try, n_layers, lamda)

            numgrad_b[j][i] = (c2 - c1)/(2*h)

    for j in range(len(Wt)):
        # numgrad_W = np.zeros(W[j].shape)
        for i in range(len(Wt[j])):
            for k in range(len(Wt[j][0])):
                W_try = Wt.copy()
                W_try[j][i][k] -= h
                c1 = computeCost(X, Y, W_try, bias, n_layers, lamda)

                W_try = Wt.copy()
                W_try[j][i][k] += h
                c2 = computeCost(X, Y, W_try, bias, n_layers, lamda)

                numgrad_W[j][i][k] = (c2-c1)/(2*h)

    return numgrad_W, numgrad_b




def ComputeGradsNum(X, Y, Weights, biases, n_layer, lamda, h):
    grad_Ws = Weights.copy()
    grad_bs = biases.copy()

    C = computeCost(X, Y, Weights, biases, n_layer, lamda)[0]


    for l in range(1, n_layer+1):
        for i in tqdm(range(biases.get(l).shape[0])):
            b_try1 = biases.get(l).copy()
            b_try1[i][0] = b_try1[i][0] + h
            biases_try = biases.copy()
            biases_try.update({l:b_try1})
            C2 = computeCost(X, Y, Weights, biases_try,n_layer, lamda)[0]
            biases_try.clear()
            test = np.zeros(grad_bs.get(l).shape)
            test[i][0] = (np.subtract(C2, C)) / h
            grad_bs.update({l: test})
        # if l ==n_layer:
        #     print("layerN: ", l)
        #     print("grad_bN: ", grad_bs.get(l))



        for i in tqdm(range(Weights.get(l).shape[0])):
            for j in range(Weights.get(l).shape[1]):
                W_try1 = Weights.get(l).copy()
                W_try1[i][j] = Weights.get(l)[i][j] + h
                weights_try = Weights.copy()
                weights_try.update({l: W_try1})
                C2 = computeCost(X, Y, weights_try, biases, n_layer, lamda)[0]
                weights_try.clear()
                testW = np.zeros(grad_Ws.get(l).shape)
                testW[i][j] = (np.subtract(C2, C)) / h
                grad_Ws.update({l: testW})


    return grad_Ws, grad_bs


def print_grad_diff(grad_w, grad_w_num, grad_b, grad_b_num):
    # diff_W, diff_b = relativeErr(grad_w, grad_b, grad_w_num, grad_b_num)
    print('Grad W:')
    # print('- relative error: {:.3e}'.format(diff_W))

    print('- sum of abs differences: {:.3e}'.format(np.abs(grad_w - grad_w_num).sum()))

    print('Grad b:')
    # print('- relative error: {:.3e}'.format(diff_b))
    print('- sum of abs differences: {:.3e}'.format(np.abs(grad_b - grad_b_num).sum()))

def checkGrad(trainX, trainY, weights, biases, n_layer, dimRed):

    gradNum_W, gradNum_b = ComputeGradsNum(trainX[:dimRed, :1000], trainY[:, :1000], weights, biases, n_layer, lamda=0, h=1e-5)
    grad_W, grad_b = ComputeGradient(trainX[:dimRed, :1000], trainY[:, :1000], weights, biases, n_layer, lamda=0)

    for l in range(1, n_layer +1):
        print("Layer", l)
        if l==n_layer:
            print("grad num: ", gradNum_b)
            print("grad: ", grad_b)
        print_grad_diff(grad_W.get(l), gradNum_W.get(l), grad_b.get(l), gradNum_b.get(l))


checkGrad(trainX, trainY, weights, biases, n_layers, dimRed)

