import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.matlib as matlib
import math

np.random.seed(400)
trainPath = 'Datasets/cifar-10-batches-mat/data_batch_1.mat'
valPath = 'Datasets/cifar-10-batches-mat/data_batch_2.mat'
testPath = 'Datasets/cifar-10-batches-mat/test_batch.mat'

nrNode = 50


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


def initVariables(X, Y, m):
    N = X.shape[1]
    d = X.shape[0]
    K = Y.shape[0]

    W1 = np.random.normal(loc=0, scale=0.001, size=(m, d))
    W2 = np.random.normal(loc=0, scale=0.001, size=(K, m))
    b1 = np.zeros((m, 1))
    b2 = np.zeros((K, 1))
    return W1, W2, b1, b2


W1, W2, b1, b2 = initVariables(trainX, trainY, 50)


def EvaluateClassifier(X, W1, b1, W2, b2):
    s1 = np.dot(W1, X) + b1
    h = np.zeros(s1.shape)
    for i in range(s1.shape[0]):
        for j in range(s1.shape[1]):
            h[i][j] = np.maximum(0, s1[i][j])
    s = np.dot(W2, h) + b2
    p = np.exp(s) / np.sum(np.exp(s))
    return h, p


#
#
#
# def computeCost(X, Y, W1, W2, b1, b2, lamda):
#     finCross = 0
#     for i in range(X.shape[1]):
#         h, P = EvaluateClassifier(np.asarray(X[:, i]).reshape(X.shape[0], 1), W1, b1, W2, b2)
#         lCross = np.dot(np.asarray(Y[:, i]).T, P)
#         logCross = -np.log(lCross)
#         finCross += logCross
#     # print('fc1: ',finCross)
#     # P = EvaluateClassifer(X,W,b)
#     # lcross = -np.log(np.dot(Y.T, P))
#     # finCross = np.sum(np.diag(lcross))
#     # print('fc2: ',finCross)
#     expr1 = (1 / X.shape[1]) * finCross
#
#     sqW1 = np.square(W1)
#     sqW1 = np.sum(sqW1)
#
#     sqW2 = np.square(W2)
#     sqW2 = np.sum(sqW2)
#
#     sqW = sqW2 + sqW1
#
#     expr2 = lamda * sqW
#     J = expr1 + expr2
#     return J, expr1
#

def ComputeGradients(X, Y, W1, W2, b1, b2, lamda):
    m = W1.shape[0]
    d = W1.shape[1]
    K = W2.shape[0]
    grad_W1 = np.zeros((m, d))
    grad_W2 = np.zeros((K, m))
    grad_b1 = np.zeros((m, 1))
    grad_b2 = np.zeros((K, 1))
    Y = np.asanyarray(Y)

    for i in range(X.shape[1]):
        h, P = EvaluateClassifier(np.asarray(X[:, i]).reshape(X.shape[0], 1), W1, b1, W2, b2)
        g = -np.asarray(np.subtract(np.asarray(Y[:, i]).reshape(Y.shape[0], 1), P)).T

        grad_b2 += g.T
        grad_W2 += np.dot(np.asarray(g).T, h.T)

        g = np.dot(g, W2)

        newS1 = []
        s1 = h
        for indexi in range(s1.shape[0]):
            if (s1[indexi][0] > 0):
                newS1.append(1)
            else:
                newS1.append(0)

        newS1 = np.asarray(newS1)
        newS1 = np.diagflat(newS1)
        g = np.dot(g, newS1)

        grad_b1 += g.T
        grad_W1 += np.dot(g.T, np.asarray(X[:, i]).reshape(X.shape[0], 1).T)

    grad_W1 /= X.shape[1]
    grad_b1 /= X.shape[1]

    grad_W2 /= X.shape[1]
    grad_b2 /= X.shape[1]

    grad_W1 += 2 * lamda * W1
    grad_W2 += 2 * lamda * W2

    return grad_W1, grad_W2, grad_b1, grad_b2




# def ComputeGradsNum(X, Y, W1, W2, b1, b2, lamda, h):
#     grad_W1 = np.zeros(W1.shape)
#     grad_W2 = np.zeros(W2.shape)
#
#     grad_b1 = np.zeros(b1.shape)
#     grad_b2 = np.zeros(b2.shape)
#
#     C = computeCost(X, Y, W1, W2, b1, b2, lamda)[0]
#
#     for i in tqdm(range(b1.shape[0])):
#         b_try1 = b1.copy()
#         b_try1[i][0] = b_try1[i][0] + h
#         C2 = computeCost(X, Y, W1, W2, b_try1, b2, lamda)[0]
#         grad_b1[i][0] = (np.subtract(C2, C)) / h
#
#     for i in tqdm(range(b2.shape[0])):
#         b_try2 = b2.copy()
#         b_try2[i][0] = b_try2[i][0] + h
#         C2 = computeCost(X, Y, W1, W2, b1, b_try2, lamda)[0]
#         grad_b2[i][0] = (np.subtract(C2, C)) / h
#
#     for i in tqdm(range(W1.shape[0])):
#         for j in range(W1.shape[1]):
#             W_try1 = W1.copy()
#             W_try1[i][j] = W_try1[i][j] + h
#             C2 = computeCost(X, Y, W_try1, W2, b1, b2, lamda)[0]
#             grad_W1[i][j] = (np.subtract(C2, C)) / h
#
#     for i in tqdm(range(W2.shape[0])):
#         for j in range(W2.shape[1]):
#             W_try2 = W2.copy()
#             W_try2[i][j] = W_try2[i][j] + h
#             C2 = computeCost(X, Y, W1, W_try2, b1, b2, lamda)[0]
#             grad_W2[i][j] = (np.subtract(C2, C)) / h
#
#     return grad_W1, grad_W2, grad_b1, grad_b2
#
#
# # reduding dimensionality of input vector
# dimRed = 2
# h = 1e-5
#
#
# # lamda = 0
#
#
# # ComputeGradsNum(trainX[0:100,:], trainY, W1[:,0:100], W2, np.asarray(b1), np.asarray(b2), lamda, h)
#
#
# def print_grad_diff(grad_w, grad_w_num, grad_b, grad_b_num):
#     print('Grad W:')
#     print('- sum of abs differences: {:.3e}'.format(np.abs(grad_w - grad_w_num).sum()))
#
#     print('Grad b:')
#     print('- sum of abs differences: {:.3e}'.format(np.abs(grad_b - grad_b_num).sum()))
#
#
# # grad_W, grad_W2, grad_b, grad_b2 = ComputeGradients(trainX[0:dimRed,:],trainY[0:dimRed,:],W1[:,0:dimRed],W2,b1,b2,lamda)
# # grad_W1_n, grad_W2_n, grad_b1_n, grad_b2_n = ComputeGradsNum(trainX[0:dimRed,:], trainY[0:dimRed,:], W1[:,0:dimRed],W2, b1,b2,lamda, h)
#
# # print("diff for W1 and b1:\n ")
# # print_grad_diff(grad_W, grad_W1_n, grad_b, grad_b1_n)
# # print("diff for W2 and b2:\n ")
# # print_grad_diff(grad_W2, grad_W2_n, grad_b2, grad_b2_n)
#
# def measureErr(X, Y, y, W1, W2, b1, b2):
#     _, P = EvaluateClassifier(X, W1, b1, W2, b2)
#     evalValues = []
#     for i in range(P.shape[1]):
#         evalValues.append(np.argmax(P[:, i]))
#
#     evalValues = np.reshape(evalValues, (len(evalValues), 1))
#     all = np.subtract(y, evalValues)
#     incorrect = np.nonzero(all)
#     err = (np.asarray(incorrect).shape[1] / y.shape[0])
#     acc = (1 - (np.asarray(incorrect).shape[1] / y.shape[0])) * 100
#     return err, acc
#
#
# def miniBatch(X, Y, n_batch, eta, W1, W2, b1, b2, lamda, rho):
#     W1_momentum = np.zeros(W1.shape)
#     W2_momentum = np.zeros(W2.shape)
#
#     b1_momentum = np.zeros(b1.shape)
#     b2_momentum = np.zeros(b2.shape)
#     for j in range(1, int(X.shape[1] / n_batch)):
#         jstart = (j - 1) * n_batch + 1
#         jend = j * n_batch
#         Xbatch = X[:, jstart: jend]
#         Ybatch = Y[:, jstart: jend]
#
#         grad_W1, grad_W2, grad_b1, grad_b2 = ComputeGradients(Xbatch, Ybatch, W1, W2, b1, b2, lamda)
#
#         W1_momentum = rho * W1_momentum + eta * grad_W1
#         W2_momentum = rho * W2_momentum + eta * grad_W2
#
#         b1_momentum = rho * b1_momentum + eta * grad_b1
#         b2_momentum = rho * b2_momentum + eta * grad_b2
#
#         W1 = W1 - W1_momentum
#         W2 = W2 - W2_momentum
#
#         b1 = b1 - b1_momentum
#         b2 = b2 - b2_momentum
#
#         # W1 = W1 - eta*grad_W1
#         # W2 = W2 - eta*grad_W2
#         #
#         # b1 = b1 - eta*grad_b1
#         # b2 = b2 - eta*grad_b2
#
#     return W1, W2, b1, b2
#
#
# ###GDParams = (n_batch, eta, epoch)
# def miniBatchGD(X, Y, y, W1, W2, b1, b2, GDParams, lamda, rho, decay_rate=0.95):
#     costs = []
#     errors = []
#
#     W1_Abatch = W1.copy()
#     W2_Abatch = W2.copy()
#
#     b1_Abatch = b1.copy()
#     b2_Abatch = b2.copy()
#     for i in range(GDParams[2]):
#         err, _ = measureErr(X, Y, y, W1_Abatch, W2_Abatch, b1_Abatch, b2_Abatch)
#         errors.append(err)
#         costs.append(computeCost(X, Y, W1_Abatch, W2_Abatch, b1_Abatch, b2_Abatch, lamda)[0])
#         W1_Abatch, W2_Abatch, b1_Abatch, b2_Abatch = miniBatch(X, Y, GDParams[0], GDParams[1], W1_Abatch, W2_Abatch,
#                                                                b1_Abatch, b2_Abatch, lamda, rho)
#         GDParams[1] *= decay_rate
#
#     return np.asarray(errors), np.asarray(costs), W1_Abatch, W2_Abatch, b1_Abatch, b2_Abatch
#
#
# def findEta(trainX, trainY, trainy, W1, W2, b1, b2, GDParams, lamda, rho, slice, etas):
#     for eta in etas:
#         GDParams[1] = eta
#         print("eta: ", eta)
#         errs, costs1, nW1, nW2, nb1, nb2 = miniBatchGD(trainX, trainY, trainy, W1,
#                                                        W2,
#                                                        b1, b2, GDParams, lamda, rho)
#
#         print("eta: ", eta)
#         print("cost for eta: ", costs1)
#
#
# #         label = "training loss with eta {0:.8f}".format(eta)
# #         plt.plot(range(GDParams[2]), costs1, label=label)
# #
# #     plt.legend()
# #     plt.show()
#
#
# def validate(trainX, trainY, trainy, valX, valY, valy, W1, W2, b1, b2, GDParams, rho, slice, lamdas):
#     e_min = np.log(0.02)
#     e_max = np.log(0.055)
#     for i in range(5):
#         e = e_min + (e_max - e_min) * np.random.random(1)[0]
#         eta = np.exp(e)
#         GDParams[1] = eta
#         for lamda in lamdas:
#             errs, costs1, nW1, nW2, nb1, nb2 = miniBatchGD(trainX, trainY, trainy,
#                                                            W1, W2,
#                                                            b1, b2, GDParams, lamda, rho)
#             # errs, costs1, nW1, nW2, nb1, nb2 = miniBatchGD(trainX[:, 0:slice], trainY[:, 0:slice], trainy[0:slice, :],
#             #                                                            W1, W2,
#             #                                                            b1, b2, GDParams, lamda, rho)
#             ErrT, acc = measureErr(valX, valY, valy, nW1, nW2, nb1, nb2)
#
#             Err, accTrain = measureErr(trainX, trainY, trainy, nW1, nW2, nb1, nb2)
#             print("performance on training set {0:.1f}%".format(accTrain))
#             print("eta: {0:.7f}     lamda: {1:.7f}".format(eta, lamda))
#             # print("eta: {0:.7f}     lamda: {1:.7f}".format(GDParams[1], lamda))
#             print("performance on validation set {0:.1f}%".format(acc))
#             print("train errror: {0: .1f}	validation error: {1: .1f}".format(ErrT, Err))
#             print("train loss:", costs1)
#
#
# # test gradient by checking by training the network, results to overfitting the training data and getting low loss
# GDParams = [100, 0.0294242, 30]
# #lamda = 0.000001
# rho = 0.9
# slice = 1000
# lamda = 0.0001  # [0.0001, 0.0003,0.0005, 0.0007 ,0.0009,0.001, 0.002, 0.003, 0.004, 0.005]  # [0.01,0.02,0.03,0.04,0.05, 0.06, 0.07, 0.08,0.09,0.1]
#
#
# # validate(trainX, trainY, trainy, valX, valY, valy, W1, W2, b1, b2, GDParams, rho, slice, lamda)
#
#
# def testAllBatches(X, Y, y, valX, valY, valy, testX, testY, testy, W1, W2, b1, b2, GDParams, rho, lamda):
#     errs, costs, nW1, nW2, nb1, nb2 = miniBatchGD(X[0], Y[0], y[0],
#                                                   W1, W2,
#                                                   b1, b2, GDParams, lamda, rho)
#     for i in range(1, len(X)):
#         errs, costs, nW1, nW2, nb1, nb2 = miniBatchGD(X[i], Y[i], y[i],
#                                                       nW1, nW2, nb1, nb2, GDParams, lamda, rho)
#
#     errs, costs1, nW1, nW2, nb1, nb2 = miniBatchGD(valX[:, 0:1000], valY[:, 0:1000], valy[0:1000, :],
#                                                    nW1, nW2, nb1, nb2, GDParams, lamda, rho)
#
#     _,acc = measureErr(testX, testY, testy, nW1, nW2, nb1, nb2)
#
#     print("test accuracy: ", acc)
#     print("training costs: ", costs)
#     print("validation costs: ", costs1)
#     label = "training cost"
#     plt.figure()
#     plt.plot(range(GDParams[2]), costs, label=label)
#     label1 = "validation cost"
#     plt.plot(range(GDParams[2]), costs1, label=label1)
#
#     plt.legend()
#     plt.savefig("trainValCosts2.png")
#     plt.show()
#     _, acc = measureErr(testX, testY, testy, nW1, nW2, nb1, nb2)
#     print("test accuracy: ", acc)
#
#
# train1X, train1Y, train1y = LoadBatch('Datasets/cifar-10-batches-mat/data_batch_3.mat')
# train2X, train2Y, train2y = LoadBatch('Datasets/cifar-10-batches-mat/data_batch_4.mat')
# train3X, train3Y, train3y = LoadBatch('Datasets/cifar-10-batches-mat/data_batch_5.mat')
#
# trainXBatch = [trainX, train1X, train2X, train3X]
# trainYBatch = [trainY, train1Y, train2Y, train3Y]
# trainyBatch = [trainy, train1y, train2y, train3y]
#
# testAllBatches(trainXBatch, trainYBatch, trainyBatch,valX,valY,valy, testX, testY, testy, W1, W2, b1, b2, GDParams, rho, lamda)
#
#
#
# # etas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
#
# # findEta(trainX, trainY, trainy, W1, W2, b1, b2, GDParams, lamda, rho, slice, etas)
#
#
# # check that the loss is initially 2.3
# # print(computeCost(trainX[:, 0:100], trainY[:, 0:100], W1, W2, b1, b2, lamda))
#
# # errs, costs, nW1, nW2, nb1, nb2 = miniBatchGD(trainX[:, 0:1000], trainY[:, 0:1000], trainy[0:1000,:], W1, W2, b1, b2, GDParams, lamda, rho)
#
# # plt.plot(range(GDParams[2]), costs,color = 'y', label= 'loss with eta 0.05')
#
# # GDParams = [10, 0.01, 10]
#
#
# # errs, costs1, nW1, nW2, nb1, nb2 = miniBatchGD(trainX, trainY, trainy, W1, W2, b1, b2, GDParams, lamda, rho)
# # print("cost: ", costs1)
# # _, accTrain = measureErr(trainX, trainY, trainy, nW1, nW2, nb1, nb2)
# # print("training acc: ",accTrain)
# # _, acc = measureErr(valX, valY, valy, nW1, nW2, nb1, nb2)
# # print("validation acc: ",acc)
# #
#
#
# # plt.plot(range(GDParams[2]), costs1, color ='g', label= 'loss with eta 0.01')
# #
# # plt.legend()
# # plt.show()
#
#
