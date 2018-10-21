import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt



np.random.seed(400)
trainPath = 'Datasets/cifar-10-batches-mat/data_batch_1.mat'
valPath = 'Datasets/cifar-10-batches-mat/data_batch_2.mat'
testPath = 'Datasets/cifar-10-batches-mat/test_batch.mat'

def LoadBatch(filename):
    enc =OneHotEncoder()

    Data = loadmat(filename)

    Im = Data['data'].transpose()
    scaler = MinMaxScaler()
    scaler.fit(Im)
    scaledIm = scaler.transform(Im)

    Label = Data['labels']
    fit = enc.fit(Label)
    labelsEnc = enc.transform(Label).toarray().transpose()

    return scaledIm, labelsEnc, Label


trainX, trainY,trainy = LoadBatch(trainPath)
N = trainX.shape[1]
d = trainX.shape[0]
K = trainY.shape[0]
valX, valY,valy = LoadBatch(valPath)
testX, testY,testy = LoadBatch(testPath)
W = np.random.normal(loc=0, scale=0.01, size=(K,d))
b = np.random.normal(loc=0, scale=0.01, size=(K,1))

# print('X: ',trainX.shape)
# print('Y: ',trainY.shape)
# print('W: ',W.shape)
# print('b: ',b.shape)


def EvaluateClassifer(X,W,b):
    WX = np.dot(W,X)
    S = np.add(WX, b)
    P = np.exp(S)/np.sum(np.exp(S))

    return P

P=EvaluateClassifer(trainX, W, b)#trainX[:, 1:100],W,b)
# print('hot encoded Y: ',trainY.shape)
#

def ComputeCost(X, Y, W, b, labda):
    finCross= 0
    for i in range(X.shape[1]):
        P = EvaluateClassifer(np.asarray(X[:,i]).reshape(X.shape[0],1), W,b)
        lCross = np.dot(np.asarray(Y[:,i]).T, P)
        logCross = -np.log(lCross)
        finCross += logCross
    # print('fc1: ',finCross)
    # P = EvaluateClassifer(X,W,b)
    # lcross = -np.log(np.dot(Y.T, P))
    # finCross = np.sum(np.diag(lcross))
    # print('fc2: ',finCross)
    expr1 = (1/X.shape[1])*finCross
    sqW = np.square(W)
    sqW = np.sum(sqW)
    expr2 = labda*sqW
    J = expr1+expr2
    return J, expr1

J = ComputeCost(trainX, trainY, W,b,1e-6)
# print('J: ',J)
def ComputeAccuracy(X, y, W, b):
    P = EvaluateClassifer(X,W,b)
    predClass = np.argmax(P, 0)

    diff = np.subtract(predClass,np.asarray(y).T)
    incorrect = np.count_nonzero(diff)
    acc = 1-(incorrect/len(y))
    #print(acc)
    return acc*100

acc = ComputeAccuracy(trainX,trainy,W,b)
# print(acc)

def ComputeGradients(X,Y,P,W, lamda):
    grad_W = np.zeros((Y.shape[0], X.shape[0]))
    grad_b = np.zeros((Y.shape[0], 1))
    Y = np.asanyarray(Y)

    for i in range(X.shape[1]):
        P = EvaluateClassifer(np.asarray(X[:,i]).reshape(X.shape[0], 1),W,b)
        g = -np.asarray(np.subtract(np.asarray(Y[:,i]).reshape(Y.shape[0],1) , P)).T
        # gFirst= -(Y[:,i].T/np.dot(Y[:,i].T,P))
        # gSecond = np.diag(P)-np.outer(P,P.T)
        # g = np.dot(gFirst,gSecond)
        # g = np.reshape(g,(grad_b.shape))
        grad_b += g.T
        grad_W += np.dot(np.asarray(g).T, np.asarray(X[:,i]).reshape(X.shape[0], 1).T)
    grad_W /= X.shape[1]
    grad_b /= X.shape[1]
    grad_W += 2*lamda*W
    return grad_W, grad_b

grad_W, grad_b = ComputeGradients(trainX,trainY,P, W, 0)
# print(grad_W)
# print(grad_b)


def MiniBatchGD(X,Y,y,GDparams, W, b, lamda):
    Xbatch = []
    Ybatch = []
    Wstar = W.copy()
    bstar = []
    losses = []
    costs = []
    WStars = []
    bStars =[]
    for i in range(GDparams[2]):
        minibatch=int(X.shape[1]/GDparams[0])
        for j in range(1,int(X.shape[1]/GDparams[0])):
            jstart = (j - 1) * GDparams[0] + 1
            jend = j * GDparams[0]
            Xbatch = X[:, jstart: jend]
            Ybatch = Y[:, jstart: jend]
            deltaW, deltab = ComputeGradients(Xbatch,Ybatch,P,Wstar,lamda)

            Wstar = W - GDparams[1]*deltaW
            bstar = b - GDparams[1]*deltab
            W = Wstar.copy()
            b = bstar.copy()
            # print(ComputeCost(Xbatch,Ybatch,Wstar,bstar,lamda))
        cost,loss = ComputeCost(X,Y,W,b,lamda)
        costs.append(cost)
        losses.append(loss)

        W = Wstar.copy()
        b = bstar.copy()
        WStars.append(Wstar)
        bStars.append(bstar)
    acc = ComputeAccuracy(X, y, W, b)

    return WStars, bStars,costs, losses, acc

def costVal(X,Y,y,GDparams, W, b, lamda):
    Wstar = W.copy()
    bstar = []
    losses = []
    costs = []
    acc = 0
    for i in range(GDparams[2]):
        cost,loss = ComputeCost(X,Y,W[i],b[i],lamda)
        acc = ComputeAccuracy(X, y, W[i], b[i])
        costs.append(cost)
        losses.append(loss)

    return costs, losses, acc


def weightImage(W):
    for i, row in enumerate(W):
        img = (row - row.min()) / (row.max() - row.min())
        plt.subplot(2, 5, i+1)
        squared_image = np.rot90(np.reshape(img, (32, 32, 3), order='F'), k=3)
        plt.imshow(squared_image)
        plt.axis('off')
    plt.show()

# weightImage(W)
# print(W.shape)
GDparams = [100, 0.1, 40]
lamda =0
Wstar, bstar, costsTrain, lossesTrain, accTrain = MiniBatchGD(trainX,trainY,trainy,GDparams,W,b,lamda)
costsVal, lossesVal,accVal = costVal(valX,valY,valy,GDparams,Wstar,bstar, lamda)#MiniBatchGD(valX,valY,valy,[100, 0.01, 40],Wstar,bstar,0)
plt.plot(range(GDparams[2]), costsTrain, color='g',label = 'train')
plt.plot(range(GDparams[2]), costsVal, color='r', label = 'val')
plt.plot(range(GDparams[2]), lossesTrain, color='y',label = 'train loss')
plt.plot(range(GDparams[2]), lossesVal, color='b', label = 'val loss')

plt.legend()
plt.show()

# print('W shape:',Wstar[-1].shape)
weightImage(np.asarray(Wstar[-1]))
print(accTrain,'%')
print(accVal,'%')


def ComputeGradsNum(X,Y,W,b, lamda,h):
    grad_W = np.zeros((Y.shape[0], X.shape[0]))
    grad_b = np.zeros((Y.shape[0], 1))
    c = ComputeCost(X, Y, W, b, lamda)

    for i in range(b.shape[0]):
        b_try = b.copy()
        b_try[i] = b_try[i] + h
        c2 = ComputeCost(X, Y, W, b_try, lamda )
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W.copy()
            W_try[i][j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda )
            grad_W[i][j] = (c2-c) / h
        print(i)
    return grad_W, grad_b


def print_grad_diff(grad_w, grad_w_num, grad_b, grad_b_num):
    print('Grad W:')
    print('- sum of abs differences: {:.3e}'.format(np.abs(grad_w - grad_w_num).sum()))

    print('Grad b:')
    print('- sum of abs differences: {:.3e}'.format(np.abs(grad_b - grad_b_num).sum()))

# grad_W, grad_b = ComputeGradients(trainX,trainY,P,W,0)
# grad_W_n, grad_b_n = ComputeGradsNum(trainX, trainY, W, b, 1e-6)

# print_grad_diff(grad_W, grad_W_N, grad_b, grad_b_N)


