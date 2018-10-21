import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

data = input_data.read_data_sets("data/fashion", one_hot=True)

trainIm = data.train.images
trainL = data.train.labels
nTrain = data.train.num_examples
nClass = 10

testIm = data.test.images
testL = data.train.labels
nTest = data.test.num_examples

inputShape = (224,224)
batchSize = 90
epochs = 30
stepsPerEpoch =1000
testSteps = nTest/batchSize

def arch(nClass, inputShape, finetune=False):
    cnnArch= VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))

    for layer in cnnArch.layers:
        layer.trainable = False
        print('{0}:{1}'.format(layer.trainable, layer.name))

    if (finetune == True):
        for layer in cnnArch.layers:
            if layer.name=='block5_conv1' or layer.name=='block5_conv2' or layer.name=='block5_conv3' or layer.name=='block5_pool' or layer.name=='block4_conv1' or layer.name=='block4_conv2' or layer.name=='block4_conv3' or layer.name=='block4_pool':
                layer.trainable = True
                print('{0}:{1}'.format(layer.trainable, layer.name))
            else:
                layer.trainable = False
                print('{0}:{1}'.format(layer.trainable, layer.name))


    top = Sequential()
    top.add(cnnArch)
    top.add(Flatten())
    top.add(Dense(units=1024, activation='relu'))
    top.add(Dropout(0.5))
    top.add(Dense(units=nClass, activation='softmax'))

    return top

def dataGeneration(inputShape, batchSize):
    trainGen = ImageDataGenerator(rescale=1./255)
    testGen = ImageDataGenerator(rescale=1./255)

    trainBatches = trainGen.flow_from_directory(directory='data/fashion/train',target_size=inputShape,batch_size=batchSize,shuffle=False)
    testBatches = testGen.flow_from_directory(directory='data/fashion/test', target_size=inputShape, batch_size=batchSize,shuffle=False)

    return trainBatches,testBatches

def plotRes(trainLoss, trainAcc, testLoss, testAcc, epochs, finetune=False):
    plt.figure()
    plt.plot(trainAcc,label='train acc.')#, epochs)
    plt.plot(testAcc,'g',label ='test acc.')#, epochs,'g')
    plt.legend()
    if finetune==True:
        plt.savefig('Res/Fine-tune/accNewest')
        plt.show()
    else:
        plt.savefig('Res/Transfer-Learn/accNewest')
        plt.show()
    plt.figure()
    plt.plot(trainLoss, label='train loss')#,epochs)
    plt.plot(testLoss,'g', label='test loss')#epochs)
    plt.legend()
    if finetune==True:
        plt.savefig('Res/Fine-tune/lossNewest')
        plt.show()
    else:
        plt.savefig('Res/Transfer-Learn/lossNewest')
        plt.show()

    plt.figure()
    plt.plot(trainAcc, label='train acc.')
    plt.plot(testAcc,'g', label='test acc.')
    plt.plot(trainLoss,'bo', label='train loss')
    plt.plot(testLoss,'go', label='test loss')
    plt.legend()
    if finetune==True:
        plt.savefig('Res/Fine-tune/togetherNewest')
        plt.show()
    else:
        plt.savefig('Res/Transfer-Learn/togetherNewest')
        plt.show()

def transferL():

    vgg16TL = arch(nClass, inputShape)
    vgg16TL.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    trainGenerator, testGenerator = dataGeneration(inputShape,batchSize)

    resTrain = vgg16TL.fit_generator(trainGenerator,steps_per_epoch=stepsPerEpoch,epochs=epochs, validation_data=testGenerator,validation_steps=testSteps)
    resTest = vgg16TL.evaluate_generator(generator=testGenerator,steps=testSteps)
    trainLoss = resTrain.history['loss']
    trainAcc = resTrain.history['categorical_accuracy']

    testLoss = resTrain.history['val_loss']
    testAcc = resTrain.history['val_categorical_accuracy']

    plotRes(trainLoss,trainAcc,testLoss,testAcc, epochs)
    print('resTrain: ',resTrain.history['categorical_accuracy'])
    print('Test loss: ',resTest[0])
    print('Test acc:', resTest[1])





def plotResFineTune(trainLoss, trainAcc, testLoss, testAcc, epochs):

    plt.figure()
    plt.plot(trainAcc,label='train acc.')#, epochs)
    plt.plot(testAcc,'g',label ='test acc.')#, epochs,'g')
    plt.legend()
    plt.savefig('Res/Fine-tune/accNewest')
    plt.show()

    plt.figure()
    plt.plot(trainLoss, label='train loss')#,epochs)
    plt.plot(testLoss,'g', label='test loss')#epochs)
    plt.legend()
    plt.savefig('Res/Fine-tune/lossNewest')
    plt.show()

    plt.figure()
    plt.plot(trainAcc, label='train acc.')
    plt.plot(testAcc,'g', label='test acc.')
    plt.plot(trainLoss,'bo', label='train loss')
    plt.plot(testLoss,'go', label='test loss')
    plt.legend()
    plt.savefig('Res/Fine-tune/togetherNewest')
    plt.show()







def finetune():
    vgg16TL = arch(nClass, inputShape, finetune=True)
    vgg16TL.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    trainGenerator, testGenerator = dataGeneration(inputShape,batchSize)

    resTrain = vgg16TL.fit_generator(trainGenerator,steps_per_epoch=stepsPerEpoch,epochs=epochs, validation_data=testGenerator,validation_steps=testSteps)
    resTest = vgg16TL.evaluate_generator(generator=testGenerator,steps=testSteps)
    trainLoss = resTrain.history['loss']
    trainAcc = resTrain.history['categorical_accuracy']

    testLoss = resTrain.history['val_loss']
    testAcc = resTrain.history['val_categorical_accuracy']

    plotResFineTune(trainLoss,trainAcc,testLoss,testAcc, epochs)#, finetune=True)
    print('resTrain: ',resTrain.history['categorical_accuracy'])
    print('Test loss: ',resTest[0])
    print('Test acc:', resTest[1])

transferL()
#finetune()