import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import scipy.misc




data = input_data.read_data_sets("data/fashion", one_hot=True)

data.train.cls = np.argmax(data.train.labels,axis=1)
data.test.cls = np.argmax(data.test.labels,axis=1)

img_size = 28
img_size_flat= img_size*img_size
num_channels = 1
img_shape = [img_size,img_size, num_channels]

train_dir = "/data/fashion/train"
test_dir = "data/fashion/test"

img_train = data.train.images
img_test = data.test.images

label_train = data.train.labels
label_test = data.test.labels

cls_train = data.train.cls
cls_test = data.test.cls

class_names = np.unique(cls_train)

num_classes = len(class_names)
#
# def subCat(labels, imgs):
#     count = 0
#     c= 0
#     for jdx, cls in enumerate(range(num_classes)):
#         idx = labels == cls
#         idx = np.where(labels==cls)[0]
#         xlc = imgs[idx, :]
#         print(labels[idx])
#         count += xlc.shape[0]
#         print('c: ', c)
#         print('count',count)
#
#         for i in range(c,count):
#             img = Image.fromarray(xlc[i].reshape((28, 28)))#, mode='RGB')
#             img1 = np.dstack((img, img))
#             img1 = np.dstack((img, img1))
#             scipy.misc.imsave("data/fashion/test/class{0}/image{1}.png".format(cls,i), img1)
#         c += xlc.shape[0]
#         print('xlc{0}: {1}'.format(cls, xlc.shape))
#
#

def subCat(labels, imgs):
    count = 0
    c= 0
    for jdx, cls in enumerate(range(num_classes)):
        idx = labels == cls
        idx = np.where(labels==cls)[0]
        xlc = imgs[idx, :]
        print(labels[idx])
        count += xlc.shape[0]
        print('c: ', c)
        print('count',count)

        for i in range(xlc.shape[0]):
            img = Image.fromarray(xlc[i].reshape((28, 28)))#, mode='RGB')
            img1 = np.dstack((img, img))
            img1 = np.dstack((img, img1))
            scipy.misc.imsave("data/fashion/train/class{0}/image{1}.png".format(cls,i), img1)
        c += xlc.shape[0]
        print('xlc{0}: {1}'.format(cls, xlc.shape))


subCat(cls_train, img_train)

# img1 = Image.fromarray(img_train[0].reshape((28, 28)))
# img = np.dstack((img1, img1))
# img = np.dstack((img, img1))
# print(img.shape)
# scipy.misc.imsave('Res/try.png',img)