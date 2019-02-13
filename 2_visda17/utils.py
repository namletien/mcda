import numpy as np
import tensorflow as tf
import _pickle as pkl
from scipy.io import loadmat
from tensorflow.python.framework import ops
import sys
import argparse
import os
import numpy as np
from sklearn.datasets import load_svmlight_files
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, MaxPooling2D, GlobalMaxPooling2D, Dropout, LocallyConnected1D, GlobalAveragePooling2D
from keras.layers.merge import _Merge
from keras.utils import to_categorical
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.datasets import mnist
from keras import backend as K
from functools import partial
from scipy.sparse import vstack
from keras.backend import set_session,tensorflow_backend
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
import scipy



config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

image_size = 256
# batch_size = 256
input_shape = (2048,)
gen_rate = 1e-4
rate = 0.5
n_classes = 12
mid_dim = 500


def get_data():
    path = os.path.expanduser('~/data/visda17/reps/')

    xs_full = np.load(path +'x_train.npy')
    ys_full = np.load(path +'y_train.npy')
    xt_full = np.load(path +'x_test.npy')
    yt_full = np.load(path +'y_test.npy')

    ys_full = to_categorical(ys_full, n_classes)
    yt_full = to_categorical(yt_full, n_classes)

    xs_full, ys_full = shuffle_data2(xs_full,ys_full)
    xt_full, yt_full = shuffle_data2(xt_full, yt_full)
    return xs_full, ys_full, xt_full, yt_full

def test_target(xt_full,yt_full, cls_model):
    acc = np.zeros(n_classes+1)
    yt_pred = cls_model.predict(x = xt_full)
    yt_p = np.argmax(yt_pred, axis = 1)
    yt_f = np.argmax(yt_full, axis = 1)

    acc[n_classes] = np.sum(yt_p==yt_f)/len(yt_f)
    for i in range(n_classes):
        acc[i] = np.sum((yt_p==yt_f)*(yt_f==i))/sum(yt_f==i)
    # print('test accuracy: ', acc * 100)
    return acc * 100


def shuffle_data2(X,y):
        assert len(X) == len(y)
        Q =np.asarray(range(len(y)))
        ind = np.random.permutation(Q)
        return X[ind], y[ind]
        

def make_generator():
    model = Sequential()
    model.add(Dense(1024,  input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(rate))
    model.add(Dense(mid_dim))
    model.add(BatchNormalization())
    model.add(Dropout(rate))
    model.add(LeakyReLU())
    return model

def make_classifier():
    model = Sequential()
    model.add(Dense(200,  input_shape = (mid_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(rate))
    model.add(Dense(12,  activation='softmax'))
    return model

def convert (y, thres):
    lss = []
    us = []
    for i in range(n_classes):
        ind = np.where(y[:,i]>=thres)[0]
        lss.append(ind)
        us.append(len(ind))
    u = min(us)
    # if u>0:
    #     print(u)

    yz = np.zeros(y.shape)

    for i in range(n_classes):
        ind = lss[i]
        v = min(u + 2, len(ind))
        ind = ind[:v]
        for j in ind:
            yz[j][i] = 1.
    # if sum(sum(yz))>0:
    #     print(sum(sum(yz)))
    return yz
