import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, MaxPooling2D, GlobalMaxPooling2D, Dropout, LocallyConnected1D
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

from keras.backend import set_session,tensorflow_backend
import tensorflow as tf 

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

gp_weight = 10
bt_size = 128
ip_shape = (28,28,1)
n_classes = 10

bn_axis = -1
mid_dim = 1024
rate = 0
tt = 1e-5
gen_rate = 1e-4

thres = 0.9
epochs = 100
steps = 200

def data_usps_mnist(mnist_source =True):
    path = os.path.expanduser('~/data/usps/')
    if mnist_source == True:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
        X_train = X_train.astype(np.float32) / 255
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        X_test = X_test.astype(np.float32) / 255
        xs = X_train
        ys = to_categorical(y_train, 10)
        xs_test = X_test
        ys_test = to_categorical(y_test, 10)

        xt = np.load(path+'usps_X_train.npy')
        xt_test = np.load(path+'usps_X_test.npy')
        xt = xt.reshape(xt.shape[0],28,28,1)
        xt_test = xt_test.reshape(xt_test.shape[0],28,28,1)
        yt = np.load(path+'usps_y_train.npy')
        yt_test = np.load(path+'usps_y_test.npy')
    else:

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
        X_train = X_train.astype(np.float32) / 255
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        X_test = X_test.astype(np.float32) / 255
        xt = X_train
        yt = to_categorical(y_train, 10)
        xt_test = X_test
        yt_test = to_categorical(y_test, 10)

        xs = np.load(path+'usps_X_train.npy')
        xs = xs.reshape(xs.shape[0],28,28,1)
        ys = np.load(path+'usps_y_train.npy')
    return xs, ys, xt, xt_test, yt_test


def make_generator():
    model = Sequential()

    model.add(Convolution2D(32,(5,5), strides = 1, padding = 'same', kernel_regularizer=l2(tt), input_shape = ip_shape))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2,2), padding = 'same'))

    model.add(Convolution2D(64, (5, 5), strides = 1, kernel_regularizer=l2(tt), padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2,2), padding = 'same'))

    model.add(Flatten())
    model.add(Dense(mid_dim))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Dropout(rate))
    return model

def make_classifier():
    model = Sequential()
    # model.add(Dense(256, kernel_regularizer=l2(tt), input_shape = (mid_dim,)))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # model.add(Dropout(rate))
    model.add(Dense(n_classes, kernel_regularizer=l2(tt), activation='softmax', input_shape = (mid_dim,)))
    return model


def shuffle_data(X):
        Q =np.asarray(range(len(X)))
        ind = np.random.permutation(Q)
        return X[ind]
def shuffle_data2(X,y):
        assert len(X) == len(y)
        Q =np.asarray(range(len(y)))
        ind = np.random.permutation(Q)
        return X[ind], y[ind]

def convert (y):
    lss = []
    us = []
    for i in range(10):
        ind = np.where(y[:,i]>=thres)[0]
        lss.append(ind)
        us.append(len(ind))
    u = min(us)

    yz = np.zeros(y.shape)

    for i in range(10):
        ind = lss[i]
        ind = ind[:u]
        for j in ind:
            yz[j][i] = 1.
    return yz
