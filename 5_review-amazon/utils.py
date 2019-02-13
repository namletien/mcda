import sys
import argparse
import os
import numpy as np
from sklearn.datasets import load_svmlight_files
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
from scipy.sparse import vstack
from keras.backend import set_session,tensorflow_backend
import tensorflow as tf 
import scipy

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))



def load_amazon(source_name, target_name, data_folder=None, verbose=False):
    if data_folder is None:
        data_folder = './data/'
    source_file = data_folder + source_name + '_train.svmlight'
    target_file = data_folder + target_name + '_train.svmlight'

    xs, ys, xt, yt = load_svmlight_files([source_file, target_file])

    # source_file = data_folder + source_name + '_train.svmlight'
    target_file = data_folder + target_name + '_test.svmlight'
    xt_test, yt_test = load_svmlight_files([target_file])

    xs = scipy.sparse.csr_matrix.todense(xs)
    xt = scipy.sparse.csr_matrix.todense(xt)
    xt_test = scipy.sparse.csr_matrix.todense(xt_test)

    ys, yt, yt_test = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, yt_test))

    return xs, ys, xt, yt, xt_test, yt_test



BATCH_SIZE = 128
TRAINING_RATIO = 5  

INPUT_SHAPE = (5000,)
mid_dim = 500
was_dim = 100
tt = 1e-5


thres = 0.8


def make_generator():
    model = Sequential()
    model.add(Dense(mid_dim, kernel_regularizer=l2(tt), input_shape = INPUT_SHAPE, activation = 'relu'))
    return model

def make_classifier():
    model = Sequential()
    model.add(Dense(2, kernel_regularizer=l2(tt), input_shape = (mid_dim,), activation='softmax'))
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
    for i in range(2):
        ind = np.where(y[:,i]>=thres)[0]
        lss.append(ind)
        us.append(len(ind))
    u = min(us)

    yz = np.zeros(y.shape)

    for i in range(2):
        ind = lss[i]
        ind = ind[:u]
        for j in ind:
            yz[j][i] = 1.
    return yz

