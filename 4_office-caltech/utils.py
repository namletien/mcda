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



BATCH_SIZE = 128
TRAINING_RATIO = 5  # The training ratio is the number of wasserstein updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
INPUT_SHAPE = (4096,)
mid_dim = 500
rate = 0.5
was_dim = 100
gen_rate = 1e-4
was_rate = 5*1e-4
thres = 0.9
epochs = 30



def make_generator():
    model = Sequential()
    model.add(Dense(1024,  input_shape = INPUT_SHAPE))
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
    model.add(Dense(100,  input_shape = (mid_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(rate))
    model.add(Dense(10,  activation='softmax'))
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


def load_office(source_name, target_name, data_dir):
    source_file = data_dir + source_name + '.mat'
    target_file = data_dir + target_name + '.mat'
    source = loadmat(source_file)
    target = loadmat(target_file)
    xs = source['fts']
    ys = source['labels']
    xt = target['fts']
    yt = target['labels']
    return xs, ys, xt, yt




class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()


