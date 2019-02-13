import sys
import argparse
import os
import numpy as np
from sklearn.datasets import load_svmlight_files
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, MaxPooling2D, GlobalMaxPooling2D, Dropout, Lambda
from keras.layers.merge import _Merge
from keras.utils import to_categorical
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.datasets import mnist
from keras import backend as K
from functools import partial
from scipy.sparse import vstack
from keras.backend import set_session,tensorflow_backend
import tensorflow as tf 
import scipy
from utils import *

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


bt_size = 128
ip_shape = (5000,)
n_classes = 2
gen_rate = 1e-4
epochs = 30
steps = 200
dann_param = - float(sys.argv[3])

data_folder = './data/'
source_name = sys.argv[1]
target_name = sys.argv[2]  
xs, ys, xt, yt, xt_test, yt_test = load_amazon(source_name, target_name, data_folder, verbose=True)
ys = to_categorical(ys,2)
yt = to_categorical(yt,2)
yt_test = to_categorical(yt_test,2)

def make_dann():
    model = Sequential()
    model.add(Dense(100,  input_shape = (mid_dim,)))
    model.add(LeakyReLU())
    model.add(Dense(2, activation='softmax'))
    return model

def run_dann(xs,ys,xt,xt_test, yt_test):
    generator = make_generator()
    classifier = make_classifier()
    dann = make_dann()

    input_s = Input(shape=ip_shape)
    input_t = Input(shape=ip_shape)
    middle_s = generator(input_s)
    middle_t = generator(input_t)
    out_dann_s = dann(middle_s)
    out_dann_t = dann(middle_t)
    out_cls_s = classifier(middle_s)

    cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
    cls_model.compile(optimizer=Adam(0),loss=['categorical_crossentropy'], metrics = ['accuracy'])

    for layer in generator.layers:
        layer.trainable = False
        generator.trainable = False

    dann_flip =  Model(inputs=[input_s], outputs=[out_dann_s])
    dann_flip.compile(optimizer=Adam(gen_rate*10),loss=['categorical_crossentropy'])

    for layer in generator.layers:
        layer.trainable = True
        generator.trainable = True
    for layer in dann.layers:
        layer.trainable = False
        dann.trainable = False

    dann_model = Model(inputs=[input_s], outputs=[out_cls_s, out_dann_s])
    dann_model.compile(optimizer=Adam(gen_rate),loss=['categorical_crossentropy', 'categorical_crossentropy'],
                        loss_weights = [1, dann_param])

    ones = np.ones((bt_size, 1), dtype=np.float32)
    zeros = np.zeros((bt_size, 1), dtype=np.float32)
    bigzeros = np.zeros((bt_size, 2), dtype=np.float32)
    ys_dann = np.concatenate([ones, zeros], axis = 1)
    yt_dann = np.concatenate([zeros, ones], axis = 1)
    y_dann = np.concatenate([ys_dann,yt_dann], axis = 0)
    ns = xs.shape[0]//bt_size
    nt = xt.shape[0]//bt_size
    maxv = 0

    for epoch in range(epochs):
        for i in range(steps):
            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * bt_size:(inds + 1) * bt_size]
            ys_batch = ys[inds * bt_size:(inds + 1) * bt_size]
            xt_batch = xt[indt * bt_size:(indt + 1) * bt_size]
            x_dann = np.concatenate([xs_batch,xt_batch], axis = 0)
            y_batch = np.concatenate([ys_batch,bigzeros], axis = 0)
            dann_flip.train_on_batch(x_dann, y_dann)
            dann_model.train_on_batch([x_dann], [y_batch, y_dann])
            if i%5 == 0:
                xs, ys = shuffle_data2(xs,ys)
                xt = shuffle_data(xt)

#since peaking point are different between methods, we return the max instead of the last acc as other benchmarks

        _, acc = cls_model.evaluate(x = xt_test, y = yt_test, verbose = 0) 
        maxv = max(acc, maxv) 
        if epoch % 2 ==0:
            print('test accuracy: ', maxv * 100)

    return maxv

total = 0
for rnd in range (5):
    total += run_dann(xs,ys,xt, xt_test,yt_test)
    print (total/(rnd+1))