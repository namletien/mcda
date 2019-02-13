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

import layer_flip


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


bt_size = 128
ip_shape = (28,28,1)
n_classes = 10

bn_axis = -1
mid_dim = 1024
gen_rate = 1e-4

epochs = 100
steps = 200

mada_param = float(sys.argv[1])

xs, ys, xt, xt_test, yt_test = data_usps_mnist(mnist_source =True) #False for training USPS-MNIST


def make_mada():
    model = Sequential()
    model.add(Dense(n_classes, kernel_regularizer=l2(tt), input_shape = (mid_dim,)))
    return model


def mada_loss(y_true, y_pred):
    mada_labels = y_true[:, :n_classes]
    mada_probs = y_true[:, n_classes:]
    sigm = tf.nn.sigmoid_cross_entropy_with_logits(labels = mada_labels, logits = y_pred)
    return K.mean(sigm * mada_probs, axis = 1)* 10


def run_mada(xs,ys,xt,xt_test, yt_test):
    generator = make_generator()
    classifier = make_classifier()
    mada = make_mada()

    input_s = Input(shape=ip_shape)
    input_t = Input(shape=ip_shape)
    middle_s = generator(input_s)
    middle_t = generator(input_t)
    out_mada_s = mada(middle_s)
    out_mada_t = mada(middle_t)
    out_cls_s = classifier(middle_s)

    cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
    cls_model.compile(optimizer=Adam(0),loss=['categorical_crossentropy'], metrics = ['accuracy'])

    for layer in generator.layers:
        layer.trainable = False
        generator.trainable = False

    mada_flip =  Model(inputs=[input_s, input_t], outputs=[out_mada_s, out_mada_t])
    mada_flip.compile(optimizer=Adam(gen_rate),loss=[mada_loss, mada_loss])

    for layer in generator.layers:
        layer.trainable = True
        generator.trainable = True
    for layer in mada.layers:
        layer.trainable = False
        mada.trainable = False

    mada_model = Model(inputs=[input_s, input_t], outputs=[out_cls_s, out_mada_s, out_mada_t])
    mada_model.compile(optimizer=Adam(gen_rate),loss=['categorical_crossentropy', mada_loss, mada_loss],
                        loss_weights = [1,-mada_param, -mada_param])



    ones = np.ones((bt_size, n_classes), dtype=np.float32)
    zeros = np.zeros((bt_size, n_classes), dtype=np.float32)
    ns = xs.shape[0]//bt_size
    nt = xt.shape[0]//bt_size
    avr = 0

    for epoch in range(epochs):
        for i in range(steps):
            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * bt_size:(inds + 1) * bt_size]
            ys_batch = ys[inds * bt_size:(inds + 1) * bt_size]
            xt_batch = xt[indt * bt_size:(indt + 1) * bt_size]
            yt_batch = cls_model.predict(xt_batch)

            ys_mada = np.concatenate([zeros,ys_batch], axis = 1)
            yt_mada = np.concatenate([ones,yt_batch], axis = 1)

            mada_flip.train_on_batch([xs_batch, xt_batch], [ys_mada, yt_mada])
            mada_model.train_on_batch([xs_batch, xt_batch], [ys_batch, ys_mada, yt_mada])

            if i%20 == 0:
                xs, ys = shuffle_data2(xs,ys)
                xt = shuffle_data(xt)

        _, acc = cls_model.evaluate(x = xt_test, y = yt_test, verbose = 0)
        if epoch % 2 ==0:
            print('test accuracy: ', acc * 100)

        if epochs - epoch <=5: # get average last 5 epochs
            avr+= acc
    return avr/5

total = 0
for rnd in range (5):
    total += run_mada(xs, ys, xt, xt_test, yt_test)
    print (total/(rnd+1))