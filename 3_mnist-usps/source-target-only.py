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
ip_shape = (28,28,1)
n_classes = 10

bn_axis = -1
mid_dim = 1024
was_dim = 100
gen_rate = 1e-4

epochs = 30
steps = 200

xs, ys, xt, xt_test, yt_test = data_usps_mnist(mnist_source =True) #False for training USPS-MNIST


def run_source(xs,ys,xt_test, yt_test):
    generator = make_generator()
    classifier = make_classifier()

    input_s = Input(shape=ip_shape)
    middle_s = generator(input_s)
    out_cls_s = classifier(middle_s)

    cls_model = Model(inputs=input_s, outputs=out_cls_s)
    cls_model.compile(optimizer=Adam(gen_rate),loss=['categorical_crossentropy'], metrics = ['accuracy'])

    ns = xs.shape[0]//bt_size
    avr = 0

    for epoch in range(epochs):
        for i in range(steps):
            inds = np.random.randint(0,ns)

            xs_batch = xs[inds * bt_size:(inds + 1) * bt_size]
            ys_batch = ys[inds * bt_size:(inds + 1) * bt_size]

            cls_model.train_on_batch(xs_batch, ys_batch)

            if i%20 == 0:
                xs, ys = shuffle_data2(xs,ys)

        _, acc = cls_model.evaluate(x = xt_test, y = yt_test, verbose = 0) # result for source-only
        # _, acc = cls_model.evaluate(x = xs, y = ys, verbose = 0) # result for target-only
        if epoch % 2 ==0:
            print('test accuracy: ', acc * 100)

        if epochs - epoch <=5: # get average last 5 epochs
            avr+= acc
    return avr/5

total = 0
for rnd in range (5):
    total += run_source(xs, ys, xt_test, yt_test)
    print (total/(rnd+1))