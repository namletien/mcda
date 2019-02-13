import sys
import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, MaxPooling2D, GlobalMaxPooling2D, Dropout
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
from utils import *

from keras.backend import set_session,tensorflow_backend
import tensorflow as tf 

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

gp_weight = 10
bt_size = 128
ip_shape = (28,28,1)
n_classes = 10

bn_axis = -1
mid_dim = 1024
gen_rate = 1e-4

epochs = 100
steps = 200

was_dim = 100
wdgrl_param = float(sys.argv[1])

xs, ys, xt, xt_test, yt_test = data_usps_mnist(mnist_source =True) #False for training USPS-MNIST


def wdgrl_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)



def make_wdgrl():
    model = Sequential()
    model.add(Dense(was_dim, kernel_initializer='he_normal', input_shape = (mid_dim,)))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((bt_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])



def run_wdgrl(xs,ys,xt,xt_test, yt_test):
    generator = make_generator()
    classifier = make_classifier()
    wdgrl = make_wdgrl()

    input_s = Input(shape=ip_shape)
    input_t = Input(shape=ip_shape)
    middle_s = generator(input_s)
    middle_t = generator(input_t)
    out_wdgrl_s = wdgrl(middle_s)
    out_wdgrl_t = wdgrl(middle_t)
    out_cls_s = classifier(middle_s)

    cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
    cls_model.compile(optimizer=Adam(0),loss=['categorical_crossentropy'], metrics = ['accuracy'])

    for layer in generator.layers:
        layer.trainable = False
        generator.trainable = False

    averaged_samples = RandomWeightedAverage()([middle_s, middle_t])
    averaged_samples_out = wdgrl(averaged_samples)
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=gp_weight)
    partial_gp_loss.__name__ = 'gradient_penalty' 

    wdgrl_critic =  Model(inputs=[input_s, input_t], outputs=[out_wdgrl_s, out_wdgrl_t, averaged_samples_out])
    wdgrl_critic.compile(optimizer=Adam(gen_rate),loss=[wdgrl_loss, wdgrl_loss, partial_gp_loss])

    for layer in generator.layers:
        layer.trainable = True
        generator.trainable = True
    for layer in wdgrl.layers:
        layer.trainable = False
        wdgrl.trainable = False

    wdgrl_model = Model(inputs=[input_s, input_t], outputs=[out_cls_s, out_wdgrl_s, out_wdgrl_t])
    wdgrl_model.compile(optimizer=Adam(gen_rate),loss=['categorical_crossentropy', wdgrl_loss, wdgrl_loss],
                        loss_weights = [1,wdgrl_param, wdgrl_param])


    ones = np.ones((bt_size, n_classes), dtype=np.float32)
    zeros = np.zeros((bt_size, n_classes), dtype=np.float32)
    ns = xs.shape[0]//bt_size
    nt = xt.shape[0]//bt_size
    avr = 0

    for epoch in range(epochs):
        for i in range(steps*5):
            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * bt_size:(inds + 1) * bt_size]
            ys_batch = ys[inds * bt_size:(inds + 1) * bt_size]
            xt_batch = xt[indt * bt_size:(indt + 1) * bt_size]

            wdgrl_critic.train_on_batch([xs_batch, xt_batch], [ - ones, ones, zeros])
            if i%5 == 0:
                wdgrl_model.train_on_batch([xs_batch, xt_batch], [ys_batch, ones, - ones])

            if i%50 == 0:
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
    total += run_wdgrl(xs, ys, xt, xt_test, yt_test)
    print (total/(rnd+1))

