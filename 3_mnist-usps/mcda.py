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
from utils import *

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
gen_rate = 1e-4

thres = 0.9
epochs = 100
steps = 200

was_dim = 100
critics_param = 1e-4 # 1e-4 for USPS-MNIST and 1e-3 for MNIST-USPS

xs, ys, xt, xt_test, yt_test = data_usps_mnist(mnist_source =False) #False for training USPS-MNIST



def critics_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis = 0, keepdims = True)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    for i in range(11):
        gradients = K.gradients(y_pred[:,i], averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = y_true[:,i]* gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        if i == 0:
            v = K.mean(gradient_penalty)
            v = tf.reshape(v, shape = (1,1))
        else:
            u = K.mean(gradient_penalty)
            u = tf.reshape(u, shape = (1,1))
            v = tf.concat([v, u], axis = 1)
    return v


def make_critics():
    model = Sequential()
    model.add(Dense(was_dim * 11, input_shape = (mid_dim,)))
    model.add(LeakyReLU())
    model.add(Reshape((was_dim * 11, 1)))
    model.add(LocallyConnected1D(1, was_dim, strides = was_dim))
    return model


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((bt_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def run_mcda(xs,ys,xt,xt_test, yt_test):
    generator = make_generator()
    classifier = make_classifier()
    critics = make_critics()

    input_s = Input(shape=ip_shape)
    input_t = Input(shape=ip_shape)
    middle_s = generator(input_s)
    middle_t = generator(input_t)
    out_critics_s = critics(middle_s)
    out_critics_t = critics(middle_t)
    out_cls_s = classifier(middle_s)
    out_cls_t = classifier(middle_t)

    cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
    cls_model.compile(optimizer=Adam(0),loss=['categorical_crossentropy'], metrics = ['accuracy'])

    for layer in generator.layers:
        layer.trainable = False
        generator.trainable = False

    averaged_samples = RandomWeightedAverage()([middle_s, middle_t])
    averaged_samples_out = critics(averaged_samples)
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=gp_weight)
    partial_gp_loss.__name__ = 'gradient_penalty' 

    critics_critic =  Model(inputs=[input_s, input_t], outputs=[out_critics_s, out_critics_t, averaged_samples_out])
    critics_critic.compile(optimizer=Adam(gen_rate),loss=[critics_loss, critics_loss, partial_gp_loss])

    for layer in generator.layers:
        layer.trainable = True
        generator.trainable = True
    for layer in critics.layers:
        layer.trainable = False
        critics.trainable = False

    main_model = Model(inputs=[input_s, input_t], outputs=[out_cls_s, out_cls_t, out_critics_s, out_critics_t])
    main_model.compile(optimizer=Adam(gen_rate),loss=['categorical_crossentropy', 'categorical_crossentropy', critics_loss, critics_loss],
                        loss_weights = [1,1, critics_param, critics_param])


    ones = np.ones((bt_size, 1), dtype=np.float32)
    ns = xs.shape[0]//bt_size
    nt = xt.shape[0]//bt_size
    avr = 0
    TT =np.ones(11)/10
    TT[0] = 1.

    for epoch in range(epochs):
        for i in range(steps*5):
            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * bt_size:(inds + 1) * bt_size]
            ys_batch = ys[inds * bt_size:(inds + 1) * bt_size]
            xt_batch = xt[indt * bt_size:(indt + 1) * bt_size]
            yt_batch = cls_model.predict(xt_batch)
            yt_batch2 = yt_batch
            yt_batch = np.argmax(yt_batch, axis = 1)
            yt_batch = to_categorical(yt_batch, n_classes)
            ys_weights = np.concatenate([ones, ys_batch], axis = 1)
            yt_weights = np.concatenate([ones, yt_batch], axis = 1)
            ys_weights = ys_weights/(np.mean(ys_weights,axis = 0) + 1e-6) 
            yt_weights = yt_weights/(np.mean(yt_weights,axis = 0) + 1e-6)
            gradient_weights = ys_weights * yt_weights*TT
            ys_weights = ys_weights *TT
            yt_weights = yt_weights *TT
            ys_weights = ys_weights.reshape((ys_weights.shape[0],ys_weights.shape[1], 1))
            yt_weights = yt_weights.reshape((yt_weights.shape[0],yt_weights.shape[1], 1))
            gradient_weights = gradient_weights.reshape((gradient_weights.shape[0],gradient_weights.shape[1], 1))

            critics_critic.train_on_batch([xs_batch, xt_batch], [ -ys_weights, yt_weights, gradient_weights])
            if i%5 == 0:
                yz_batch = convert(yt_batch2)
                main_model.train_on_batch([xs_batch, xt_batch], [ys_batch, yz_batch, ys_weights, - yt_weights])

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
    total += run_mcda(xs, ys, xt, xt_test, yt_test)
    print (total/(rnd+1))

