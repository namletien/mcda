import sys
import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense
# from keras.layers import Reshape, Flatten, MaxPooling2D, GlobalMaxPooling2D, Dropout, Lambda
from keras.layers.merge import _Merge
from keras.utils import to_categorical
# from keras.layers.convolutional import Convolution2D, Conv2DTranspose
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
# from keras.regularizers import l2
# from keras import backend as K
# from functools import partial
# from scipy.sparse import vstack
from keras.backend import set_session,tensorflow_backend
import tensorflow as tf 
# import scipy
from utils import *


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


n_classes = 12
batch_size = 256
input_shape = (2048,)
gen_rate = 1e-4
gp_weight = 10
gp_ratio = 4
thres = 0.8
epochs = 50
steps = 150

was_dim = 30
mcda_param = 1e-2 

print(thres, mcda_param, batch_size)

xs_full, ys_full, xt_full, yt_full = get_data()


def mcda_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis = 0, keepdims = True)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    for i in range(n_classes + 1):
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


def make_mcda():
    model = Sequential()
    model.add(Dense(was_dim * (n_classes+1), input_shape = (mid_dim,)))
    model.add(LeakyReLU())
    model.add(Reshape((was_dim * (n_classes+1), 1)))
    model.add(LocallyConnected1D(1, was_dim, strides = was_dim))
    return model


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


generator = make_generator()
classifier = make_classifier()
mcda = make_mcda()


input_s = Input(shape=input_shape)
input_t = Input(shape=input_shape)
middle_s = generator(input_s)
middle_t = generator(input_t)
out_mcda_s = mcda(middle_s)
out_mcda_t = mcda(middle_t)
out_cls_s = classifier(middle_s)
out_cls_t = classifier(middle_t)

cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
cls_model.compile(optimizer=Adam(0),
    loss=['categorical_crossentropy'], 
    metrics = ['accuracy'])
cls_model.summary()

for layer in generator.layers:
    layer.trainable = False
    generator.trainable = False

averaged_samples = RandomWeightedAverage()([middle_s, middle_t])
averaged_samples_out = mcda(averaged_samples)
partial_gp_loss = partial(gradient_penalty_loss, 
    averaged_samples=averaged_samples, 
    gradient_penalty_weight=gp_weight)
partial_gp_loss.__name__ = 'gradient_penalty' 

mcda_critic =  Model(inputs=[input_s, input_t], 
    outputs=[out_mcda_s, out_mcda_t, averaged_samples_out])
mcda_critic.compile(optimizer=Adam(gen_rate),
    loss=[mcda_loss, mcda_loss, partial_gp_loss])

for layer in generator.layers:
    layer.trainable = True
    generator.trainable = True
for layer in mcda.layers:
    layer.trainable = False
    mcda.trainable = False

main_model = Model(inputs=[input_s, input_t], 
    outputs=[out_cls_s, out_cls_t, out_mcda_s, out_mcda_t])
main_model.compile(optimizer=Adam(gen_rate),
    loss=['categorical_crossentropy', 'categorical_crossentropy', mcda_loss, mcda_loss],
    loss_weights = [1,1, mcda_param, mcda_param])


ones = np.ones((batch_size, 1), dtype=np.float32)
TT =np.ones(n_classes+1)/n_classes
TT[0] = 1.

ns = xs_full.shape[0]//batch_size
nt = xt_full.shape[0]//batch_size
avr = np.zeros(65).reshape((5,13))
for epoch in range(epochs):

    for i in range(steps * gp_ratio):

        inds = np.random.randint(0,ns)
        indt = np.random.randint(0,nt)

        xs_batch = xs_full[inds * batch_size:(inds + 1) * batch_size]
        xt_batch = xt_full[indt * batch_size:(indt + 1) * batch_size]
        ys_batch = ys_full[inds * batch_size:(inds + 1) * batch_size]
        yt_batch = cls_model.predict(xt_batch)
        
        ys_weights = np.concatenate([ones, ys_batch], axis = 1)
        yt_weights = np.concatenate([ones, yt_batch], axis = 1)
        ys_weights = ys_weights/(np.mean(ys_weights,axis = 0) + 1e-6) 
        yt_weights = yt_weights/(np.mean(yt_weights,axis = 0) + 1e-6)
        gradient_weights = ys_weights * yt_weights*TT
        ys_weights = ys_weights *TT
        yt_weights = yt_weights *TT
        ys_weights = ys_weights.reshape((
            ys_weights.shape[0],ys_weights.shape[1], 1))
        yt_weights = yt_weights.reshape((
            yt_weights.shape[0],yt_weights.shape[1], 1))
        gradient_weights = gradient_weights.reshape((
            gradient_weights.shape[0],gradient_weights.shape[1], 1))

        mcda_critic.train_on_batch([xs_batch, xt_batch], 
            [ -ys_weights, yt_weights, gradient_weights])

        if i%gp_ratio == 0:
            yz_batch = convert(yt_batch, thres)
            main_model.train_on_batch(
                [xs_batch, xt_batch], 
                [ys_batch, yz_batch, ys_weights, - yt_weights])

            xs_full, ys_full = shuffle_data2(xs_full,ys_full)
            xt_full, yt_full = shuffle_data2(xt_full, yt_full)

    # generator.save('mcda_model.h5')

    avr[epoch%5] = test_target(xt_full,yt_full, cls_model)
    if epoch >=4:
        print (np.round(np.mean(avr, axis = 0),1))