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

import layer_flip

from utils import *
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


batch_size = 256
input_shape = (2048,)
gen_rate = 1e-4
n_classes = 12
epochs = 50
steps = 30

mada_par = - float(sys.argv[1])

print('mada', mada_par, batch_size)

xs_full, ys_full, xt_full, yt_full = get_data()


def make_mada():
    inputs = Input(shape = (mid_dim,))
    Flip = layer_flip.GradientReversal(mada_par)
    x = Flip(inputs)
    outputs = Dense(n_classes)(x)
    model = Model(inputs = inputs, outputs = outputs)
    return model


def mada_loss(y_true, y_pred):
    labels = y_true[:,:n_classes]
    softmax = y_true[:,n_classes:]
    sigm = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = y_pred)
    return K.mean(sigm * softmax, axis = 0, keepdims = True)


generator = make_generator()
classifier = make_classifier()

input_s = Input(shape=input_shape)
middle_s = generator(input_s)
out_cls_s = classifier(middle_s)
cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
cls_model.compile(optimizer=Adam(0),
    loss=['categorical_crossentropy'], metrics = ['accuracy'])

mada = make_mada()

generator_input_for_mada1 = Input(shape=input_shape)
generator_input_for_mada2 = Input(shape=input_shape)

generated_samples_for_mada1 = generator(generator_input_for_mada1)
generated_samples_for_mada2 = generator(generator_input_for_mada2)

mada_output_from_generator1 = mada(generated_samples_for_mada1)
mada_output_from_generator2 = mada(generated_samples_for_mada2)

classifier_output_from_generator1 = classifier(generated_samples_for_mada1)


main_model = Model(inputs=[generator_input_for_mada2, generator_input_for_mada1],
                            outputs=[classifier_output_from_generator1,
                                     mada_output_from_generator2,
                                     mada_output_from_generator1])
main_model.compile(optimizer=Adam(gen_rate),
                            loss=['categorical_crossentropy',
                                   mada_loss,
                                   mada_loss,])

positive_y = np.ones((batch_size, 1), dtype=np.float32)
dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

left_y = np.concatenate([positive_y, dummy_y], axis = 1)
right_y = np.concatenate([dummy_y, positive_y], axis = 1)


ns = xs_full.shape[0]//batch_size
nt = xt_full.shape[0]//batch_size
avr = np.zeros(65).reshape((5,13))
for epoch in range(epochs):
    for i in range(steps):

        inds = np.random.randint(0,ns)
        indt = np.random.randint(0,nt)

        xs_batch = xs_full[inds * batch_size:(inds + 1) * batch_size]
        xt_batch = xt_full[indt * batch_size:(indt + 1) * batch_size]
        ys_batch = ys_full[inds * batch_size:(inds + 1) * batch_size]
        yt_batch = cls_model.predict(xt_batch)
        ys_bunch = np.concatenate([dummy_y,ys_batch], axis = 1)
        yt_bunch = np.concatenate([positive_y,yt_batch], axis = 1)

        main_model.train_on_batch([xt_batch, xs_batch], 
            [ys_batch, yt_bunch, ys_bunch])

        xs_full, ys_full = shuffle_data2(xs_full,ys_full)
        xt_full, yt_full = shuffle_data2(xt_full, yt_full)

    avr[epoch%5] = test_target(xt_full,yt_full, cls_model)
    if epoch >=4:
        print (np.round(np.mean(avr, axis = 0),1))