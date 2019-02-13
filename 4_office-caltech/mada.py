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

data_dir = 'data/features/CaffeNet4096/'
source_name = sys.argv[1]
target_name = sys.argv[2]
xs, ys, xt, yt = load_office(source_name, target_name, data_dir)
ys = ys.flatten()-1
yt = yt.flatten()-1
ys = to_categorical(ys,10)
yt = to_categorical(yt,10)



print(xs.shape, xt.shape, xt.shape,ys.shape, yt.shape, yt.shape)


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

BATCH_SIZE = 128
INPUT_SHAPE = (xs.shape[1],)
mid_dim = 500
gen_rate = 1e-4

steps = 40
epochs = 30

mada_par = - float(sys.argv[3])


def make_mada():
    inputs = Input(shape = (mid_dim,))
    Flip = layer_flip.GradientReversal(mada_par)
    x = Flip(inputs)
    outputs = Dense(10)(x)
    model = Model(inputs = inputs, outputs = outputs)
    return model


def mada_loss(y_true, y_pred):
    labels = y_true[:,:10]
    softmax = y_true[:,10:]
    sigm = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = y_pred)
    return K.mean(sigm * softmax, axis = 0, keepdims = True)


positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

def run_mada(xs,ys,xt, xt_test,yt_test):
    # Now we initialize the generator and mada.
    generator = make_generator()

    classifier = make_classifier()
    classifier.compile(optimizer=Adam(gen_rate), loss=['categorical_crossentropy'],
         metrics = ['accuracy'])
    input_s = Input(shape=INPUT_SHAPE)
    middle_s = generator(input_s)
    out_cls_s = classifier(middle_s)
    cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
    cls_model.compile(optimizer=Adam(0),loss=['categorical_crossentropy'], metrics = ['accuracy'])

    mada = make_mada()

    generator_input_for_mada1 = Input(shape=INPUT_SHAPE)
    generator_input_for_mada2 = Input(shape=INPUT_SHAPE)


    generated_samples_for_mada1 = generator(generator_input_for_mada1)
    generated_samples_for_mada2 = generator(generator_input_for_mada2)



    mada_output_from_generator1 = mada(generated_samples_for_mada1)
    mada_output_from_generator2 = mada(generated_samples_for_mada2)

    classifier_output_from_generator1 = classifier(generated_samples_for_mada1)


    generator_model = Model(inputs=[generator_input_for_mada2, generator_input_for_mada1],
                                outputs=[classifier_output_from_generator1,
                                         mada_output_from_generator2,
                                         mada_output_from_generator1])
    generator_model.compile(optimizer=Adam(gen_rate),
                                loss=['categorical_crossentropy',
                                       mada_loss,
                                       mada_loss,])


    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
    left_y = np.concatenate([positive_y, dummy_y], axis = 1)
    right_y = np.concatenate([dummy_y, positive_y], axis = 1)
    tt = 0
    ns = xs.shape[0]//BATCH_SIZE
    nt = xt.shape[0]//BATCH_SIZE
    avr = 0
    for epoch in range(epochs):

        for i in range(steps):

            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            xt_batch = xt[indt * BATCH_SIZE:(indt + 1) * BATCH_SIZE]
            ys_batch = ys[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            zt_batch = generator.predict(xt_batch)
            yt_batch = classifier.predict(zt_batch)
            ys_bunch = np.concatenate([dummy_y,ys_batch], axis = 1)
            yt_bunch = np.concatenate([positive_y,yt_batch], axis = 1)

            generator_model.train_on_batch([xt_batch, xs_batch], [ys_batch, yt_bunch, ys_bunch])

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
    total += run_mada(xs, ys, xt, xt, yt)
    print (total/(rnd+1))
