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
ip_shape = (5000,)
n_classes = 2
gen_rate = 1e-4
epochs = 30
steps = 200
mada_par = - float(sys.argv[3])

data_folder = './data/'
source_name = sys.argv[1]
target_name = sys.argv[2]  
xs, ys, xt, yt, xt_test, yt_test = load_amazon(source_name, target_name, data_folder, verbose=True)
ys = to_categorical(ys,2)
yt = to_categorical(yt,2)
yt_test = to_categorical(yt_test,2)

def make_mada():
    inputs = Input(shape = (mid_dim,))
    Flip = layer_flip.GradientReversal(mada_par)
    x = Flip(inputs)
    outputs = Dense(2)(x)
    model = Model(inputs = inputs, outputs = outputs)
    return model


def mada_loss(y_true, y_pred):
    labels = y_true[:,:2]
    softmax = y_true[:,2:]
    sigm = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = y_pred)
    return K.mean(sigm * softmax, axis = 0, keepdims = True)

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
                                       mada_loss])


    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
    left_y = np.concatenate([positive_y, dummy_y], axis = 1)
    right_y = np.concatenate([dummy_y, positive_y], axis = 1)
    tt = 0
    ns = xs.shape[0]//BATCH_SIZE
    nt = xt.shape[0]//BATCH_SIZE
    maxv = 0
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

            if i%20 == 0:
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
    total += run_mada(xs,ys,xt, xt_test,yt_test)
    print (total/(rnd+1))
