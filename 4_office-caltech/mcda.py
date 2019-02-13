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

from utils import *
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


data_dir = 'data/features/CaffeNet4096/'
source_name = sys.argv[1]
target_name = sys.argv[2]
xs, ys, xt, yt = load_office(source_name, target_name, data_dir)
ys = ys.flatten()-1
yt = yt.flatten()-1
ys = to_categorical(ys,10)
yt = to_categorical(yt,10)

print(xs.shape, xt.shape, xt.shape,ys.shape, yt.shape, yt.shape)

BATCH_SIZE = 128
TRAINING_RATIO = 5  
GRADIENT_PENALTY_WEIGHT = 10  
INPUT_SHAPE = (xs.shape[1],)
mid_dim = 500
was_dim = 100
gen_rate = 1e-4
was_rate = 5*1e-4
thres = 0.9

steps = 40
epochs = 30

if source_name == 'w' or target_name == 'w':
    wd_par = 0.01
else:
    wd_par = 0.0001


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis = 0, keepdims = True)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    for i in range(11):
        gradients = K.gradients(y_pred[:,i], averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = y_true[:,i]* gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        if i == 0:
            v = K.mean(gradient_penalty)
            v = tf.reshape(v, shape = (1,1))
        else:
            u = K.mean(gradient_penalty)
            u = tf.reshape(u, shape = (1,1))
            v = tf.concat([v, u], axis = 1)
    return v



def make_wasserstein():
    model = Sequential()
    model.add(Dense(was_dim * 11, kernel_initializer='he_normal', input_shape = (mid_dim,)))
    model.add(LeakyReLU())
    model.add(Reshape((was_dim * 11, 1)))
    model.add(LocallyConnected1D(1, was_dim, strides = was_dim, kernel_initializer='he_normal'))
    return model



class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])





def run_mcda(xs,ys,xt, xt_test,yt_test):
    generator = make_generator()

    classifier = make_classifier()
    classifier.compile(optimizer=Adam(gen_rate), loss=['categorical_crossentropy'],
         metrics = ['accuracy'])
    input_s = Input(shape=INPUT_SHAPE)
    middle_s = generator(input_s)
    out_cls_s = classifier(middle_s)
    cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
    cls_model.compile(optimizer=Adam(0),loss=['categorical_crossentropy'], metrics = ['accuracy'])

    generator_input_for_wasserstein1 = Input(shape=INPUT_SHAPE)
    generator_input_for_wasserstein2 = Input(shape=INPUT_SHAPE)


    generated_samples_for_wasserstein1 = generator(generator_input_for_wasserstein1)
    generated_samples_for_wasserstein2 = generator(generator_input_for_wasserstein2)


    wasserstein = make_wasserstein()

    wasserstein_output_from_generator1 = wasserstein(generated_samples_for_wasserstein1)
    wasserstein_output_from_generator2 = wasserstein(generated_samples_for_wasserstein2)

    classifier_output_from_generator1 = classifier(generated_samples_for_wasserstein1)
    classifier_output_from_generator2 = classifier(generated_samples_for_wasserstein2)


    averaged_samples = RandomWeightedAverage()([generated_samples_for_wasserstein2, generated_samples_for_wasserstein1])
    averaged_samples_out = wasserstein(averaged_samples)
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error


    # Now that the generator_model is compiled, we can make the wasserstein layers trainable.
    for layer in wasserstein.layers:
        layer.trainable = True
    wasserstein.trainable = True

    for layer in generator.layers:
        layer.trainable = False
    generator.trainable = False

    for layer in classifier.layers:
        layer.trainable = False
    classifier.trainable = False


    wasserstein_model = Model(inputs=[generator_input_for_wasserstein2, generator_input_for_wasserstein1],
                                outputs=[wasserstein_output_from_generator2,
                                         wasserstein_output_from_generator1,
                                         averaged_samples_out])

    wasserstein_model.compile(optimizer=Adam(was_rate),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    # wasserstein_model.summary()


    for layer in wasserstein.layers:
        layer.trainable = False
    wasserstein.trainable = False

    for layer in generator.layers:
        layer.trainable = True
    generator.trainable = True

    for layer in classifier.layers:
        layer.trainable = True
    classifier.trainable = True

    generator_model = Model(inputs=[generator_input_for_wasserstein2, generator_input_for_wasserstein1],
                                outputs=[classifier_output_from_generator2,classifier_output_from_generator1,
                                         wasserstein_output_from_generator2,
                                         wasserstein_output_from_generator1])
    generator_model.compile(optimizer=Adam(gen_rate),
                                loss=['categorical_crossentropy','categorical_crossentropy',
                                       wasserstein_loss,
                                       wasserstein_loss], loss_weights = [1, 1,wd_par, wd_par])




    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    # negative_y = -positive_y
    # dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    tt = 0
    ns = xs.shape[0]//BATCH_SIZE
    nt = xt.shape[0]//BATCH_SIZE
    avr = 0
    for epoch in range(epochs):

        for i in range(40 * TRAINING_RATIO):

            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            xt_batch = xt[indt * BATCH_SIZE:(indt + 1) * BATCH_SIZE]
            ys_batch = ys[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            zt_batch = generator.predict(xt_batch)
            yt_batch = classifier.predict(zt_batch)
            ys_weights = np.concatenate([positive_y, ys_batch], axis = 1)
            yt_weights = np.concatenate([positive_y, yt_batch], axis = 1)
            ys_weights = ys_weights/(np.mean(ys_weights,axis = 0) + 1e-6) 
            yt_weights = yt_weights/(np.mean(yt_weights,axis = 0) + 1e-6) 
            TT =np.ones(11)/10
            TT[0] = 1.
            gradient_weights = ys_weights * yt_weights*TT
            ys_weights = ys_weights *TT
            yt_weights = yt_weights *TT
            ys_weights = ys_weights.reshape((ys_weights.shape[0],ys_weights.shape[1], 1))
            yt_weights = yt_weights.reshape((yt_weights.shape[0],yt_weights.shape[1], 1))
            gradient_weights = gradient_weights.reshape((gradient_weights.shape[0],gradient_weights.shape[1], 1))
            # switch positive-negative because we put - in the loss function in gen model.
            wasserstein_model.train_on_batch([xt_batch, xs_batch],[yt_weights, - ys_weights, gradient_weights])
            if i%TRAINING_RATIO == 0:
                yz_batch = convert(yt_batch)
                generator_model.train_on_batch([xt_batch, xs_batch], [yz_batch, ys_batch, - yt_weights, ys_weights])

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
    total += run_mcda(xs, ys, xt, xt, yt)
    print (total/(rnd+1))
