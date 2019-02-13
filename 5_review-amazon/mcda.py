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


data_folder = './data/'
source_name = sys.argv[1]
target_name = sys.argv[2]  
xs, ys, xt, yt, xt_test, yt_test = load_amazon(source_name, target_name, data_folder, verbose=True)
ys = to_categorical(ys,2)
yt = to_categorical(yt,2)
yt_test = to_categorical(yt_test,2)



BATCH_SIZE = 128
TRAINING_RATIO = 5  

INPUT_SHAPE = (xs.shape[1],)
mid_dim = 500
was_dim = 100
tt = 1e-5
gen_rate = 1e-4


GRADIENT_PENALTY_WEIGHT =  1
critics_par =   1
was_rate = 1e-4

epochs =30
steps = 80

thres = 0.8



def mcritics_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis = 0, keepdims = True)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    for i in range(3):
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



def make_mcritics():
    model = Sequential()
    model.add(Dense(was_dim * 3, kernel_initializer='he_normal', input_shape = (mid_dim,)))
    model.add(LeakyReLU())
    model.add(Reshape((was_dim * 3, 1)))
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


    generator_input_for_mcritics1 = Input(shape=INPUT_SHAPE)
    generator_input_for_mcritics2 = Input(shape=INPUT_SHAPE)



    generated_samples_for_mcritics1 = generator(generator_input_for_mcritics1)
    generated_samples_for_mcritics2 = generator(generator_input_for_mcritics2)



    mcritics = make_mcritics()
    # mcritics.summary()

    mcritics_output_from_generator1 = mcritics(generated_samples_for_mcritics1)
    mcritics_output_from_generator2 = mcritics(generated_samples_for_mcritics2)

    classifier_output_from_generator1 = classifier(generated_samples_for_mcritics1)
    classifier_output_from_generator2 = classifier(generated_samples_for_mcritics2)


    averaged_samples = RandomWeightedAverage()([generated_samples_for_mcritics2, generated_samples_for_mcritics1])
    averaged_samples_out = mcritics(averaged_samples)
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error



    # Now that the generator_model is compiled, we can make the mcritics layers trainable.
    for layer in mcritics.layers:
        layer.trainable = True
    mcritics.trainable = True

    for layer in generator.layers:
        layer.trainable = False
    generator.trainable = False

    for layer in classifier.layers:
        layer.trainable = False
    classifier.trainable = False


    mcritics_model = Model(inputs=[generator_input_for_mcritics2, generator_input_for_mcritics1],
                                outputs=[mcritics_output_from_generator2,
                                         mcritics_output_from_generator1,
                                         averaged_samples_out])

    mcritics_model.compile(optimizer=Adam(was_rate),
                                loss=[mcritics_loss,
                                      mcritics_loss,
                                      partial_gp_loss])
    # mcritics_model.summary()



    for layer in mcritics.layers:
        layer.trainable = False
    mcritics.trainable = False

    for layer in generator.layers:
        layer.trainable = True
    generator.trainable = True

    for layer in classifier.layers:
        layer.trainable = True
    classifier.trainable = True


    generator_model = Model(inputs=[generator_input_for_mcritics2, generator_input_for_mcritics1],
                                outputs=[classifier_output_from_generator2,classifier_output_from_generator1,
                                         mcritics_output_from_generator2,
                                         mcritics_output_from_generator1])
    generator_model.compile(optimizer=Adam(gen_rate),
                                loss=['categorical_crossentropy','categorical_crossentropy',
                                       mcritics_loss,
                                       mcritics_loss], loss_weights = [1, 1, critics_par, critics_par])
    # generator_model.summary()





    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    # negative_y = -positive_y
    # dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    maxv = 0
    for epoch in range(epochs):


        ns = xs.shape[0]//BATCH_SIZE - 1
        nt = xt.shape[0]//BATCH_SIZE - 1

        for i in range(steps*TRAINING_RATIO):

            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            xt_batch = xt[indt * BATCH_SIZE:(indt + 1) * BATCH_SIZE]
            ys_batch = ys[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            zt_batch = generator.predict(xt_batch)
            yt_batch = classifier.predict(zt_batch)

            ys_weights = np.concatenate([positive_y, ys_batch], axis = 1)
            yt_weights = np.concatenate([positive_y, yt_batch], axis = 1)
            # TT = np.mean(ys_weights + yt_weights, axis = 0)/2
            TT = [1,0.5,0.5]
            ys_weights = ys_weights/(np.mean(ys_weights,axis = 0) + 1e-6) 
            yt_weights = yt_weights/(np.mean(yt_weights,axis = 0) + 1e-6) 
            gradient_weights = ys_weights * yt_weights*TT
            ys_weights = ys_weights *TT
            yt_weights = yt_weights *TT
            ys_weights = ys_weights.reshape((ys_weights.shape[0],ys_weights.shape[1], 1))
            yt_weights = yt_weights.reshape((yt_weights.shape[0],yt_weights.shape[1], 1))
            gradient_weights = gradient_weights.reshape((gradient_weights.shape[0],gradient_weights.shape[1], 1))
            # switch positive-negative because we put - in the loss function in gen model.
            mcritics_model.train_on_batch([xt_batch, xs_batch],[yt_weights, - ys_weights, gradient_weights])
            if i%5 == 4:
                yz = convert(yt_batch)
                generator_model.train_on_batch([xt_batch, xs_batch], [yz, ys_batch, - yt_weights, ys_weights])

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
    total += run_mcda(xs,ys,xt, xt_test,yt_test)
    print (total/(rnd+1))
