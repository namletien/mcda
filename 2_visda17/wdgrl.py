
import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, MaxPooling2D
from keras.layers import GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.layers.merge import _Merge
from keras.utils import to_categorical
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2
from keras.datasets import mnist
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from functools import partial
from utils import *

from keras.backend import set_session,tensorflow_backend
import tensorflow as tf 

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


n_classes = 12
batch_size = 256
input_shape = (2048,)
gen_rate = 1e-4
gp_weight = 10
gp_ratio = 4
was_dim = 30

wd_par = float(sys.argv[1])

epochs = 50
steps = 30

print('wdgrl', wd_par, batch_size)

xs_full, ys_full, xt_full, yt_full = get_data()

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gp_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gp_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

def make_wasserstein():
    model = Sequential()
    model.add(Dense(was_dim,  input_shape = (mid_dim,)))
    model.add(LeakyReLU())
    model.add(Dense(1))
    return model

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

generator = make_generator()
classifier = make_classifier()

input_s = Input(shape=input_shape)
middle_s = generator(input_s)
out_cls_s = classifier(middle_s)
cls_model = Model(inputs=[input_s], outputs=[out_cls_s])
cls_model.compile(optimizer=Adam(0),
    loss=['categorical_crossentropy'], 
    metrics = ['accuracy'])

generator_input_for_wasserstein1 = Input(shape=input_shape)
generator_input_for_wasserstein2 = Input(shape=input_shape)



generated_samples_for_wasserstein1 = generator(generator_input_for_wasserstein1)
generated_samples_for_wasserstein2 = generator(generator_input_for_wasserstein2)


wasserstein = make_wasserstein()

wasserstein_output_from_generator1 = wasserstein(generated_samples_for_wasserstein1)
wasserstein_output_from_generator2 = wasserstein(generated_samples_for_wasserstein2)

classifier_output_from_generator1 = classifier(generated_samples_for_wasserstein1)


averaged_samples = RandomWeightedAverage()([generated_samples_for_wasserstein2, generated_samples_for_wasserstein1])
averaged_samples_out = wasserstein(averaged_samples)
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gp_weight=gp_weight)
partial_gp_loss.__name__ = 'gradient_penalty' 

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

wasserstein_model.compile(optimizer=Adam(gen_rate),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])


for layer in wasserstein.layers:
    layer.trainable = False
wasserstein.trainable = False

for layer in generator.layers:
    layer.trainable = True
generator.trainable = True

for layer in classifier.layers:
    layer.trainable = True
classifier.trainable = True


main_model = Model(inputs=[generator_input_for_wasserstein2, generator_input_for_wasserstein1],
                            outputs=[classifier_output_from_generator1,
                                     wasserstein_output_from_generator2,
                                     wasserstein_output_from_generator1])
main_model.compile(optimizer=Adam(gen_rate),
                            loss=['categorical_crossentropy',
                                   wasserstein_loss,
                                   wasserstein_loss], 
                                   loss_weights = [1, wd_par, wd_par])



positive_y = np.ones((batch_size, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

ns = xs_full.shape[0]//batch_size
nt = xt_full.shape[0]//batch_size
avr = np.zeros(65).reshape((5,13))
for epoch in range(epochs):
    for i in range(steps*gp_ratio):
        inds = np.random.randint(0,ns)
        indt = np.random.randint(0,nt)

        xs_batch = xs_full[inds * batch_size:(inds + 1) * batch_size]
        xt_batch = xt_full[indt * batch_size:(indt + 1) * batch_size]
        ys_batch = ys_full[inds * batch_size:(inds + 1) * batch_size]

        wasserstein_model.train_on_batch([xt_batch, xs_batch],
            [positive_y, negative_y, dummy_y])
        
        if i%gp_ratio == 0:            
            main_model.train_on_batch([xt_batch, xs_batch], 
                [ys_batch, negative_y, positive_y])

            xs_full, ys_full = shuffle_data2(xs_full,ys_full)
            xt_full, yt_full = shuffle_data2(xt_full, yt_full)

    avr[epoch%5] = test_target(xt_full,yt_full, cls_model)
    if epoch >=4:
        print (np.round(np.mean(avr, axis = 0),1))
