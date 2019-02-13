import sys
import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense
# from keras.layers import Reshape, Flatten, MaxPooling2D, GlobalMaxPooling2D, Dropout, Lambda
from keras.layers.merge import _Merge
# from keras.utils import to_categorical
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


batch_size = 128
input_shape = (2048,)
gen_rate = 1e-4
n_classes = 12


xs_full, ys_full, xt_full, yt_full = get_data()

generator = make_generator()
classifier = make_classifier()

input_s = Input(shape=input_shape)
middle_s = generator(input_s)
out_cls_s = classifier(middle_s)

cls_model = Model(inputs=input_s, outputs=out_cls_s)
cls_model.compile(optimizer=Adam(gen_rate),
    loss=['categorical_crossentropy'], 
    metrics = ['accuracy'])

avr = np.zeros(65).reshape((5,13))
for epoch in range(10):
    cls_model.fit(xs_full,ys_full,
              batch_size = 256,
              verbose = 0)
    avr[epoch%5] = test_target(xt_full,yt_full, cls_model)
    if epoch >=4:
        print (np.round(np.mean(avr, axis = 0),1)) 