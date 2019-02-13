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
import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from scipy.spatial.distance import cdist
from keras.engine import Layer


BATCH_SIZE = 128
INPUT_SHAPE = (2,)
tt = 1e-4
mid_dim = 15
was_dim = 10
thres = 0.95



def make_generator():
    model = Sequential()
    model.add(Dense(30, kernel_regularizer=l2(tt), input_shape = INPUT_SHAPE, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(mid_dim, kernel_regularizer=l2(tt), activation = 'relu'))
    # model.add(Dropout(0.2))
    return model

def make_classifier():
    model = Sequential()
    model.add(Dense(2, kernel_regularizer=l2(tt), input_shape = (mid_dim,), activation='softmax'))
    return model

def make_wasserstein():
    model = Sequential()
    model.add(Dense(10, kernel_regularizer=l2(tt), input_shape = (mid_dim,), activation = 'relu'))
    model.add(Dense(1, kernel_regularizer=l2(tt)))
    return model

def make_critics():
    model = Sequential()
    model.add(Dense(was_dim * 3, input_shape = (mid_dim,), activation = 'relu',kernel_regularizer=l2(tt)))
    model.add(Reshape((was_dim * 3, 1)))
    model.add(LocallyConnected1D(1, was_dim, strides = was_dim, kernel_regularizer=l2(tt)))
    model.add(Reshape((3,)))
    return model



def make_trans_moons(theta=40, nb=100, noise=.05):
    tf.set_random_seed(0)
    np.random.seed(0)
    from math import cos, sin, pi
    
    X, y = make_moons(nb, noise=noise, random_state=1) 
    Xt, yt = make_moons(nb, noise=noise, random_state=2)
    
    trans = -np.mean(X, axis=0) 
    X  = 2*(X+trans)
    Xt = 2*(Xt+trans)
    
    theta = -theta*pi/180
    rotation = np.array( [  [cos(theta), sin(theta)], [-sin(theta), cos(theta)] ] )
    Xt = np.dot(Xt, rotation.T)
    
    return X, y, Xt, yt


def draw_trans_data(X, y, Xt, predict_fct=None, neurons_to_draw=None, colormap_index=0, special_points=None, special_xytext=None):
    # Some line of codes come from: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    if colormap_index==0:
        cm_bright = ListedColormap(['#FF0000', '#00FF00'])
    else:
        cm_bright = ListedColormap(['#0000FF', '#000000'])

    x_min, x_max = 1.1*X[:, 0].min(), 1.1*X[:, 0].max()
    y_min, y_max = 1.5*X[:, 1].min(), 1.5*X[:, 1].max()
        
    pyplot.xlim((x_min,x_max))
    pyplot.ylim((y_min,y_max))
    
    pyplot.tick_params(direction='in', labelleft=False)    


    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = predict_fct.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis = 1)
    Z = Z.reshape(xx.shape)
    pyplot.contourf(xx, yy, Z, cmap=cm_bright, alpha=.4)


    if X is not None:
        ind1 = np.where(y ==0)
        ind2 = np.where(y == 1)
        pyplot.scatter(X[ind1, 0], X[ind1, 1], c='r', s=5)

        pyplot.scatter(X[ind2,0], X[ind2, 1], c='g', s=5)

    if Xt is not None:
        pyplot.scatter(Xt[:, 0], Xt[:, 1], c='k', s=5)

    pyplot.contour(xx, yy, Z, colors='blue', linewidths=2)

def convert (y):
    lss = []
    us = []
    for i in range(2):
        ind = np.where(y[:,i]>=thres)[0]
        lss.append(ind)
        us.append(len(ind))
    u = min(us)

    yz = np.zeros(y.shape)

    for i in range(2):
        ind = lss[i]
        ind = ind[:u]
        for j in ind:
            yz[j][i] = 1.
    return yz


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])




def shuffle_data(X):
        Q =np.asarray(range(len(X)))
        ind = np.random.permutation(Q)
        return X[ind]
def shuffle_data2(X,y):
        assert len(X) == len(y)
        Q =np.asarray(range(len(y)))
        ind = np.random.permutation(Q)
        return X[ind], y[ind]



def reverse_gradient(X, hp_lambda):
    # source: https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py 
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def compute_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))








def run_MCDA(xs,ys,xt,xt_test, yt_test):

    def wasserstein_loss(y_true, y_pred):
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

    tf.set_random_seed(100)
    np.random.seed(100)

    TRAINING_RATIO = 1  
    GRADIENT_PENALTY_WEIGHT = 2

    wd_par = 0.001
    gen_rate = 5*1e-4
    was_rate = 5*1e-4

    generator = make_generator()

    classifier = make_classifier()
    classifier.compile(optimizer=Adam(gen_rate), loss=['categorical_crossentropy'],
         metrics = ['accuracy'])

    aa = Input(shape=INPUT_SHAPE)
    bb = generator(aa)
    cc = classifier(bb)
    full_model = Model(inputs = aa, outputs = cc)

    full_model.compile(optimizer=Adam(gen_rate), loss=['categorical_crossentropy'],
         metrics = ['accuracy'])

    
    generator_input_for_wasserstein1 = Input(shape=INPUT_SHAPE)
    generator_input_for_wasserstein2 = Input(shape=INPUT_SHAPE)

    generated_samples_for_wasserstein1 = generator(generator_input_for_wasserstein1)
    generated_samples_for_wasserstein2 = generator(generator_input_for_wasserstein2)

    wasserstein = make_critics()
    # wasserstein.summary()

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
                                       wasserstein_loss], loss_weights = [1, 1, wd_par, wd_par])

    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)

    ns = xs.shape[0]//BATCH_SIZE - 1
    nt = xt.shape[0]//BATCH_SIZE - 1
    TT = [1,0.5,0.5]    
    for epoch in range(2001):
        xs, ys = shuffle_data2(xs,ys)
        xt = shuffle_data(xt)


        ns = xs.shape[0]//BATCH_SIZE - 1
        nt = xt.shape[0]//BATCH_SIZE - 1

        for i in range(5):
            for j in range(TRAINING_RATIO):
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
                gradient_weights = ys_weights * yt_weights*TT
                ys_weights = ys_weights *TT
                yt_weights = yt_weights *TT
                wasserstein_model.train_on_batch([xt_batch, xs_batch],[yt_weights, - ys_weights, gradient_weights])
                if j == 0:
                    yz = convert(yt_batch)
                    generator_model.train_on_batch([xt_batch, xs_batch], [yz, ys_batch, - yt_weights, ys_weights])



        if epoch%200 ==0:
            # print('epoch: ', epoch)
            _, y1 = full_model.evaluate(x = xs, y = ys, verbose = 0)

            _, y2 = full_model.evaluate(x = xt_test, y = yt_test, verbose = 0)

            # print('train accuracy: ', y1)
            # print('test accuracy: ', y2)
            # print('max test accuracy: ', tt)
            print("Epoch: ", epoch//200, '.\t Max train acc: ', y1 * 100, '.\t Max test acc: ', y2 * 100)
    return full_model




def run_WDGRL(xs,ys,xt,xt_test, yt_test):

    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    tf.set_random_seed(10)
    np.random.seed(10)

    TRAINING_RATIO = 1  
    GRADIENT_PENALTY_WEIGHT = 2

    wd_par = 0.001
    gen_rate = 5*1e-4
    was_rate = 5*1e-4
    generator = make_generator()

    classifier = make_classifier()
    classifier.compile(optimizer=Adam(gen_rate), loss=['categorical_crossentropy'],
         metrics = ['accuracy'])
    aa = Input(shape=INPUT_SHAPE)
    bb = generator(aa)
    cc = classifier(bb)
    full_model = Model(inputs = aa, outputs = cc)

    full_model.compile(optimizer=Adam(gen_rate), loss=['categorical_crossentropy'],
         metrics = ['accuracy'])



    generator_input_for_wasserstein1 = Input(shape=INPUT_SHAPE)
    generator_input_for_wasserstein2 = Input(shape=INPUT_SHAPE)



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
                                outputs=[classifier_output_from_generator1,
                                         wasserstein_output_from_generator2,
                                         wasserstein_output_from_generator1])
    generator_model.compile(optimizer=Adam(gen_rate),
                                loss=['categorical_crossentropy',
                                       wasserstein_loss,
                                       wasserstein_loss], loss_weights = [1, wd_par, wd_par])
    # generator_model.summary()





    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)


    for epoch in range(2001):
        xs, ys = shuffle_data2(xs,ys)
        xt = shuffle_data(xt)


        ns = xs.shape[0]//BATCH_SIZE - 1
        nt = xt.shape[0]//BATCH_SIZE - 1

        for i in range(5):

            for j in range(TRAINING_RATIO):
                inds = np.random.randint(0,ns)
                indt = np.random.randint(0,nt)

                xs_batch = xs[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
                xt_batch = xt[indt * BATCH_SIZE:(indt + 1) * BATCH_SIZE]
                # switch positive-negative because we put - in the loss function in gen model.
                wasserstein_model.train_on_batch([xt_batch, xs_batch],[positive_y, negative_y, dummy_y])



            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            xt_batch = xt[indt * BATCH_SIZE:(indt + 1) * BATCH_SIZE]
            ys_batch = ys[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]

            generator_model.train_on_batch([xt_batch, xs_batch], [ys_batch, negative_y, positive_y])


        if epoch%200 ==0:
            # print('epoch: ', epoch)
            _, y1 = full_model.evaluate(x = xs, y = ys, verbose = 0)

            _, y2 = full_model.evaluate(x = xt_test, y = yt_test, verbose = 0)

        if epoch%200 ==0:
            print("Epoch: ", epoch//200, '.\t Max train acc: ', y1 * 100, '.\t Max test acc: ', y2 * 100)
    return full_model






def run_MADA(xs,ys,xt,xt_test, yt_test):

    def make_mada():
        inputs = Input(shape = (mid_dim,))
        Flip = GradientReversal(grl_lambda)
        x = Flip(inputs)
        x = Dense(10, kernel_regularizer=l2(tt))(x)
        outputs = Dense(2, kernel_regularizer=l2(tt))(x)
        model = Model(inputs = inputs, outputs = outputs)
        return model
    def mada_loss(y_true, y_pred):
        labels = y_true[:,:2]
        softmax = y_true[:,2:]
        sigm = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = y_pred)
        return K.mean(sigm * softmax, axis = 0, keepdims = True)

    tf.set_random_seed(100)
    np.random.seed(100)

    grl_lambda = 0.01
    gen_rate = 5*1e-4
    tt = 1e-2

    generator = make_generator()

    classifier = make_classifier()
    classifier.compile(optimizer=Adam(gen_rate), loss=['categorical_crossentropy'],
         metrics = ['accuracy'])
    wasserstein = make_mada()

    aa = Input(shape=INPUT_SHAPE)
    bb = generator(aa)
    cc = classifier(bb)
    full_model = Model(inputs = aa, outputs = cc)

    full_model.compile(optimizer=Adam(gen_rate), loss=['categorical_crossentropy'],
         metrics = ['accuracy'])
    
    generator_input_for_wasserstein1 = Input(shape=INPUT_SHAPE)
    generator_input_for_wasserstein2 = Input(shape=INPUT_SHAPE)


    generated_samples_for_wasserstein1 = generator(generator_input_for_wasserstein1)
    generated_samples_for_wasserstein2 = generator(generator_input_for_wasserstein2)



    wasserstein_output_from_generator1 = wasserstein(generated_samples_for_wasserstein1)
    wasserstein_output_from_generator2 = wasserstein(generated_samples_for_wasserstein2)

    classifier_output_from_generator1 = classifier(generated_samples_for_wasserstein1)


    generator_model = Model(inputs=[generator_input_for_wasserstein2, generator_input_for_wasserstein1],
                                outputs=[classifier_output_from_generator1,
                                         wasserstein_output_from_generator2,
                                         wasserstein_output_from_generator1])
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
    for epoch in range(801):

        for i in range(5):

            inds = np.random.randint(0,ns)
            indt = np.random.randint(0,nt)

            xs_batch = xs[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            xt_batch = xt[indt * BATCH_SIZE:(indt + 1) * BATCH_SIZE]
            ys_batch = ys[inds * BATCH_SIZE:(inds + 1) * BATCH_SIZE]
            yt_batch = full_model.predict(xt_batch)
            ys_bunch = np.concatenate([dummy_y,ys_batch], axis = 1)
            yt_bunch = np.concatenate([positive_y,yt_batch], axis = 1)

            generator_model.train_on_batch([xt_batch, xs_batch], [ys_batch, yt_bunch, ys_bunch])

            xs, ys = shuffle_data2(xs,ys)
            xt = shuffle_data(xt)


        if epoch%80 ==0:
            # print('epoch: ', epoch)
            _, y1 = full_model.evaluate(x = xs, y = ys, verbose = 0)

            _, y2 = full_model.evaluate(x = xt_test, y = yt_test, verbose = 0)

            print("Epoch: ", epoch//80, '.\t Max train acc: ', y1 * 100, '.\t Max test acc: ', y2 * 100)
    return full_model
