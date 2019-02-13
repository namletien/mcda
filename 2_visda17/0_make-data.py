import keras
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Lambda
# from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.backend import set_session,tensorflow_backend
import scipy



config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

img_width, img_height = 256,256

train_data_dir = os.path.expanduser('~/data/visda17/train/')
test_data_dir = os.path.expanduser('~/data/visda17/validation/')

base_model = InceptionV3(weights='imagenet', include_top=False)
y = base_model.output
features = GlobalAveragePooling2D()(y)
model = Model(inputs=base_model.input, outputs=features)
for layer in base_model.layers:
    layer.trainable = False
opt = keras.optimizers.rmsprop(lr=0.)
model.compile(loss='categorical_crossentropy',
              optimizer=opt)

datagen = ImageDataGenerator(
    rescale=1./255,
    )

train_generator = datagen.flow_from_directory(
    train_data_dir,
    shuffle = False,
    target_size=(img_width, img_height),
    batch_size=32)

x_train = model.predict_generator(train_generator)

np.save('x_train', x_train)

Q = DirectoryIterator(train_data_dir,image_data_generator = train_generator, 
    shuffle = False)
y_train = Q.classes
np.save('y_train', y_train)




test_generator = datagen.flow_from_directory(
    test_data_dir,
    shuffle = False,
    target_size=(img_width, img_height),
    batch_size=32)

x_test = model.predict_generator(test_generator)

np.save('x_test', x_test)

Q = DirectoryIterator(test_data_dir,image_data_generator = test_generator, 
    shuffle = False)
y_test = Q.classes
np.save('y_test', y_test)
