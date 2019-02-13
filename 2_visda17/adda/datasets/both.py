"""Dataset setting and data loader for MNIST."""

from keras.utils import to_categorical
import torch
from torchvision import datasets, transforms
import torch.utils.data as utils
import params
import os
import numpy as np

n_classes = 12

def get_data(source):
    path = os.path.expanduser('~/data/visda17/reps/')
    if source:
        xx = np.load(path +'x_train.npy')
        yy = np.load(path +'y_train.npy')
    else:
        xx = np.load(path +'x_test.npy')
        yy = np.load(path +'y_test.npy')

    #xx = xx.reshape((len(xx),1,28,28))
    # yy = to_categorical(yy, n_classes)
    # print(yy.shape)

    tensor_x = torch.Tensor(xx)
    tensor_y = torch.Tensor(yy)

    dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    dataloader = utils.DataLoader(dataset,
            batch_size=params.batch_size,
            shuffle=True) # create your dataloader

    return dataloader
