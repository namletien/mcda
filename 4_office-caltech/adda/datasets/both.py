import torch
from torchvision import datasets, transforms
import torch.utils.data as utils
import params
import os
import numpy as np
from scipy.io import loadmat

n_classes = 12


def load_office(name):
    data_dir = os.path.expanduser('~/data/office-caltech/')
    file = data_dir + name + '.mat'
    source = loadmat(file)
    x = source['fts']
    y = source['labels']
    y = y.reshape(len(y[0]))-1
    return x, y

def get_data(name):
    xx, yy = load_office(name)

    tensor_x = torch.Tensor(xx)
    tensor_y = torch.Tensor(yy)

    dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    dataloader = utils.DataLoader(dataset,
            batch_size=params.batch_size,
            shuffle=True) # create your dataloader

    return dataloader
