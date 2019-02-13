import torch
from torchvision import datasets, transforms
import torch.utils.data as utils
import params
import os
import numpy as np
from sklearn.datasets import load_svmlight_files
import scipy

def load_office(name,train = True):
    data_dir = os.path.expanduser('~/data/amazon_review/')
    if train == True:
        file = data_dir + name + '_train.svmlight'
    else:
        file = data_dir + name + '_test.svmlight'
    print(file)
    x, y  = load_svmlight_files([file])   
    x = scipy.sparse.csr_matrix.todense(x)     
    y = np.array((y + 1) / 2, dtype=int)
    return x, y

def get_data(name, train = True):
    xx, yy = load_office(name, train)

    tensor_x = torch.Tensor(xx)
    tensor_y = torch.Tensor(yy)

    dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    dataloader = utils.DataLoader(dataset,
            batch_size=params.batch_size,
            shuffle=True) # create your dataloader

    return dataloader
