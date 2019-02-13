"""Test script to classify target data."""

import torch
import torch.nn as nn
import numpy as np
from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    acc = np.zeros(13)
    total = np.zeros(13)

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = classifier(encoder(images))

        pred_cls = preds.data.max(1)[1]
        total[12] += len(labels)
        acc[12] += pred_cls.eq(labels.data.to(dtype=torch.int64)).cpu().sum()

        for i in range(12):
            temp = (labels == i)
            total[i]+= int(sum(temp))
            temp_labels = labels.to(dtype=torch.int64) + 100*(1-temp.to(dtype=torch.int64))
            acc[i] += (pred_cls.eq(temp_labels.data.to(dtype=torch.int64)).cpu()).sum()
            

    acc = acc/total *100
    acc = np.round(acc,1)

    print(acc)
