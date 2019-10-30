import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable as V

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def cuda(x):
    if torch.cuda.is_available():
        return x.cuda()  
    else :
        return x


class LossMulti:
    def __init__(self, jaccard_weight=0, num_classes=2):

        #self.nll_loss = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax()
        self.nll_loss = nn.NLLLoss()
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        output = self.logsoftmax(outputs, dim =1)
        loss = self.nll_loss(output, targets)

        eps=1e-7        
        num_classes = outputs.shape[1]
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(outputs, dim=1)
        true_1_hot = true_1_hot.type(outputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        score = 2. * intersection / (cardinality + eps)
        jacard = intersection / (cardinality - intersection + eps)
        dice_loss = 1- score.mean()
        return loss , dice_loss, jacard.mean()

