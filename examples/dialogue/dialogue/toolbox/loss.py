import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MILCrossEntropyLoss(nn.Module):
    def __init__(self, method="max", lambda_=None):
        self.method = method
        self.lambda_ = lambda_
        super(MILCrossEntropyLoss, self).__init__()

    def forward(self, input_, target, bag, **kw):
        """
        :param input: list of Nk x C input matrix, length = M
        :param target: M LongTensor
        :return: output: scalar


        Example:
            Three-class multi-instance loss
            input:
                tensor([[-1.4585, -1.2980,  0.8303],
                        [ 0.1142, -1.0814, -2.3347],
                        [ 0.1737, -0.9211,  0.3935],
                        [ 1.8032, -0.7215,  0.8315],
                        [ 0.4165,  0.7164,  1.1776],
                        [ 1.7244,  0.8736,  0.8184]])
            target:
                tensor([1, 0, 2])
            bag:
                tensor([0, 1, 2, 2, 0, 2])
        """
        k = np.unique(bag.cpu().numpy()).shape[0]
        if self.method == "max":
            bag_input = torch.cat(
                [input_[bag == i].max(0)[0].unsqueeze(0) for i in range(k)], dim=0)
            return F.cross_entropy(bag_input, target, **kw)
        elif self.method == "mean":
            bag_input = torch.cat(
                [input_[bag == i].mean(0).unsqueeze(0) for i in range(k)], dim=0)
            return F.cross_entropy(bag_input, target, **kw)
        elif self.method == "sum":
            input_ = torch.sigmoid(input_)
            flatten_target = target.index_select(-1, bag)
            t1 = torch.sum(torch.log(1 - input_))
            x = input_.index_select(1, flatten_target).diag()
            t2 = torch.sum(torch.log(1 - x))
            return (-t1 + t2)

    def get_probs(self, input_, bag):
        k = np.unique(bag.cpu().numpy()).shape[0]
        if self.method == "max":
            bag_input = torch.cat(
                [input_[bag == i].max(0)[0].unsqueeze(0) for i in range(k)], dim=0)
            prob = torch.softmax(bag_input, dim=1)
        elif self.method == "mean":
            bag_input = torch.cat(
                [input_[bag == i].mean(0).unsqueeze(0) for i in range(k)], dim=0)
            prob = torch.softmax(bag_input, dim=1)
        elif self.method == "sum":
            prob = torch.cat(
                [(1 - torch.prod(1 - torch.sigmoid(input_[bag == i]), dim=0)).unsqueeze(0)
                 for i in range(k)], dim=0)
        return prob


