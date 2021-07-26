"""
Hardware performance analysis
"""

import os
import sys
import time
import math
import shutil
import tabulate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test(testloader, net, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    test_loss = 0

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            mean_loader = []
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            test_loss += loss.item()

            batch_time.update(time.time() - end)
            end = time.time()

            # single image test
            if batch_idx == 0:
                break
    return top1.avg, losses.avg

class IMCEstimator(object):
    def __init__(self, model, array_size, cellbit, wbit):
        self.model = model
        self.state_dict = model.state_dict()

        # rram property
        self.array_size = array_size
        self.col_per_weight = wbit // cellbit

        assert len(array_size) == 2, "Array size must contains two dims (row, col)"

    def num_array(self):
        for k, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight
                o, c, kw, kh = weight.size()

                # consumption
                n_rows = math.ceil(c*kw*kh / self.array_size[0])
                n_cols = math.ceil(o / self.array_size[1])*self.col_per_weight
                n_array = n_rows * n_cols

                