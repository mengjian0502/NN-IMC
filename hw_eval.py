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
    def __init__(self, model, wbit, hw_specs):
        self.model = model
        self.state_dict = model.state_dict()

        # rram property
        self.hw_specs = hw_specs
        self.array_size = hw_specs["array_size"]
        self.col_per_weight = wbit // hw_specs["cellbit"]

        assert len(self.array_size) == 2, "Array size must contains two dims (row, col)"

    def array_by_layer(self):
        num_array = {}
        total_param = 0
        for k, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight
                o, c, kw, kh = weight.size()

                # consumption
                n_rows = math.ceil(c*kw*kh / self.array_size[0])
                n_cols = math.ceil(o / self.array_size[1])*self.col_per_weight
                n_array = n_rows * n_cols

                num_array[k] = n_array
                total_param += o*c*kw*kh
        return num_array, total_param
    
    def nops(self, ofm_size, narr, spars):
        return narr*ofm_size[0]*ofm_size[1]*(1-spars)
    
    def cell_energy(self, N, G, V, T):
        return G*N*(V**2)*T
    
    def get_relu_energy(self):
        pass

    def get_model_energy(self, ofm, logger):
        arrays, total_param = self.array_by_layer()
        model_energy = 0
        for k, v in ofm.items():
            ops = self.nops(v, arrays[k], 0)
            import pdb;pdb.set_trace()
            e = ops*self.hw_specs["energy"]
            
            logger.info("Layer: {}, Energy: {}".format(k, e))
            model_energy += e
        
        # return the value in terms of micro juels
        model_energy = model_energy / 1e6
        return model_energy

    def get_model_area(self):
        pass

    def reprogram_energy(self, spars, cell_specs):
        """
        Element-wise sparsity for reprogramming
        """
        arrays, total_param = self.array_by_layer()
        sparse_weight = total_param * spars
        
        cell_energy = self.cell_energy(cell_specs["NPulse"], cell_specs["R"], cell_specs["wdv"], cell_specs["wdp"])
        rp_energy = sparse_weight * cell_energy * self.col_per_weight
        return rp_energy
    
    