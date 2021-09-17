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
    def __init__(self, model, wbit, hw_specs, p_specs):
        self.model = model
        self.state_dict = model.state_dict()

        # rram property
        self.hw_specs = hw_specs
        self.p_specs = p_specs
        self.array_size = hw_specs["array_size"]
        self.col_per_weight = wbit // hw_specs["cellbit"]

        assert len(self.array_size) == 2, "Array size must contains two dims (row, col)"

    def array_by_layer(self):
        num_array = {}
        num_rows = {}
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
                num_rows[k] = n_rows
                total_param += o*c*kw*kh
        return num_array, total_param, num_rows
    
    def nops(self, ofm_size, narr, spars):
        return narr*ofm_size[2]*ofm_size[3]*(1-spars)
    
    def cell_energy(self, N, G, V, T):
        return G*N*(V**2)*T
    
    def get_relu_energy(self, relu_energy, nunits, ofm_size):
        return ofm_size[1]*ofm_size[2]*ofm_size[3]*relu_energy / nunits
    
    def get_buffer_energy(self, buffer_energy, bw, ofm_size):
        return ofm_size[1]*ofm_size[2]*ofm_size[3] * buffer_energy * 4
    
    def get_adder_energy(self, add_energy, nunits, ofm_size):
        return ofm_size[1]*ofm_size[2]*ofm_size[3] * add_energy / nunits

    def get_model_energy(self, ofm, grp_spars=0):
        arrays, total_param, num_rows = self.array_by_layer()
        
        total_energy = 0
        total_arr_energy = 0
        total_relu_energy = 0
        total_buf_energy = 0
        total_add_energy = 0

        for k, v in ofm.items():
            ops = self.nops(v, arrays[k], grp_spars)
            add_pow = math.ceil(math.log2(math.sqrt(num_rows[k])))
            add_stage = 2**add_pow
            
            arr_e = ops*self.hw_specs["energy"]
            relu_e = self.get_relu_energy(self.p_specs["relu_energy"], self.p_specs["relu_nunits"], v)
            buf_e = self.get_buffer_energy(self.p_specs["buffer_energy"], 4, v)
            add_e = self.get_adder_energy(self.p_specs["add_tree"][add_stage], self.p_specs["add_nunits"], v)
            e = arr_e + relu_e + buf_e + add_e

            # logger.info("Layer: {}, Nrows: {}, Array Energy: {}, ReLU Energy: {}\n".format(k, num_rows[k], arr_e, relu_e))
            total_energy += e
            total_arr_energy += arr_e
            total_relu_energy += relu_e
            total_buf_energy += buf_e
            total_add_energy += add_e
        
        model_energy = {
            "total": total_energy*1e+6,
            "array": total_arr_energy*1e+6,
            "relu": total_relu_energy*1e+6,
            "buffer": total_buf_energy*1e+6,
            "adder": total_add_energy*1e+6,
            "m_buffer": self.p_specs["buffer_energy"]*40e+3
        }
        return model_energy

    def get_model_area(self, ofm):
        arrays, total_param, num_rows = self.array_by_layer()
        mem_area = 0
        for k, v in ofm.items():
            n_arrays = arrays[k]
            mem_area += self.hw_specs["arr_area"] * n_arrays
        
        total_area = mem_area + self.p_specs["add_area"] + self.p_specs["buffer_area"] + self.p_specs["mbuffer_area"]
        # convert to um^2
        model_area = {
            "total": total_area*1e+6,
            "array": mem_area*1e+6,
            "relu": self.p_specs["relu_area"]*1e+6,
            "add_area": self.p_specs["add_area"]*1e+6,
            "buffer_area": self.p_specs["buffer_area"]*1e+6,
            "mbuffer_area": self.p_specs["mbuffer_area"]*1e+6
        }

        return model_area

    def reprogram_energy(self, spars, cell_specs):
        """
        Element-wise sparsity for reprogramming
        """
        arrays, total_param, num_rows = self.array_by_layer()
        sparse_weight = round(total_param * spars)
        
        cell_energy = self.cell_energy(cell_specs["NPulse"], cell_specs["R"], cell_specs["wdv"], cell_specs["wdp"])
        rp_energy = sparse_weight * cell_energy * self.col_per_weight*1e+6
        return rp_energy

    def finetune_reprogram_energy(self, ref_dict, ft_dict, cell_specs):
        rp_energy = 0
        for k, v_ref in ref_dict.items():
            v_ft = ft_dict[k]

            # add offset (TODO: make the offset general)
            v_ref = v_ref + 7.
            v_ft = v_ft + 7.

            # compute the average difference
            diff = v_ft.sub(v_ref).mean().ceil()
            print("Layer {}; Average level change = {}".format(k, diff))
            cell_change = cell_specs["R"]*diff

            cell_energy = self.cell_energy(cell_specs["NPulse"], cell_change, cell_specs["wdv"], cell_specs["wdp"])
            rp_energy += cell_energy*v_ft.numel()*self.col_per_weight*1e+6
        return rp_energy


    
    