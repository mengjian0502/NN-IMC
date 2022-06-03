"""
Pytorch model to RRAM mapping

Power and energy estimation (per image)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import time
import sys
import logging
import torch.utils.model_zoo as model_zoo
import torchvision
from torchsummary import summary
from dataset import get_loader 
from hw_eval import test, IMCEstimator

# HW specs
array_specs = {
    "cellbit":2,
    "array_size":[72,72],
    "energy": 12.79,
    "latency": 15.54,
    "area":11180.72
}

cell_specs = {
    "NPulse":20,
    "R": 4e-5,
    "wiv": 0.9,
    "wip": 100e-6,
    "wdv": 1.,
    "wdp": 100e-6
}

peripheral_specs = {

}

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet evaluation')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--log_file', type=str, default=None,
                    help='path to log file')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=3, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# quantization
parser.add_argument('--wbit', type=int, default=4, help='weight precision')
parser.add_argument('--abit', type=int, default=4, help='activation precision')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

activation = {}
def get_activation(name):
    def hook_fn(m, i, o):
        activation[name] = list(o.size())
    return hook_fn

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    logger = logging.getLogger('evaluation')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)

    # Prepare the data
    trainloader, testloader, num_classes = get_loader(args)
    
    # Build the model
    logger.info('==> Building model..\n')
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, num_classes)
    logger.info(net)
    
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    
    # register hook
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(get_activation(name))

    # run test    
    test_acc, val_loss = test(testloader, net, criterion)
    logger.info("Test acc = {}".format(test_acc))

    # hw eval
    est = IMCEstimator(net, wbit=4, hw_specs=array_specs)
    
    # inference energy
    model_energy = est.get_model_energy(activation, logger)
    logger.info("\nInference energy per image = {} uJ".format(model_energy))
    energy_per_val_set = model_energy * len(testloader)
    logger.info("({}) Inference energy per val set = {} uJ; size = {}".format(args.dataset, energy_per_val_set, len(testloader)))
    
    # reprogram energy
    sparsity = 5/100
    rp_energy = est.reprogram_energy(spars=sparsity, cell_specs=cell_specs)
    rp_energy = rp_energy*10**6
    logger.info("Reprogramming energy = {}uJ".format(rp_energy))
    logger.info("Reprogramming / Inference = {}%".format(rp_energy/energy_per_val_set*100))
    logger.info("Total / Inference = {}".format((energy_per_val_set + rp_energy)/energy_per_val_set))
    logger.info("Reprogramming percentage = {}".format(sparsity))

if __name__ == '__main__':
    main()
    
