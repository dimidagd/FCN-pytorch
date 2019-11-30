
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from Cityscapes_loader import CityScapesDataset as CityscapesDataset
from CamVid_loader import CamVidDataset, show_batch
from plot_segmentation import plot_pred
from matplotlib import pyplot as plt
import pdb
import numpy as np
import time
import sys
import os


n_class = 32

batch_size = 5

epochs = 500

momentum = 0
w_decay = 1e-5
step_size = 4
gamma = 0.9

train_vgg = True



argv1 = 'CamVid'
if argv1 == 'CamVid':
    root_dir = "CamVid/"
else:
    root_dir = "CityScapes/"
train_file = os.path.join(root_dir, "train.csv")
val_file = os.path.join(root_dir, "val.csv")

if argv1 == 'CamVid':
    print("Using CamVid dataset")
    train_data = CamVidDataset(csv_file=train_file, phase='train')
else:
    train_data = CityscapesDataset(csv_file=train_file, phase='train')

print("Running DataLoader from ", train_file)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=8)

if argv1 == 'CamVid':
    val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
else:
    val_data = CityscapesDataset(csv_file=val_file, phase='val', flip_rate=0)

print("loading validation file from", val_file)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)









use_gpu = torch.cuda.is_available()

def val(fcn_model, model):

    fcn_model.eval()  # Deactivate dropout and batch normalization or inconsistent inference

    for iter, batch in enumerate(val_loader):
        # print('Validation Iteration ',iter)
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        if iter%10 == 0:
            plot_pred(inputs, target, pred, iter, model)



if __name__ == "__main__":
    print('Validating without train')

    fold_list = os.listdir('models')
    for model in fold_list:
        if model[-4:] == 'FULL':
            print("Loading model " + model)
            fcn_model = torch.load(model)
            val(fcn_model)

