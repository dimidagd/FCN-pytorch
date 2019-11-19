# -*- coding: utf-8 -*-

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

from matplotlib import pyplot as plt
import pdb
import numpy as np
import time
import sys
import os



n_class = 32

batch_size = 3

epochs = 500

momentum = 0
w_decay = 1e-5
step_size = 4
gamma = 0.9



print("Loading FCN")
print("Please choose submodel")
print("1. FCN8s")
print("2. FCN16s")
print("3. FCN32s")
print("4. FCNs")
inp = int(input(": "))

if inp == 1:
    submodel = '8'
    lr = 1e-4
elif inp == 2:
    submodel = '16'
    lr = 1e-3  # FCN16s
elif inp == 3:
    submodel = '32'
    lr = 1e-2  # FCN32s
elif inp == 4:
    submodel =''
    lr = 1e-2
else:
    print("Invalid input!")


configs = "FCN{}s-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(submodel,
    batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

argv1 = 'CamVid'
if argv1 == 'CamVid':
    root_dir = "CamVid/"
else:
    root_dir = "CityScapes/"
train_file = os.path.join(root_dir, "train.csv")
val_file = os.path.join(root_dir, "val.csv")

# create dir for model
model_dir = "models"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

if argv1 == 'CamVid':
    print("Using CamVid dataset")
    train_data = CamVidDataset(csv_file=train_file, phase='train')
else:
    train_data = CityscapesDataset(csv_file=train_file, phase='train')

print("Running DataLoader from ", train_file)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

if argv1 == 'CamVid':
    val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
else:
    val_data = CityscapesDataset(csv_file=val_file, phase='val', flip_rate=0)

print("loading validation file from", val_file)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

print("Loading VGG")
vgg_model = VGGNet(requires_grad=True, remove_fc=True)


if inp == 1:
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)
elif inp == 2:
    fcn_model = FCN16s(pretrained_net=vgg_model, n_class=n_class)
elif inp == 3:
    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)
elif inp == 4:
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
else:
    print("Invalid input!")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
start_epoch = 0


print("Use GPU :", use_gpu)
if use_gpu:
    print("Using GPU, loading cuda shiat")
    ts = time.time()
    print("Loading vgg on cuda")
    vgg_model = vgg_model.cuda()
    print("Loading FC on cuda")
    fcn_model = fcn_model.cuda()
    print("Number of GPUs", num_gpu)
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))




checkpoint_load_chk = ''
# Load checkpoint after cuda DataParallel initialization
if os.path.exists(model_path):
    checkpoint_load_chk = str(input("Found previous checkpoint, use it? "))
    if os.path.exists(model_path) and checkpoint_load_chk == 'y':
        print("Load last checkpoint..")
        checkpoint = torch.load(model_path)
        fcn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        # Option to override learning params:
        print("Would you like to reset learning rate to a value?(d:default,c: custom,n: keep checkpoint :")
        opt = input(': ')
        if opt == 'd':
            for g in optimizer.param_groups:
                g['lr'] =  lr
        elif opt == 'c':
            lr_val = float(input('Insert value for lr: '))
            for g in optimizer.param_groups:
                g['lr'] = lr_val
        elif opt == 'n':
            pass
    else:
        start_epoch = 0


# TODO: Load other layers params as well

#  Initialize from previous models if didnt use checkpoint
if checkpoint_load_chk != 'y':
    if inp == 2 and os.path.exists(model_path.replace("FCN16s", "FCN32s")) :
        inp2 = str(input("Found previously trained FCN32s, put it as initialization point? (y/n) "))
        if inp2 == 'y':
            if use_gpu:
                checkpoint = torch.load(model_path.replace("FCN16s", "FCN32s"))
            else:
                checkpoint = torch.load(model_path.replace("FCN16s", "FCN32s"), map_location=torch.device('cpu'))
            classifier_params = {}
            classifier_params['weight'] = checkpoint['model_state_dict']['module.classifier.weight']
            classifier_params['bias'] = checkpoint['model_state_dict']['module.classifier.bias']
            if use_gpu:
                fcn_model.module.classifier.load_state_dict(classifier_params)
            else:
                fcn_model.classifier.load_state_dict(classifier_params)

    if inp == 1 and os.path.exists(model_path.replace("FCN8s", "FCN16s")):
        inp2 = str(input("Found previously trained FCN16s, put it as initialization point? (y/n) "))
        if inp2 == 'y':
            if use_gpu:
                checkpoint = torch.load(model_path.replace("FCN8s", "FCN16s"))
            else:
                checkpoint = torch.load(model_path.replace("FCN8s", "FCN16s"), map_location=torch.device('cpu'))
            classifier_params = {}
            classifier_params['weight'] = checkpoint['model_state_dict']['module.classifier.weight']
            classifier_params['bias'] = checkpoint['model_state_dict']['module.classifier.bias']
            if use_gpu:
                fcn_model.module.classifier.load_state_dict(classifier_params)
            else:
                fcn_model.classifier.load_state_dict(classifier_params)

    if inp == 4 and os.path.exists(model_path.replace("FCNs", "FCN8s")):
        inp2 = str(input("Found previously trained FCN8s, put it as initialization point? (y/n) "))
        if inp2 == 'y':
            if use_gpu:
                checkpoint = torch.load(model_path.replace("FCNs", "FCN8"))
            else:
                checkpoint = torch.load(model_path.replace("FCNs", "FCN8s"), map_location=torch.device('cpu'))

            classifier_params = {}
            classifier_params['weight'] = checkpoint['model_state_dict']['module.classifier.weight']
            classifier_params['bias'] = checkpoint['model_state_dict']['module.classifier.bias']
            if use_gpu:
                fcn_model.module.classifier.load_state_dict(classifier_params)
            else:
                fcn_model.classifier.load_state_dict(classifier_params)





# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)
train_loss = np.zeros(epochs)
learning_rate = np.zeros(epochs)

def train():
    fcn_model.train()  # Reactivate batch norm etc
    for epoch in range(start_epoch, epochs):

        print(optimizer)
        ts = time.time()
        loss_acc = 0
        for iter, batch in enumerate(train_loader):

            optimizer.zero_grad()  # Necessary to reset before optimizer.step()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)

            loss = criterion(outputs, labels)
            loss_acc += loss.data/(batch_size)
            loss.backward() # TODO: Should I sum the loss on every batch, calculate mean of it and do optimizer.step()?  ANS: We are performing mini-batch training, the loss is already the average of the batch. Increasing the bathch size should make training more smooth(less stochastic), but needs more memory.
            optimizer.step() #

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data))
        scheduler.step()
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        # pdb.set_trace()
        train_loss[epoch] = loss_acc
        learning_rate[epoch] = optimizer.param_groups[0]['lr']
        np.save(os.path.join(score_dir, "train_loss"), train_loss)
        np.save(os.path.join(score_dir, "learning_rate"), optimizer.param_groups[0]['lr'])
        torch.save(fcn_model, model_path+'FULL')


        # Alternative save
        torch.save({
            'epoch': epoch,
            'model_state_dict': fcn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': loss
        }, model_path)
        #pdb.set_trace()
        val(epoch)


def val(epoch):
    fcn_model.eval()  # Deactivate dropout and batch normalization or inconsistent inference
    total_ious = []
    pixel_accs = []
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
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)



# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/ma ster/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        #print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    #print("pixel accuracy debug", correct, total)
    return correct / total


# def heat_maps(batch, predictions):
#     show_batch(batch)
#
#     heatmap = np.zeros((*predictions.shape, 3))
#
#     r = np.zeros(predictions.shape)
#     b = 0*r
#     g = 0*r
#
#     label2color = {}
#     color2label = {}
#     label2index = {}
#     index2label = {}
#
#     f = open('CamVid/label_colors.txt', "r").read().split("\n")[:-1]  # ignore the last empty line
#
#     for idx, line in enumerate(f):
#         label = line.split()[-1]
#         color = tuple([int(x) for x in line.split()[:-1]])
#         print(label, color)
#         label2color[label] = color
#         color2label[color] = label
#         label2index[label] = idx
#         index2label[idx] = label
#
#     heatmap = heatmap.reshape(3, heatmap.shape[1] * heatmap.shape[2])
#
#     for i, pixel in enumerate(predictions.squeeze(0)):
#
#         heatmap[:,i] = np.array(label2color[index2label[pixel]])
#
#     heatmap.reshape((3,*predictions.shape))






if __name__ == "__main__":
    print('Validating without train')
    val(0)  # show the accuracy before training
    print('Starting train loop')
    train()
