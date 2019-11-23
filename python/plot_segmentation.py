#### compare image, ground truth and prediction

import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils


def plot_pred(img, ground_truth, pred, iter):
    label2color = {}
    color2label = {}
    label2index = {}
    index2label = {}
    means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of three channels in the order of BGR
    f = open('CamVid/label_colors.txt', "r").read().split("\n")[:-1]  # ignore the last empty line
    for idx, line in enumerate(f):
        label = line.split()[-1]
        color = tuple([int(x) for x in line.split()[:-1]])
        #print(label, color)
        label2color[label] = color
        color2label[color] = label
        label2index[label] = idx
        index2label[idx] = label

    # pdb.set_trace()
    if str(ground_truth.dtype).split('.')[0] == 'torch':
        ground_truth = ground_truth.data.cpu().numpy()
    if str(pred.dtype).split('.')[0] == 'torch':
        pred = pred.data.cpu().numpy()

    _, h, w = pred.shape
    pred_R = np.zeros((h, w))
    pred_G = np.zeros((h, w))
    pred_B = np.zeros((h, w))
    pred_BGR = np.zeros((3, h, w))

    # label2color is RGB
    for index1 in range(h):
        for index2 in range(w):
            pred_R[index1, index2] = label2color[index2label[pred[0, index1, index2]]][0]
            pred_G[index1, index2] = label2color[index2label[pred[0, index1, index2]]][1]
            pred_B[index1, index2] = label2color[index2label[pred[0, index1, index2]]][2]

    # RGB color
    pred_BGR[0] = pred_R / 255
    pred_BGR[1] = pred_G / 255
    pred_BGR[2] = pred_B / 255
    # pdb.set_trace()
    _, h, w = ground_truth.shape
    ground_truth_R = np.zeros((h, w))
    ground_truth_G = np.zeros((h, w))
    ground_truth_B = np.zeros((h, w))
    ground_truth_BGR = np.zeros((3, h, w))

    # label2color is RGB
    for index1 in range(h):
        for index2 in range(w):
            ground_truth_R[index1, index2] = label2color[index2label[ground_truth[0, index1, index2]]][0]
            ground_truth_G[index1, index2] = label2color[index2label[ground_truth[0, index1, index2]]][1]
            ground_truth_B[index1, index2] = label2color[index2label[ground_truth[0, index1, index2]]][2]

    # RGB color
    ground_truth_BGR[0] = ground_truth_R / 255
    ground_truth_BGR[1] = ground_truth_G / 255
    ground_truth_BGR[2] = ground_truth_B / 255

    img[:, 0, ...].add_(means[0])
    img[:, 1, ...].add_(means[1])
    img[:, 2, ...].add_(means[2])
    batch_size = len(img)

    grid = utils.make_grid(img)

    plt.figure(figsize=(22, 12))
    plt.subplot(1, 3, 1)
    plt.imshow(grid.cpu().numpy()[::-1].transpose((1, 2, 0)))
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_BGR.transpose((1, 2, 0)))
    plt.subplot(1, 3, 3)
    plt.imshow(pred_BGR.transpose((1, 2, 0)))
    plt.savefig('result/result' + str(iter) + '.png')
    plt.close()
