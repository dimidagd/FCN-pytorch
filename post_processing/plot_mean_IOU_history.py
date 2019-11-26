import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import lfilter

def filter(params=(30,1), inp=np.zeros((100,1))):
    (n, a) = params
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    return lfilter(b, a, inp)


scores_dir = os.path.join("..", 'scores')
fold_list = os.listdir(scores_dir)

print("Input results folder")
for iter, item in enumerate(fold_list):
    print(str(iter)+'. ' + item)

inp = int(input(': '))
file_path = os.path.join(scores_dir, fold_list[inp], 'meanIU.npy')
which_model = fold_list[inp].partition("-")[0]
hist = np.load(file_path)
hist = hist[:np.max(np.nonzero(hist)),:]

classes = np.array(range(32))




ax = plt.subplot(1,1,1)
p = []
for iter,cl in enumerate(classes):
    plt.plot(filter(inp=hist[:,cl]), label=str(cl))
ax.set_title(which_model+ " class IoU", fontweight="bold")
ax.set_xlabel("epoch")
ax.set_ylabel("IoU")
plt.show()
plt.close()


## BARH PLOT

label2color = {}
color2label = {}
label2index = {}
index2label = {}
means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of three channels in the order of BGR
f = open('../CamVid/label_colors.txt', "r").read().split("\n")[:-1]  # ignore the last empty line
for idx, line in enumerate(f):
    label = line.split()[-1]
    color = tuple([int(x) for x in line.split()[:-1]])
    # print(label, color)
    label2color[label] = color
    color2label[color] = label
    label2index[label] = idx
    index2label[idx] = label



plt.rcdefaults()
fig, ax = plt.subplots(figsize=(5, 10))
people = np.array(list(label2color.keys()))
colors = np.array(list(color2label.keys()))/255
y_pos = np.arange(len(people))

data = np.max(hist,axis=0) # performance data

# Sorting of values from max to min
sorting = np.flip(np.argsort(data))
performance =  data[sorting]
people = people[sorting]
colors = colors[sorting]
# Skipping empty and nan classes

filter_empty_nan = performance > 0.01
performance = performance[filter_empty_nan]
people = people[filter_empty_nan]
colors = colors[filter_empty_nan]
y_pos = np.arange(len(colors))


ax.barh(y_pos, performance,height=0.8, align='center',color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)

ax.set_xlabel('IoU')
ax.set_title(which_model+' IoU accuracy', fontweight="bold")
plt.ylim(min(y_pos)+0.5, max(y_pos)-0.5)
ax.invert_yaxis()  # labels read top-to-bottom
plt.show()
plt.close()