import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import lfilter

def filter(params=(30,1), inp=np.zeros((100,1))):
    (n, a) = params
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    return lfilter(b, a, inp)



# Import labels
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


classes = np.arange(len(label2color))

# Files
scores_dir = os.path.join("..", 'scores')
fold_list = os.listdir(scores_dir)
models = []

# IoU
histories = {}
for iter, item in enumerate(fold_list):
    print(str(iter)+'. ' + item)
    file_path = os.path.join(scores_dir, item, 'meanIU.npy')
    which_model = item.partition("-")[0]
    models.append(which_model)
    hist = np.load(file_path)
    hist = hist[:np.max(np.nonzero(hist)), :]
    histories[which_model] = hist








which_model = models[0]
hist = histories[which_model]

ax = plt.subplot(1,1,1)
p = []
for iter,cl in enumerate(classes):
    plt.plot(100*filter(inp=hist[:,cl]), label=str(cl))
ax.set_title(which_model+ " class IoU", fontweight="bold")
ax.set_xlabel("epoch")
ax.set_ylabel("IoU %")


#Beautify
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

r1,r2 = plt.xlim()
r = (int(r1), int(r2))
for y in range(10, 91, 10):
    plt.plot(range(*r), [y] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3)

plt.savefig(os.path.join("files","Single_model_IoU_training.pdf"), bbox_inches="tight")
plt.show()
plt.close()




## BARH PLOT
width = 0.35
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


ax.barh(y_pos, 100*performance,height=0.8, align='center',color=colors,zorder=10)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)

ax.set_xlabel('IoU')
ax.set_title(which_model+' IoU accuracy', fontweight="bold")
plt.ylim(min(y_pos)+0.5, max(y_pos)-0.5)


#Beautify
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)


r1,r2 = plt.ylim()
r = (int(r1-0.5), int(r2+0.5))
for y in range(1, 90, 10):
    plt.plot([y] * len(range(*r)),range(*r), "--", lw=0.5, color="black", alpha=0.3,zorder=1)

ax.invert_yaxis()  # labels read top-to-bottom

plt.savefig(os.path.join("files","Single_model_IoU_classes_colored.pdf"), bbox_inches="tight")
plt.show()
plt.close()


#Combined bar chart

width = 0.2
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(5, 10))
people = np.array(list(label2color.keys()))[sorting]
colors = np.array(list(color2label.keys()))[sorting]/255
y_pos = np.arange(len(people))




plt.barh(y_pos, 100*np.max(histories[models[0]],axis=0)[sorting], width, label=models[0],zorder=10)
plt.barh(y_pos + width, 100*np.max(histories[models[1]], axis=0)[sorting], width,label=models[1],zorder=10)
plt.barh(y_pos - width, 100*np.max(histories[models[2]], axis=0)[sorting], width, label=models[2],zorder=10)

plt.yticks(y_pos,people)
ax.set_title("Model comparison on IoU", fontweight="bold")
ax.set_xlabel("IoU %")
plt.legend(loc='best')

#Beautify
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()



r1,r2 = plt.ylim()
r = (int(r1-0.5), int(r2+0.5))
for y in range(1, 90, 10):
    plt.plot([y] * len(range(*r)),range(*r), "--", lw=0.5, color="black", alpha=0.3,zorder=1)

ax.invert_yaxis()  # labels read top-to-bottom

plt.savefig(os.path.join("files","All_models_IoU_classes_comparison.pdf"), bbox_inches="tight")
plt.show()
plt.close()


# Pixel accuracy training
histories = {}
models = []
for iter, item in enumerate(fold_list):
    # print(str(iter)+'. ' + item)
    file_path = os.path.join(scores_dir, item, 'meanPixel.npy')
    which_model = item.partition("-")[0]
    models.append(which_model)
    hist = np.load(file_path)
    hist = hist[:np.max(np.nonzero(hist))]  # Crop until trained epoch
    histories[which_model] = hist



ax = plt.subplot(1,1,1)
p = []
for iter,mdl in enumerate(models):
    plt.plot(100*filter(inp=histories[mdl]), label=str(mdl))
ax.set_title("Model pixel accuracy", fontweight="bold")
ax.set_xlabel("epoch")
ax.set_ylabel("Px accuracy %")


#Beautify
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

r1,r2 = plt.xlim()
r = (int(r1), int(r2))
for y in range(10, 91, 10):
    plt.plot(range(*r), [y] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3)


plt.legend(loc='best')

#Beautify
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.savefig(os.path.join("files","Models_pixel_accuracy.pdf"), bbox_inches="tight")
plt.show()
plt.close()


#Plot learning rate




# Pixel accuracy training
histories = {}
models = []
for iter, item in enumerate(fold_list):
    # print(str(iter)+'. ' + item)
    file_path = os.path.join(scores_dir, item, 'learning_rate.npy')
    which_model = item.partition("-")[0]
    models.append(which_model)
    hist = np.load(file_path)
    hist = hist[:np.max(np.nonzero(hist))]  # Crop until trained epoch
    histories[which_model] = hist


ax = plt.subplot(1,1,1)
p = []
for iter,mdl in enumerate(models):
    plt.plot(histories[mdl], label=mdl,zorder=10)


ax.set_title("Learning rate", fontweight="bold")
ax.set_xlabel("epoch")
ax.set_ylabel("lr")

#
r1,r2 = plt.xlim()
r = (int(r1), int(r2))

r1y, r2y = plt.ylim()

for y in list(plt.yticks()[0][1:-1]):
    plt.plot(range(*r), [y] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3,zorder=1)



#Beautify
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.legend(loc='best')

plt.savefig(os.path.join("files","Learning_rate.pdf"), bbox_inches="tight")
plt.show()
plt.close()