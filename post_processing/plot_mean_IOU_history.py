import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import lfilter
from scipy.signal import lfiltic
def filter(params=(30,1), inp=np.zeros((100,1)),init=None):
    (n, a) = params
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    if init is not None:
        i = lfiltic(b,a,init,x=0)
    else:
        i = None
    return lfilter(b, a, inp,zi=i)

def smooth(x, window_len=11, window='hanning'):

    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


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
    if item[-4:] == 'True': which_model = which_model + 'VGG_trained'
    models.append(which_model)
    hist = np.load(file_path)
    hist = hist[:np.max(np.nonzero(hist)), :]
    histories[which_model] = hist








which_model = models[3]
hist = histories[which_model]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(27, 20))

p = []
colors = np.array(list(color2label.keys()))/255
for iter,cl in enumerate(classes):
    data = 100*filter(inp=hist[:,cl])
    if np.max(data)>1:
        ax1.plot(data, label=str(cl),color=colors[cl],linewidth=6)
ax1.set_title("Training",fontsize=30)
ax1.set_xlabel("epoch",fontsize=30)
ax1.set_ylabel("IoU %",fontsize=30)
ax1.tick_params(labelsize=20)
#Beautify
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)

r1,r2 = plt.xlim()
r = (50, 300)
for y in range(10, 91, 10):
    ax1.plot(range(*r), [y] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3)

# plt.savefig(os.path.join("files","Single_model_IoU_training.pdf"), bbox_inches="tight")
# plt.show()
# plt.close()




## BARH PLOT
width = 0.35
plt.rcdefaults()


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


ax2.barh(y_pos, 100*performance,height=0.8, align='center',color=colors,zorder=10)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(people)
ax2.tick_params(labelsize=20)

ax2.set_xlabel('IoU %',fontsize=30)
ax2.set_title("Final",fontsize=30)
ax2.set_ylim(min(y_pos)-0.5, max(y_pos)+0.5)


#Beautify
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)


r1,r2 = plt.ylim()
r = (int(r1-0.5), int(r2+0.5))
for y in range(1, 90, 10):
    ax2.plot([y] * len(range(*r)),range(*r), "--", lw=0.5, color="black", alpha=0.3,zorder=1)

ax2.invert_yaxis()  # labels read top-to-bottom

plt.savefig(os.path.join("files","Single_model_IoU_classes_colored.pdf"),bbox_inches='tight') #, bbox_inches="tight")
plt.show()
plt.close()


#Combined bar chart

width = 0.2
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(5, 10))

people = np.array(list(label2color.keys()))[sorting]
colors = np.array(list(color2label.keys()))[sorting]/255


people = people[filter_empty_nan]
colors = colors[filter_empty_nan]

sorting_filter_nans = (sorting * filter_empty_nan)[sorting * filter_empty_nan > 0]
y_pos = np.arange(len(people))



plt.barh(y_pos - width, 100*np.max(histories[models[3]], axis=0)[sorting_filter_nans], width, label=models[3],zorder=10)
plt.barh(y_pos, 100*np.max(histories[models[5]],axis=0)[sorting_filter_nans], width, label=models[5],zorder=10)
plt.barh(y_pos + width, 100*np.max(histories[models[6]], axis=0)[sorting_filter_nans], width,label=models[6],zorder=10)


plt.yticks(y_pos,people)
ax.set_title("Model comparison on IoU")
ax.set_xlabel("IoU %",fontsize=20)
ax.legend(loc='best')


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
    ax.plot([y] * len(range(*r)),range(*r), "--", lw=0.5, color="black", alpha=0.3,zorder=1)

ax.invert_yaxis()  # labels read top-to-bottom

handles, labels = ax.get_legend_handles_labels()
# ax.legend(reversed(handles), reversed(labels), title='Models', loc='best')

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
    if item[-4:] == 'True': which_model = which_model + 'VGGtrained'
    models.append(which_model)
    hist = np.load(file_path)
    hist = hist[:np.max(np.nonzero(hist))]  # Crop until trained epoch
    histories[which_model] = hist



ax = plt.subplot(1,1,1)
p = []
for iter,mdl in enumerate(models):
    if histories[mdl][-1]> 0.4:
        ax.plot(100 * filter(inp=histories[mdl]), label=str(mdl),linewidth=4)

ax.set_title("Model pixel accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("Pixel accuracy %")


r1,r2 = plt.xlim()
r = (int(r1), int(r2))
for y in range(10, 91, 10):
    ax.plot(range(*r), [y] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3)


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



# Learning rate
histories = {}
models = []
for iter, item in enumerate(fold_list):
    # print(str(iter)+'. ' + item)
    file_path = os.path.join(scores_dir, item, 'learning_rate.npy')
    file_path2 = os.path.join(scores_dir, item, 'meanPixel.npy')
    check_if_good_one = np.load(file_path2)
    which_model = item.partition("-")[0]
    if not item[-4:] == 'True':
        models.append(which_model)
        hist = np.load(file_path)
        hist = hist[:np.max(np.nonzero(hist))]  # Crop until trained epoch
        histories[which_model] = hist


ax = plt.subplot(1,1,1)
p = []
for iter,mdl in enumerate(models):
    ax.plot(histories[mdl], label=mdl,zorder=10,linewidth=3)


ax.set_title("Learning rate",fontsize=20)
ax.set_xlabel("epoch",fontsize=15)
ax.set_ylabel("lr",fontsize=15)

#
r1,r2 = plt.xlim()
r = (int(r1), int(r2))

r1y, r2y = plt.ylim()

for y in list(plt.yticks()[0][1:-1]):
    ax.plot(range(*r), [y] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3,zorder=1)



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

# Combine last 3 =====================

fig1 = plt.figure(figsize=(27, 20))
# fig2 = plt.figure()  # more figures are easily accessible
# fig3 = plt.figure()  # more figures are easily accessible

ax11 = fig1.add_subplot(221)  # add subplot into first position in a 2x2 grid (upper left)
ax12 = fig1.add_subplot(223, sharex=ax11)  # add to third position in 2x2 grid (lower left) and sharex with ax11
ax13 = fig1.add_subplot(122)  # add subplot to cover both upper and lower right, in a 2x2 grid. This is the same as the rightmost panel in a 1x2 grid.
# ax21 = fig2.add_subplot(211)  # add axes to the extra figures
# ax21 = fig2.add_subplot(212)  # add axes to the extra figures
# ax31 = fig3.add_subplot(111)  # add axes to the extra figures



# Learning rate

histories = {}
models = []
for iter, item in enumerate(fold_list):
    # print(str(iter)+'. ' + item)
    file_path = os.path.join(scores_dir, item, 'learning_rate.npy')
    file_path2 = os.path.join(scores_dir, item, 'meanPixel.npy')
    check_if_good_one = np.load(file_path2)
    which_model = item.partition("-")[0]
    if not item[-4:] == 'True':
        models.append(which_model)
        hist = np.load(file_path)
        hist = hist[:np.max(np.nonzero(hist))]  # Crop until trained epoch
        histories[which_model] = hist


p = []
for iter,mdl in enumerate(models):
    ax11.plot(histories[mdl], label=mdl,zorder=10,linewidth=2)


ax11.set_title("Learning rate",fontsize=10)
ax11.set_xlabel("epoch",fontsize=10)
ax11.set_ylabel("lr",fontsize=10)

#
r1,r2 = ax11.get_xlim()
r = (int(r1), int(r2))

r1y, r2y = ax11.get_ylim()

for y in list(ax11.get_yticks()[1:-1]):
    ax11.plot(range(*r), [y] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3,zorder=1)



#Beautify
ax11.spines["top"].set_visible(False)
ax11.spines["bottom"].set_visible(False)
ax11.spines["right"].set_visible(False)
ax11.spines["left"].set_visible(False)
ax11.get_xaxis().tick_bottom()
ax11.get_yaxis().tick_left()
ax11.legend(loc='best',fontsize=7)




# Pixel accuracy training
histories = {}
models = []
for iter, item in enumerate(fold_list):
    # print(str(iter)+'. ' + item)
    file_path = os.path.join(scores_dir, item, 'meanPixel.npy')

    which_model = item.partition("-")[0]
    if item[-4:] == 'True': which_model = which_model + 'VGGtrained'
    models.append(which_model)
    hist = np.load(file_path)
    hist = hist[:np.max(np.nonzero(hist))]  # Crop until trained epoch
    histories[which_model] = hist



p = []
for iter,mdl in enumerate(models):
    if histories[mdl][-1]> 0.4:
        ax12.plot(100 * filter(inp=histories[mdl]), label=str(mdl),linewidth=2)

ax12.set_title("Model pixel accuracy")
ax12.set_xlabel("epoch")
ax12.set_ylabel("Pixel accuracy %")


#
r1,r2 = ax12.get_xlim()
r = (int(r1), int(r2))

r1y, r2y = ax12.get_ylim()

for y in list(ax12.get_yticks()[1:-1]):
    ax12.plot(range(*r), [y] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3,zorder=1)



ax12.legend(loc='best',fontsize=7)

#Beautify
ax12.spines["top"].set_visible(False)
ax12.spines["bottom"].set_visible(False)
ax12.spines["right"].set_visible(False)
ax12.spines["left"].set_visible(False)
ax12.get_xaxis().tick_bottom()
ax12.get_yaxis().tick_left()





#Combined bar chart


# IoU
histories = {}
models = []
for iter, item in enumerate(fold_list):
    print(str(iter)+'. ' + item)

    file_path = os.path.join(scores_dir, item, 'meanIU.npy')

    which_model = item.partition("-")[0]
    if item[-4:] == 'True': which_model = which_model + 'VGG_trained'
    models.append(which_model)
    hist = np.load(file_path)
    hist = hist[:np.max(np.nonzero(hist)), :]
    histories[which_model] = hist


width = 0.2

# fig, ax = plt.subplots(figsize=(5, 10))

people = np.array(list(label2color.keys()))[sorting]
colors = np.array(list(color2label.keys()))[sorting]/255


people = people[filter_empty_nan]
colors = colors[filter_empty_nan]

sorting_filter_nans = (sorting * filter_empty_nan)[sorting * filter_empty_nan > 0]
y_pos = np.arange(len(people))



ax13.barh(y_pos - width, 100*np.max(histories[models[3]], axis=0)[sorting_filter_nans], width, label=models[3],zorder=10)
ax13.barh(y_pos, 100*np.max(histories[models[5]],axis=0)[sorting_filter_nans], width, label=models[5],zorder=10)
ax13.barh(y_pos + width, 100*np.max(histories[models[6]], axis=0)[sorting_filter_nans], width,label=models[6],zorder=10)


plt.yticks(y_pos,people)
ax13.set_title("Model comparison on IoU")
ax13.set_xlabel("IoU %")
ax13.legend(loc='best',fontsize=7)


#Beautify
ax13.spines["top"].set_visible(False)
ax13.spines["bottom"].set_visible(False)
ax13.spines["right"].set_visible(False)
ax13.spines["left"].set_visible(False)
ax13.get_xaxis().tick_bottom()
ax13.get_yaxis().tick_left()



r1,r2 = ax13.get_ylim()
r = (int(r1-0.5), int(r2+0.5))
for y in range(1, 90, 10):
    ax13.plot([y] * len(range(*r)),range(*r), "--", lw=0.5, color="black", alpha=0.3,zorder=1)

ax13.invert_yaxis()  # labels read top-to-bottom

handles, labels = ax13.get_legend_handles_labels()
# ax.legend(reversed(handles), reversed(labels), title='Models', loc='best')



plt.savefig(os.path.join("files","combined3.pdf"), bbox_inches="tight")
plt.show()
plt.close()



# IoU
histories = {}
models = []
for iter, item in enumerate(fold_list):
    print(str(iter)+'. ' + item)

    file_path = os.path.join(scores_dir, item, 'train_loss.npy')

    which_model = item.partition("-")[0]
    if item[-4:] == 'True': which_model = which_model + 'VGG_trained'
    models.append(which_model)
    hist = np.load(file_path)
    hist = hist[:np.max(np.nonzero(hist))]
    histories[which_model] = hist





ax = plt.subplot(1,1,1)
p = []
for iter,mdl in enumerate(models):
    if iter>0 and histories[mdl][-1]<0.8:
        ax.plot(smooth(np.tanh(histories[mdl])), label=str(mdl),linewidth=4)


# ax.plot(filter(inp=histories[models[3]]), label=str(3),linewidth=4)
ax.set_title("Loss",fontsize=25)
ax.set_xlabel("epoch",fontsize=20)
ax.set_ylabel("tanh(loss)",fontsize=20)


r1,r2 = ax.get_xlim()
r = (int(r1), int(r2))

y1,y2 = ax.get_ylim()
yy = (int(y1*100),int(y2*100))
for y in range(yy[0], yy[1], 10):
    ax.plot(range(*r), [y/100] * len(range(*r)), "--", lw=0.5, color="black", alpha=0.3)


ax.legend(loc='best')

#Beautify
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.savefig(os.path.join("files","loss.pdf"), bbox_inches="tight")
plt.show()
plt.close()

