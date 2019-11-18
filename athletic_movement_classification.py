# -*- coding: utf-8 -*-
"""
Sensor data required for automatic recognition of athletic tasks using deep neural networks.
Authors: AL Clouthier, GB Ross, RB Graham

movement_screen_data.pickle contains the simulated IMU data all 13 body segments 
for three athletes performing 13 athletic tasks. 
This code automatically classifies the movement being performed in each frame 
of data using DNNs previously trained on subsets of the sIMU data.

DNN is based on: 
	Ordóñez, F. J., and Roggen, D. (2016). Deep convolutional and LSTM recurrent neural networks 
	for multimodal wearable activity recognition. Sensors (Switzerland) 16, 115. doi:10.3390/s16010115.

Tested using:
    Python 3.7.3
    pytorch 1.1.0
    numpy 1.16.2
    scikit-learn 0.20.3
    scipy 1.2.1 
    matplotlib 3.0.3
"""

# ------------------------------- #
# --- Data and DNN selection  --- #
# ------------------------------- #

# Select athlete (0, 1, or 2) to classify movements for.
ath = 0 
# Select DNN to use (sIMU1, sIMU2, sIMU3L, sIMU3U, sIMU4, sIMU4D, sIMU4P, sIMU5, or sIMU13)
dnn = 'sIMU3L'
# ------------------------------ #

import torch
import torch.utils.data
import torch.nn as nn
import pickle
import math
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix

# Load sIMU data and frame labels
with open('.\\movement_screen_data.pickle','rb') as f:
    data = pickle.load(f)   

# Data columns to use for input to DNNs
segmentCols = {'sIMU1' : [3,4,5,40,53], # torso
               'sIMU2' : [3,4,5,6,7,8,40,41,53,54],  # torso, pelvis
               'sIMU3L' : [3,4,5,27,28,29,30,31,32,40,48,49,53,61,62], # torso, shanks
               'sIMU3U' : [3,4,5,9,10,11,12,13,14,40,42,43,53,55,56], # torso, upper arms
               'sIMU4' : [3,4,5,6,7,8,21,22,23,24,25,26,40,41,46,47,53,54,59,60], # torso, pelvis, thighs
               'sIMU4D' : [15,16,17,18,19,20,27,28,29,30,31,32,44,45,48,49,57,58,61,62], # forearms, shanks
               'sIMU4P' : [9,10,11,12,13,14,21,22,23,24,25,26,42,43,46,47,55,56,59,60], # upper arms, thighs
               'sIMU5' : [3,4,5,15,16,17,18,19,20,27,28,29,30,31,32,40,44,45,48,49,53,57,58,61,62], # torso, forearms, shanks
               'sIMU13' : range(65)} # head, torso, pelvis, upper arms, forearms, thighs, shanks, feet
n_col = len(segmentCols[dnn]) 

tasks = ['DJ','HDL','HDR','LHL','LHR','LL','LR','BDL','BDR','SDL','SDR','TBL','TBR']
taskID = [5,   11,   0,    7,    4,    8,  10,   2,     1,   3,     6,    12,  9] # class number assigned to each task
num_classes = len(tasks) + 1

# Parameters for the sliding window used to segment the data
windsize = 48
stride = 12

# ------------------------------- #
# --- Define DNN Architecture --- #
# ------------------------------- #

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define DNN layers. Based on DeepConvLSTM from Ordonez et al., 2016, Sensors 16(1).
class Net(nn.Module):
    def __init__(self,num_classes=num_classes):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,32,kernel_size=[6,1],stride=1,padding=0),
                nn.Conv2d(32,32,kernel_size=[6,1]),
                nn.Conv2d(32,32,kernel_size=[6,1]),
                nn.Conv2d(32,32,kernel_size=[6,1]),
				nn.ReLU()) 
        self.lstm1 = nn.LSTM(32*n_col,64)
        self.lstm2 = nn.LSTM(64,64)
        self.fc = nn.Linear(64*28,num_classes)
        self.sm = nn.Softmax(1)
 
    def forward(self,x):
        out = self.layer1(x)
        out = out.view(out.size(0),out.size(2),-1)
        out,_ = self.lstm1(out)
        out,_ = self.lstm2(out)
        out = self.fc(out.view(out.size(0),-1))
        return out
    
net = Net(num_classes).to(device)

# Load in checkpoint for previously trained DNN
net.load_state_dict(torch.load('.\\dnn_checkpoints\\' + dnn + '.ckpt',map_location=device)) 
print('Loaded trained network weights for ' + dnn)    

# --------------------------- #
# --- Compile data matrix --- #
# --------------------------- #

# Get final tensor size
m = 0
for t in tasks:
    if t in data['data'][ath]['linacc']:
        m += len(data['data'][ath]['linacc'][t])

# pad with zeros so that all data gets windowed
n = math.ceil(m/windsize) * windsize
compdata = torch.zeros(n,data['data'][ath]['angpos'][t].shape[1] + 
                       data['data'][ath]['linacc'][t].shape[1] + data['data'][ath]['angvel'][t].shape[1])
y = [-1] * n

# Compile data from all tasks into one tensor
k = 0
for t in tasks:
    if t in data['data'][ath]['linacc']:
        nFrames = data['data'][ath]['linacc'][t].shape[0]
        compdata[k:k+nFrames,:] = torch.cat((data['data'][ath]['angpos'][t], 
                       data['data'][ath]['linacc'][t],data['data'][ath]['angvel'][t]),1)
        
        lbl = [-1] * nFrames
        if data['data'][ath]['startstop'][t][1] > nFrames:
                data['data'][ath]['startstop'][t][1] = nFrames
        lbl[data['data'][ath]['startstop'][t][0]:data['data'][ath]['startstop'][t][1]] = \
            [taskID[tasks.index(t)]] * (data['data'][ath]['startstop'][t][1] - data['data'][ath]['startstop'][t][0])
        y[k:k+nFrames] = lbl
        
        k += nFrames
y = [i+1 for i in y]

# Normalize data using mean and standard devidation from the training set
compdata[0:m,:] = (compdata[0:m,:] - data['train_mean'])/data['train_std']

# Remove any rows with NaNs
y = np.delete(y,np.argwhere(np.isnan(compdata).any(1)))
compdata = compdata[~np.isnan(compdata).any(1)]

# Window data using sliding windows 
num_wind = int((compdata.shape[0] - windsize)/stride) +  1
X = torch.empty(num_wind,windsize,compdata.shape[1])
Y = torch.zeros(num_wind,dtype=torch.long)
i = 0
j = 0
while j + windsize < compdata.shape[0]:
    X[i,:,:] = compdata[j:j+windsize,:]
    Y[i] = int(stats.mode(y[j:j+windsize])[0][0]) 
    i += 1
    j += stride     
X = X[:,None,:,segmentCols[dnn]] # Only include data columns corresponding to body segments for this DNN

# ------------------------------------------- #
# --- Predict Movements using trained DNN --- #
# ------------------------------------------- #

# Predict movement performed in each window
net.eval() 
with torch.no_grad():
    inputdata = X.to(device)
    outputs = net(inputdata)
    _,Y_pred = torch.max(outputs.data,1)

# Get movement prediction for each frame of data
y_pred = [0] * len(y)
y_prob = torch.zeros((len(y),num_classes))
I = [0] # indices of windows that contain this frame of data
k = 0
for i in range(compdata.shape[0]):
    wind_scores = torch.zeros(len(I),num_classes,dtype=torch.float32)
    jj = 0
    for j in I:
        wind_scores[jj,:] = outputs[j,:]
        jj=jj+1
    # take the mean of the probabilities for the included windows then make prediction
    maxscore,idx = torch.max(torch.mean(wind_scores,0),0)
    y_pred[i] = idx.item()
    y_prob[i,:] = torch.mean(wind_scores,0)

    # update window indices for next data frame
    if (i >= windsize) and (i % stride == 0):
        del I[0]
    if i % stride == 0:
        if (k < Y_pred.shape[0]):
            I.append(k)
        k = k+1

# ------------------------------------------ #
# --- Classification performance metrics --- #
# ------------------------------------------ #
        
tasklist = tasks.copy()
tasklist.insert(0,'Null')
        
# Calculate precision, recall, and F1 scores
precision = dict()
precision['full'] = precision_score(y,y_pred,average=None)
precision['micro'] = precision_score(y,y_pred,average='micro')
precision['macro'] = precision_score(y,y_pred,average='macro')
recall = dict()
recall['full'] = recall_score(y,y_pred,average=None)
recall['micro'] = recall_score(y,y_pred,average='micro')
recall['macro'] = recall_score(y,y_pred,average='macro')
accuracy = accuracy_score(y,y_pred)
fmeasure = dict()
fmeasure['full'] = f1_score(y,y_pred,average=None)
fmeasure['micro'] = f1_score(y,y_pred,average='micro')
fmeasure['macro'] = f1_score(y,y_pred,average='macro')

print('\nAthlete %d' % ath)
print('Micro-avg precision = %.3f' % precision['micro'])
print('Micro-avg recall = %.3f' % recall['micro'])
print('Micro-avg F1 Score = %.3f' % fmeasure['micro'])

# Confusion matrix - rows=truth, cols=prediction
confmat = confusion_matrix(y,y_pred)
# Reorder classes for nicer presentation
tid = [x+1 for x in taskID]
tid.insert(0,0)
confmat = confmat[tid,:]
confmat = confmat[:,tid]

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(confmat.shape[1]),
       yticks=np.arange(confmat.shape[0]),
       xticklabels=tasklist, yticklabels=tasklist,
       title='F1 Score = %.3f' % fmeasure['micro'],
       ylabel='True Movement',
       xlabel='Predicted Movement')
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
thresh = confmat.max() / 2.
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(j, i, format(confmat[i, j], 'd'),
                ha="center", va="center",
                color="white" if confmat[i, j] > thresh else "black")
fig.tight_layout()
plt.show()


# Plot true and predicted movements for all data frames
# reorder labels to match class ID
classes = [-1] * num_classes
for i in range(len(tid)):
    classes[tid[i]] = tasklist[i]

fig,ax = plt.subplots()
plt.plot(y)
plt.plot(y_pred)
plt.grid(True,'both','y')
plt.xlabel('Frame')
plt.ylabel('Movement')
plt.legend(('True','Predicted'))
plt.title('Athlete %d' % ath)
ax.set(yticks=np.arange(num_classes), yticklabels=classes)
plt.show()
