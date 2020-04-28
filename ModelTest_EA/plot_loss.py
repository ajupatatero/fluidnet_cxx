import numpy as np
import torch
import torch.utils.data
import os
import matplotlib.pyplot as plt
import random

from matplotlib.pyplot import *

Val_loss_0= np.load('/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/ModelTest_EA/val_loss.npy')
Train_loss_0= np.load('/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/ModelTest_EA/train_loss.npy')


print(Val_loss_0.shape)


#Variables
epoch = Val_loss_0[:,0]

Train_loss = Train_loss_0[:,1]
Div_train_loss = Train_loss_0[:,3]
Div_LT_train_loss = Train_loss_0[:,6]

Val_loss = Val_loss_0[:,1]
Div_val_loss = Val_loss_0[:,3]
Div_LT_val_loss = Val_loss_0[:,6]

#plot training/validation losses on relevant graph
plt.figure(figsize=(20,10))

plt.plot(epoch,Train_loss, label='Training loss REF', color='blue', linestyle='solid', linewidth=5)
plt.plot(epoch,Val_loss, label='Validation loss REF', color='blue', marker='+', linestyle='None', linewidth=5)

plt.plot(epoch,Div_train_loss, label='Divergency Training loss', color='red', linestyle='solid', linewidth=5)
plt.plot(epoch,Div_val_loss, label='Divergency Validation loss', color='red', marker='+', linestyle='None', linewidth=5)

plt.plot(epoch,Div_LT_train_loss, label='Training Long Term loss', color='green', linestyle='solid', linewidth=5)
plt.plot(epoch,Div_LT_val_loss, label='Validation Long Term loss', color='green', marker='+', linestyle='None', linewidth=5)

#print('Val Loss', Val_loss)

plt.legend(fontsize = 20)
plt.yscale("log")
#plt.ylim(0.01,0.5)

plt.title('Training & Validation loss during training', fontsize = 25)
plt.xlabel('Epoch', fontsize = 25)
plt.ylabel('Loss', fontsize = 25)

plt.savefig('/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/ModelTest_EA/Loss_total.png')
