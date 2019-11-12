import glob
import argparse
import yaml

import torch
import torch.autograd
import time

import matplotlib
if 'DISPLAY' not in glob.os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

import numpy as np
import numpy.ma as ma

import pyevtk.hl as vtk

from shutil import copyfile
import importlib.util
from timeit import default_timer

folder = '/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/pytorch/Swirl'

radius_factor = 0.05

resX = 128
resY = 128

cenX = 63
cenY = 63

#x_tensor = (torch.arange(resX).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view((1,1,1,resX,resY))
x_tensor = torch.arange(resX, dtype=torch.float).view((1,resX)).expand((1, 1, 1,resY, resX))
y_tensor = torch.arange(resY, dtype=torch.float).view((1,resY,1)).expand((1, 1, 1,resY, resX))

center_x = torch.ones_like(x_tensor, dtype=torch.float)*cenX
center_y = torch.ones_like(x_tensor, dtype=torch.float)*cenY

dis_x = torch.zeros_like(x_tensor)
dis_y = torch.zeros_like(y_tensor)

dis_x = (- center_x + x_tensor)*radius_factor
dis_y = (- center_y + y_tensor)*radius_factor

print("dis_x ", dis_x)
print("dis_y ", dis_y)

radius = torch.sqrt(((dis_x*dis_x)+(dis_y*dis_y)))

print("Radius ", radius)

teta_cos = torch.acos((dis_x)/radius)
teta_sin = torch.asin((dis_y)/radius)

teta_cos[0,0,0,cenY, cenX]=0
teta_sin[0,0,0,cenY, cenX]=0

print("teta cos ", teta_cos *180/np.pi)
print("teta sin ", teta_sin *180/np.pi)

Lamb = np.pi
delta = 1

u_teta = (Lamb / (2*np.pi))*(1/radius)*(1-torch.exp(-(radius*radius)/(delta*delta)))
u_teta[0,0,0,cenY, cenX]=0

a_r = -(u_teta*u_teta)/radius
a_r[0,0,0,cenY, cenX]=0

rho_c_b = 3
e_ct = 0.3
m =2
epsilon = 1
delta_p = delta/epsilon

r_p = radius*(1+e_ct*torch.cos(m*teta_cos))
rho_1 = 1 + (rho_c_b - 1)*torch.exp(-(r_p*r_p)/(delta_p*delta_p))

epsilon = 4
delta_p = delta/epsilon
rho_4 = 1 + (rho_c_b - 1)*torch.exp(-(r_p*r_p)/(delta_p*delta_p))

x_delta = np.zeros(resX-cenX)
u_teta_line = np.zeros(resX-cenX)
a_r_line = np.zeros(resX-cenX)
rho_1_line = np.zeros(resX-cenX)
rho_4_line = np.zeros(resX-cenX)

u_x = torch.zeros_like(dis_x)
u_y = torch.zeros_like(dis_y)


u_x = - u_teta*torch.sin(teta_sin)
u_y = u_teta*torch.cos(teta_cos)

U = torch.zeros((1,2,1,resY,resX))

U[0,0]= u_x[0,0]
U[0,1]= u_y[0,0]


fig,axes = plt.subplots(figsize=(45,15),nrows = 1, ncols =3 )
fig.subplots_adjust(hspace=0.4, wspace=0.4)

fig.suptitle(' U and Rho field ', fontsize = 45)

tt=axes[0].imshow(U[0,0,0],origin='lower')
axes[0].set_title('U x', fontsize = 30)
axes[0].axis('off')
fig.colorbar(tt, ax =axes[0])

ll=axes[1].imshow(U[0,1,0],origin='lower')
axes[1].set_title('U y', fontsize = 30)
axes[1].axis('off')
fig.colorbar(ll, ax =axes[1])

lll=axes[2].imshow(rho_1[0,0,0],origin='lower')
axes[2].set_title('Rho 1', fontsize = 30)
axes[2].axis('off') 
fig.colorbar(lll, ax =axes[2])

savefile = folder + '/Initial_U_Rho_field.png'
plt.savefig(savefile)
plt.close()


for i in range(cenX, resX):

    x_delta[i-cenX] = np.sqrt(2*(((i-cenX)*radius_factor)*((i-cenX)*radius_factor)))
    u_teta_line[i-cenX]=  u_teta[0,0,0,i,i]
    a_r_line[i-cenX]=a_r[0,0,0,i,i]
    rho_1_line[i-cenX]= rho_1[0,0,0,i,i]-1
    rho_4_line[i-cenX]= rho_4[0,0,0,i,i]-1


plt.figure()

plt.plot(x_delta,np.abs(10*a_r_line), label='Abs 10 * Ac_r',color='black', linestyle='solid', linewidth=2)
plt.plot(x_delta,u_teta_line, label='U theta', color='black', linestyle='dashdot', linewidth=2)
plt.plot(x_delta,rho_1_line, label='EPS 1 Rho/Rho b - 1', color='black', linestyle='dashed', linewidth=2)
plt.plot(x_delta,rho_4_line, label='EPS 4 Rho/Rho b - 1', color='black', linestyle='dotted', linewidth=2)
#plt.ylim(0.01,0.5)

plt.title('Parameter Evolution')
plt.xlabel('r / delta')
plt.ylabel('Various')

plt.legend()

plt.savefig(folder+'/Var_Evolution.png')

'''
dt = arguments.setdt or simConf['dt']

p =       torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
U =       torch.zeros((1,2,1,resY,resX), dtype=torch.float).cuda()
flags =   torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
density = torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
fluid.emptyDomain(flags)

Outside_Ja = simConf['outside_Ja']
Threshold_Div = arguments.setThreshold or simConf['threshold_Div']

batch_dict = {}
batch_dict['p'] = p
batch_dict['U'] = U
batch_dict['flags'] = flags
batch_dict['density'] = density
'''
