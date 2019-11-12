import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timeit import default_timer

from lib import fluid, MultiScaleNet
from math import inf

class _ScaleNet(nn.Module):
    def __init__(self, mconf):
        super(_ScaleNet, self).__init__()
        self.mconf = mconf

    def forward(self, x):
        bsz = x.size(0)
        # Rehaspe form (b x chan x d x h x w) to (b x -1)
        y = x.view(bsz, -1)
        # Calculate std using Bessel's correction (correction with n/n-1)
        std = torch.std(y, dim=1, keepdim=True) # output is size (b x 1)
        scale = torch.clamp(std, \
            self.mconf['normalizeInputThreshold'] , inf)
        scale = scale.view(bsz, 1, 1, 1, 1)

        return scale

class _HiddenConvBlock(nn.Module):
    def __init__(self, dropout=True):
        super(_HiddenConvBlock, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, padding = 0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, padding = 0),
            nn.ReLU(),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class FluidNetTGV(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, mconf, it, dropout=False):
        super(FluidNetTGV, self).__init__()


        self.dropout = dropout
        self.mconf = mconf
        self.inDims = mconf['inputDim']
        self.is3D = mconf['is3D']
        self.it = it

        self.scale = _ScaleNet(self.mconf)
        # Input channels = 3 (inDims, flags)
        # We add padding to make sure that Win = Wout and Hin = Hout with ker_size=3
        self.conv1 = torch.nn.Conv2d(self.inDims, 16, kernel_size=3, padding=1)

        self.modDown1 = torch.nn.AvgPool2d(kernel_size=2)
        self.modDown2 = torch.nn.AvgPool2d(kernel_size=4)

        self.convBank = _HiddenConvBlock(dropout=False)

        #self.deconv1 = torch.nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        #self.deconv2 = torch.nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4)

        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(16, 8, kernel_size=1)

        # Output channels = 1 (pressure)
        self.convOut = torch.nn.Conv2d(8, 1, kernel_size=1)

        # MultiScaleNet
        self.multiScale = MultiScaleNet(self.inDims)

    def forward(self, input_, it):

        # data indexes     |           |
        #       (dim 1)    |    2D     |    3D
        # ----------------------------------------
        #   DATA:
        #       pDiv       |    0      |    0
        #       UDiv       |    1:3    |    1:4
        #       flags      |    3      |    4
        #       densityDiv |    4      |    5
        #   TARGET:
        #       p          |    0      |    0
        #       U          |    1:3    |    1:4
        #       density    |    3      |    4

        # For now, we work ONLY in 2d
        assert self.is3D == False, 'Input can only be 2D'

        assert self.mconf['inputChannels']['pDiv'] or \
                self.mconf['inputChannels']['UDiv'] or \
                self.mconf['inputChannels']['div'], 'Choose at least one field (U, div or p).'

        pDiv = None
        UDiv = None
        div = None

        flags = input_[:,1].unsqueeze(1).contiguous()
        div = input_[:,0].unsqueeze(1).contiguous()
        x = torch.FloatTensor(input_.size(0), \
                              2,    \
                              input_.size(2), \
                              input_.size(3), \
                              input_.size(4)).type_as(input_)

        chan = 0
        x[:, chan] = div[:,0]
        chan += 1


        #Print Before U
        #Uinter1_cpu = UDiv.cpu()
        #filename_inter1 = folder + '/U1_NN_Intermediate1_{0:05}'.format(it)
        #np.save(filename_inter1,Uinter1_cpu)

        # FlagsToOccupancy creates a [0,1] grid out of the manta flags
        x[:,chan,:,:,:] = fluid.flagsToOccupancy(flags).squeeze(1)


        if not self.is3D:
            # Squeeze unary dimension as we are in 2D
            x = torch.squeeze(x,2)

        if self.mconf['model'] == 'ScaleNet':

            start = default_timer()

            p = self.multiScale(x)

            end = default_timer()
            time=(end - start)


        else:
            # Inital layers
            x = F.relu(self.conv1(x))

            # We divide the network in 3 banks, applying average pooling
            x1 = self.modDown1(x)
            x2 = self.modDown2(x)

            # Process every bank in parallel
            x0 = self.convBank(x)
            x1 = self.convBank(x1)
            x2 = self.convBank(x2)

            # Upsample banks 1 and 2 to bank 0 size and accumulate inputs

            #x1 = self.upscale1(x1)
            #x2 = self.upscale2(x2)

            x1 = F.interpolate(x1, scale_factor=2)
            x2 = F.interpolate(x2, scale_factor=4)
            #x1 = self.deconv1(x1)
            #x2 = self.deconv2(x2)

            #x = torch.cat((x0, x1, x2), dim=1)
            x = x0 + x1 + x2

            # Apply last 2 convolutions
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            # Output pressure (1 chan)
            p = self.convOut(x)


        # Add back the unary dimension
        if not self.is3D:
            p = torch.unsqueeze(p, 2)
    
        #Print Pressures
        #P_cpu = p.cpu()
        #filename = folder + '/P_NN_output_{0:05}'.format(it)
        #np.save(filename,P_cpu)


        #Print Before U
        #Uinxb = UDiv.clone()
        #Ubef_cpu = Uinxb.cpu()
        #filename_1 = folder + '/U1_NN_Bef_update{0:05}'.format(it)
        #np.save(filename_1,Ubef_cpu)

        # Correct U = UDiv - grad(p)
        # flags is the one with Manta's values, not occupancy in [0,1]
        #fluid.velocityUpdate(pressure=p, U=UDiv, flags=flags)
       
        # We now UNDO the scale factor we applied on the input.
        #if self.mconf['normalizeInput']:
        #    p = torch.mul(p,s)  # Applies p' = *= scale
        #    UDiv = torch.mul(UDiv,s)

        # Set BCs after velocity update.
        #UDiv = fluid.setWallBcs(UDiv, flags)

        #Print After U normalization
     
        #Ua_cpu = UDiv.cpu()
        #filename_2 = folder + '/U1_NN_After_update{0:05}'.format(it)
        #np.save(filename_2,Ua_cpu)
     
        return p, div, time


