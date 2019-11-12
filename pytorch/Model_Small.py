import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import fluid, MultiScaleNet
from math import inf

class FluidNet(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, mconf, dropout=False):
        super(FluidNet, self).__init__()


        self.dropout = dropout
        self.mconf = mconf
        self.inDims = mconf['inputDim']
        self.is3D = mconf['is3D']

        # MultiScaleNet
        self.multiScale = MultiScaleNet(self.inDims)

    def forward(self, input_):

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

        x = torch.FloatTensor(input_.size(0), \
                              2,    \
                              input_.size(2), \
                              input_.size(3), \
                              input_.size(4)).type_as(input_)

        chan = 0
        x[:, chan] = div[:,0]
        chan += 1

        # FlagsToOccupancy creates a [0,1] grid out of the manta flags
        x[:,chan,:,:,:] = fluid.flagsToOccupancy(flags).squeeze(1)

        if not self.is3D:
            # Squeeze unary dimension as we are in 2D
            x = torch.squeeze(x,2)

        if self.mconf['model'] == 'ScaleNet':
            p = self.multiScale(x)


        return p, UDiv


