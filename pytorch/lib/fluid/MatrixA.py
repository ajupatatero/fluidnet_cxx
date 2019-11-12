import torch
from . import CellType

def createMatrixA(flags):

    #cuda = torch.device('cuda')
    assert (flags.dim() == 5), 'Dimension mismatch'
    assert flags.size(1) == 1, 'flags is not a scalar'

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (flags.size(1) == 3)
    if (not is3D):
        assert d == 1, '2D velocity field but zdepth > 1'

    assert (flags.is_contiguous() ), 'Input is not contiguous'

    # We declare the super tensor A. The declaration order is very importan
    # as we will have to flatten the divergency consecuently!

    #This module might be one of the most inefficient codes in the entire FluidNet
    # Just have to run it once
    A = torch.zeros(h*w,h*w)

    i = torch.arange(start=0, end=w, dtype=torch.long).cuda() \
            .view(1,w).expand(bsz, d, h, w)
    j = torch.arange(start=0, end=h, dtype=torch.long).cuda() \
            .view(1,h,1).expand(bsz, d, h, w)
    k = torch.zeros_like(i)
    if (is3D):
        k = torch.arange(start=0, end=d, dtype=torch.long).cuda() \
                .view(1,d,1,1).expand(bsz, d, h, w)

    zero = torch.zeros_like(i)
    maxX = torch.zeros_like(i).fill_(w-1)
    maxY = torch.zeros_like(i).fill_(h-1)
    zeroF = zero.float()
    zeroBy = torch.zeros(i.size(), dtype=torch.uint8).cuda()

    idx_b = torch.arange(start=0, end=bsz, dtype=torch.long).cuda() \
                .view(bsz, 1, 1, 1).expand(bsz,d,h,w)

    mCont = torch.ones_like(zeroBy)

    cur_fluid = flags.eq(CellType.TypeFluid).squeeze(1)
    cur_obs = flags.eq(CellType.TypeObstacle).squeeze(1)
    mNotCells = cur_fluid.ne(1).__and__\
                (cur_obs.ne(1))
    mCont.masked_fill_(mNotCells, 0)


    counter =0
    for j in range(h):
        for i in range(w):

           if flags[0,0,0,j,i].eq(CellType.TypeFluid):

                A[counter,counter]=4

                if flags[0,0,0,j,i-1].eq(CellType.TypeObstacle):
                    A[counter,counter]-=1
                if flags[0,0,0,j,i+1].eq(CellType.TypeObstacle):
                    A[counter,counter]-=1
                if flags[0,0,0,j-1,i].eq(CellType.TypeObstacle):
                    A[counter,counter]-=1
                if flags[0,0,0,j+1,i].eq(CellType.TypeObstacle):
                    A[counter,counter]-=1

                if flags[0,0,0,j,i-1].eq(CellType.TypeFluid):
                    A[counter,counter-1]-=1
                if flags[0,0,0,j,i+1].eq(CellType.TypeFluid):
                    A[counter,counter+1]-=1
                if flags[0,0,0,j-1,i].eq(CellType.TypeFluid):
                    A[counter,counter-w]-=1
                if flags[0,0,0,j+1,i].eq(CellType.TypeFluid):
                    A[counter,counter+w]-=1

           counter+=1

    b = torch.rand(h,w)
    print("b ", b)
    c = torch.flatten(b)
    print("c ", c)
    return A
