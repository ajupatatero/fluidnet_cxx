import torch
import math
import numpy as np

def createVKBCs(batch_dict, density_val, u_scale, rad):
    r"""Creates masks to enforce an inlet at the domain bottom wall.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        density_val (float): Inlet density.
        u_scale (float); Inlet velocity.
        rad (float): radius of inlet circle (centered around midpoint of wall)
    """

    #Jet length (jl -a) 
    jl = 4
    #Jet first cell point
    a=1

    flags = batch_dict['flags']

    cuda = torch.device('cuda')
    # batch_dict at input: {p, UDiv, flags, density, Ustar, Div_input,VK, simutype}
    #assert len(batch_dict) == 8, "Batch must contain 8 tensors (p, UDiv, flags, density, flags_inflow, Ustar, Div input, VK, simutype)"
    UDiv = batch_dict['U']
    density = batch_dict['density']
    UBC = UDiv.clone().fill_(0)
    UBCInvMask = UDiv.clone().fill_(1)

    # Single density value
    densityBC = density.clone().fill_(0)
    densityBCInvMask = density.clone().fill_(1)

    assert UBC.dim() == 5, 'UBC must have 5 dimensions'
    assert UBC.size(0) == 1, 'Only single batches allowed (inference)'

    xdim = UBC.size(4)
    ydim = UBC.size(3)
    zdim = UBC.size(2)
    is3D = (UBC.size(1) == 3)

    if not is3D:
        assert zdim == 1, 'For 2D, zdim must be 1'
    centerX = xdim // 2
    centerZ = max( zdim // 2, 1.0)
    #Remember that floor (5.6 = 5, -7.1 = -7)
    plumeRad = math.floor(xdim*rad)

    y = 1
    if (not is3D):
        #vec = (0,1)
        vec = torch.arange(0,2, device=cuda).float()
    else:
        vec = torch.arange(0,3, device=cuda).float()
        vec[2] = 0

    # vec = vec * u_scale (vinj)
    vec.mul_(u_scale)
    print("V INJ = ", vec[1])
    print("Scale", u_scale)

    # Equal to = vector size H, then reshaped to a matrix of size (H,1) and expanded
    index_x = torch.arange(0, xdim, device=cuda).view(xdim).expand_as(density[0][0])
    index_y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand_as(density[0][0])
    if (is3D):
        index_z = torch.arange(0, zdim, device=cuda).view(zdim, 1, 1).expand_as(density[0][0])

    if (not is3D):
        index_ten = torch.stack((index_x, index_y), dim=0)
    else:
        index_ten = torch.stack((index_x, index_y, index_z), dim=0)

    #TODO 3d implementation
    indx_circle = index_ten[:,:,a:jl]
    maskInside = (indx_circle[1] <= jl)

    # Inside the plume. Set the BCs.

    #It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    #DEBUG
    UBC[:,:,:,a:jl] = maskInside_f * vec.view(1,2,1,1,1).expand_as(UBC[:,:,:,a:jl]).float()
    #UBC[:,:,:,0:jl].masked_fill_(maskInside, u_scale)
    UBCInvMask[:,:,:,a:jl].masked_fill_(maskInside, 0)

    densityBC[:,:,:,a:jl].masked_fill_(maskInside, density_val)
    densityBCInvMask[:,:,:,a:jl].masked_fill_(maskInside, 0)

    # Outside the plume. Set the velocity to zero and leave density alone.

    maskOutside = (maskInside == 0)
    UBC[:,:,:,a:jl].masked_fill_(maskOutside, 0)
    UBCInvMask[:,:,:,a:jl].masked_fill_(maskOutside, 0)

    #Outflow


    indx_circle = index_ten[:,:,-jl:]
    maskInside = (indx_circle[1] >= ydim-jl)
    
    # Inside the plume. Set the BCs.

    #It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    #DEBUG
    UBC[:,:,:,-jl:] = maskInside_f * vec.view(1,2,1,1,1).expand_as(UBC[:,:,:,-jl:]).float()
    #UBC[:,:,:,0:jl].masked_fill_(maskInside, u_scale)
    UBCInvMask[:,:,:,-jl:].masked_fill_(maskInside, 0)
    densityBCInvMask[:,:,:,-jl:].masked_fill_(maskInside, 0)
    
    # Outside the plume. Set the velocity to zero and leave density alone.
    
    maskOutside = (maskInside == 0)
    UBC[:,:,:,-jl:].masked_fill_(maskOutside, 0)
    UBCInvMask[:,:,:,-jl:].masked_fill_(maskOutside, 0)

    # Insert the new tensors in the batch_dict.
    batch_dict['UBC'] = UBC
    batch_dict['UBCInvMask'] = UBCInvMask
    batch_dict['densityBC'] = densityBC
    batch_dict['densityBCInvMask'] = densityBCInvMask

def createStepBCs(batch_dict, density_val, u_scale, rad, resX, Long_S_X):
    r"""Creates masks to enforce an inlet at the domain bottom wall.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        density_val (float): Inlet density.
        u_scale (float); Inlet velocity.
        rad (float): radius of inlet circle (centered around midpoint of wall)
    """

    #Jet length (jl -a) 
    jl = 4
    #Jet first cell point
    a=1

    flags = batch_dict['flags']

    cuda = torch.device('cuda')
    # batch_dict at input: {p, UDiv, flags, density, Ustar, Div_input}
    #assert len(batch_dict) == 8, "Batch must contain 8 tensors (p, UDiv, flags, density, flags_inflow, Ustar, Div input)"
    UDiv = batch_dict['U']
    density = batch_dict['density']
    UBC = UDiv.clone().fill_(0)
    UBCInvMask = UDiv.clone().fill_(1)

    # Single density value
    densityBC = density.clone().fill_(0)
    densityBCInvMask = density.clone().fill_(1)

    assert UBC.dim() == 5, 'UBC must have 5 dimensions'
    assert UBC.size(0) == 1, 'Only single batches allowed (inference)'

    xdim = UBC.size(4)
    ydim = UBC.size(3)
    zdim = UBC.size(2)
    is3D = (UBC.size(1) == 3)

    if not is3D:
        assert zdim == 1, 'For 2D, zdim must be 1'
    centerX = xdim // 2
    centerZ = max( zdim // 2, 1.0)
    #Remember that floor (5.6 = 5, -7.1 = -7)
    plumeRad = math.floor(xdim*rad)

    y = 1
    if (not is3D):
        #vec = (0,1)
        vec = torch.arange(0,2, device=cuda).float()
    else:
        vec = torch.arange(0,3, device=cuda).float()
        vec[2] = 0

    # Equal to = vector size H, then reshaped to a matrix of size (H,1) and expanded
    index_x = torch.arange(0, xdim, device=cuda).view(xdim).expand_as(density[0][0])
    index_y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand_as(density[0][0])
    if (is3D):
        index_z = torch.arange(0, zdim, device=cuda).view(zdim, 1, 1).expand_as(density[0][0])

    if (not is3D):
        index_ten = torch.stack((index_x, index_y), dim=0)
    else:
        index_ten = torch.stack((index_x, index_y, index_z), dim=0)

    h_s = np.int(xdim/2.0)

    input_U = -6.0*u_scale*(index_x)*(index_x-h_s)/(h_s*h_s)
    input_U[:,:,h_s:-1]=0.0
    output_U = -3.0*u_scale*(index_x)*(index_x-2.0*h_s)/(4.*(h_s*h_s))

    BC_Uy = input_U
    BC_Uy[:,ydim//2:,:] = output_U[:,ydim//2:,:]
    BC_Uy[:,jl:ydim-jl-1,:] = 0.0

    BC_Ux = torch.zeros_like(BC_Uy)


    BC_U = torch.stack((BC_Ux, BC_Uy), dim=0)

    #TODO 3d implementation
    maskInside_in = (index_y[0,:,:] <= jl)
    maskInside_out = (index_y[0,:,:] >= ydim - jl)

    maskInside = maskInside_in + maskInside_out
    # Inside the plume. Set the BCs.

    #It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    #DEBUG
    #UBC[:,:,:,a:jl] = maskInside_f * vec.view(1,2,1,1,1).expand_as(UBC[:,:,:,a:jl]).float()

    UBC = BC_U.unsqueeze(0)

    #UBC[:,:,:,0:jl].masked_fill_(maskInside, u_scale)
    #UBCInvMask[:,:,:,a:jl].masked_fill_(maskInside, 0)

    densityBC[:,:,:,a:jl].masked_fill_(maskInside[a:jl], density_val)
    densityBCInvMask[:,:,:,a:jl].masked_fill_(maskInside[a:jl], 0)


    densityBC[:,:,:,-jl-1:].masked_fill_(maskInside[-jl-1:], density_val)
    densityBCInvMask[:,:,:,-jl-1:].masked_fill_(maskInside[-jl-1:], 0)

    # Outside the plume. Set the velocity to zero and leave density alone.

    maskOutside = (maskInside == 0)
    UBC.masked_fill_(maskOutside, 0)
    UBCInvMask.masked_fill_(maskOutside, 0)

    #Outflow
    indx_circle = index_ten[:,:,-jl:]
    maskInside = (indx_circle[1] >= ydim-jl)

    # Inside the plume. Set the BCs.

    #It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    #DEBUG
    #UBC[:,:,:,0:jl].masked_fill_(maskInside, u_scale)
    UBCInvMask[:,:,:,-jl:].masked_fill_(maskInside, 0)
    densityBCInvMask[:,:,:,-jl:].masked_fill_(maskInside, 0)

    # Outside the plume. Set the velocity to zero and leave density alone.

    maskOutside = (maskInside == 0)
    #UBC[:,:,:,-jl:].masked_fill_(maskOutside, 0)
    #UBCInvMask[:,:,:,-jl:].masked_fill_(maskOutside, 0)
    # Insert the new tensors in the batch_dict.
    batch_dict['UBC'] = UBC
    batch_dict['UBCInvMask'] = UBCInvMask
    batch_dict['densityBC'] = densityBC
    batch_dict['densityBCInvMask'] = densityBCInvMask







def createPlumeBCs(batch_dict, density_val, u_scale, rad):
    r"""Creates masks to enforce an inlet at the domain bottom wall.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        density_val (float): Inlet density.
        u_scale (float); Inlet velocity.
        rad (float): radius of inlet circle (centered around midpoint of wall)
    """

    #Jet length (jl -a) 
    jl = 2
    #Jet first cell point
    a=1
    
    flags = batch_dict['flags']
    
    cuda = torch.device('cuda')
    # batch_dict at input: {p, UDiv, flags, density, Ustar, Div_input}
    #assert len(batch_dict) == 8, "Batch must contain 8 tensors (p, UDiv, flags, density, flags_inflow, Ustar, Div input)"
    UDiv = batch_dict['U']
    density = batch_dict['density']
    UBC = UDiv.clone().fill_(0)
    UBCInvMask = UDiv.clone().fill_(1)

    # Single density value
    densityBC = density.clone().fill_(0)
    densityBCInvMask = density.clone().fill_(1)

    assert UBC.dim() == 5, 'UBC must have 5 dimensions'
    assert UBC.size(0) == 1, 'Only single batches allowed (inference)'

    xdim = UBC.size(4)
    ydim = UBC.size(3)
    zdim = UBC.size(2)
    is3D = (UBC.size(1) == 3)
    if not is3D:
        assert zdim == 1, 'For 2D, zdim must be 1'
    centerX = xdim // 2
    centerZ = max( zdim // 2, 1.0)
    #Remember that floor (5.6 = 5, -7.1 = -7)
    plumeRad = math.floor(xdim*rad)

    y = 1
    if (not is3D):
        #vec = (0,1)
        vec = torch.arange(0,2, device=cuda).float()
    else:
        vec = torch.arange(0,3, device=cuda).float()
        vec[2] = 0

    # vec = vec * u_scale (vinj)
    vec.mul_(u_scale)
    print("V INJ = ", vec[1]) 
    print("Scale", u_scale)

    # Equal to = vector size H, then reshaped to a matrix of size (H,1) and expanded
    index_x = torch.arange(0, xdim, device=cuda).view(xdim).expand_as(density[0][0])
    index_y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand_as(density[0][0])
    if (is3D):
        index_z = torch.arange(0, zdim, device=cuda).view(zdim, 1, 1).expand_as(density[0][0])

    if (not is3D):
        index_ten = torch.stack((index_x, index_y), dim=0)
    else:
        index_ten = torch.stack((index_x, index_y, index_z), dim=0)

    #TODO 3d implementation
    indx_circle = index_ten[:,:,a:jl]
    indx_circle[0] -= centerX
    maskInside = (indx_circle[0].pow(2) <= plumeRad*plumeRad)

    # Inside the plume. Set the BCs.

    #It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()
   
    # Tan H try:
    delt = 5
    ind_x = torch.arange(0, xdim).view(xdim).float()

    #UBC[:,1,:,a:jl] = maskInside_f * ((u_scale*0.25)*(torch.tanh(((ind_x[:])-((xdim/2)-plumeRad)-10)/10).cuda()+1)*\
    #       (1-torch.tanh(((ind_x[:])-((xdim/2)+plumeRad)+10)/10).cuda())).expand_as(UBC[:,1,:,a:jl]).float()

    #DEBUG
    UBC[:,:,:,a:jl] = maskInside_f * vec.view(1,2,1,1,1).expand_as(UBC[:,:,:,a:jl]).float()
    #UBC[:,:,:,0:jl].masked_fill_(maskInside, u_scale)
    UBCInvMask[:,:,:,a:jl].masked_fill_(maskInside, 0)

    #densityBC[:,:,:,a:jl] = maskInside_f * ((density_val*0.25)*(torch.tanh(((ind_x[:])-((xdim/2)-plumeRad)-10)/10).cuda()+1)*\
    #        (1-torch.tanh(((ind_x[:])-((xdim/2)+plumeRad)+10)/10).cuda())).expand_as(densityBC[:,:,:,a:jl]).float()
    densityBC[:,:,:,a:jl].masked_fill_(maskInside, density_val)
    densityBCInvMask[:,:,:,a:jl].masked_fill_(maskInside, 0)

    # Outside the plume. Set the velocity to zero and leave density alone.

    maskOutside = (maskInside == 0)
    UBC[:,:,:,a:jl].masked_fill_(maskOutside, 0)
    UBCInvMask[:,:,:,a:jl].masked_fill_(maskOutside, 0)

    # Insert the new tensors in the batch_dict.
    batch_dict['UBC'] = UBC
    batch_dict['UBCInvMask'] = UBCInvMask
    batch_dict['densityBC'] = densityBC
    batch_dict['densityBCInvMask'] = densityBCInvMask

    #Debug 03/06/19
    #print("UBC INV mask ", UBCInvMask)
    #print("UBC mask ", UBC)

    # batch_dict at output = {p, UDiv, flags, density, UBC,
    #                         UBCInvMask, densityBC, densityBCInvMask}

def createCilinder(batch_dict):
    """
    Creates a cilinder in the flags. It will be located in the point x = 64 and y = 80. 
    Radius = 10
    """
    flags = batch_dict['flags']
    resX = flags.size(4)
    resY = flags.size(3)

    # Here, we just impose initial conditions.
    # Upper layer rho2, vel = 0
    # Lower layer rho1, vel = 0


    centerX = 64
    centerY = 80    

    radCyl = 10

    X = torch.arange(0, resX, device=cuda).view(resX).expand((1,resY,resX))
    Y = torch.arange(0, resY, device=cuda).view(resY, 1).expand((1,resY,resX))

    dist_from_center = (X - centerX).pow(2) + (Y-centerY).pow(2)
    mask_cylinder = dist_from_center <= radCyl * radCyl

    flags = flags.masked_fill_(mask_cylinder, 2)

def createRayleighTaylorBCs(batch_dict, mconf, rho1, rho2):
    r"""Creates masks to enforce a Rayleigh-Taylor instability initial conditions.
    Top fluid has a density rho1 and lower one rho2. rho1 > rho2 to trigger instability.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        mconf (dict): configuration dict (to set thickness and amplitude of interface).
        rho1 (float): Top fluid density.
        rho2 (float): Lower fluid density.
    """

    cuda = torch.device('cuda')

    print("len batch ", len(batch_dict))
    # batch_dict at input: {p, UDiv, flags, density, UStar, divergency input}
    assert len(batch_dict) == 7, "Batch must contain 7 tensors (p, UDiv, flags, density,Ustar, divergency input)"
    UDiv = batch_dict['U']
    flags = batch_dict['flags']

    resX = UDiv.size(4)
    resY = UDiv.size(3)

    # Here, we just impose initial conditions.
    # Upper layer rho2, vel = 0
    # Lower layer rho1, vel = 0


    # New BC Distribution 
    #rex_3 = np.int(resX/3)
    #rey_3 = np.int(resY/3)

    #rex_23 = np.int(2*resX/3)
    #rey_23 = np.int(2*resY/3)

    #X1 = torch.arange(0, rex_3, device=cuda).view(rex_3).expand((1,rey_3,rex_3))
    #Y1 = torch.arange(0, rey_3, device=cuda).view(rey_3, 1).expand((1,rey_3,rex_3))

    #coord_1 = torch.cat((X1,Y1), dim=0).unsqueeze(0).unsqueeze(2)

    #X = torch.arange(0, resX, device=cuda).view(resX).expand((1,resY,resX))
    #Y = torch.arange(0, resY, device=cuda).view(resY, 1).expand((1,resY,resX))
    #coord_a = torch.cat((X,Y), dim=0).unsqueeze(0).unsqueeze(2)

    #coord = torch.zeros((1,2,1,resY,resX))
    #coord[:,0,:,(rey_3):(rey_23-1),rex_3:(rex_23-1)]= coord_1[:,0]
    #coord[:,1,:,:,:]=coord_a[:,1]

    #coord= coord.cuda()

    #normalized_x = (coord[:,0].float()/np.float(rex_3)).cuda()
    #normalized_y = (coord[:,1].float()/np.float(resY)).cuda()


    # Old BC Distribution

    X = torch.arange(0, resX, device=cuda).view(resX).expand((1,resY,resX))
    Y = torch.arange(0, resY, device=cuda).view(resY, 1).expand((1,resY,resX))
    coord = torch.cat((X,Y), dim=0).unsqueeze(0).unsqueeze(2)


    normalized_x = (coord[:,0].float()/np.float(resX-1))
    normalized_y = (coord[:,1].float()/np.float(resY-1))

    # Atwood number
    #A = ((1+rho2) - (1+rho1)) / ((1+rho2) + (1+rho1))
    #print('Atwood number : ' + str(A))
    #density = ((1-A) * torch.tanh(100*(coord[:,1]/resY - (0.85 - \
    #                0.05*torch.cos(math.pi*(coord[:,0]/resX)))))).unsqueeze(1)
    thick = mconf['perturbThickness']
    ampl = mconf['perturbAmplitude']
    h = mconf['height']

    teta_cos = 2.0*math.pi*normalized_x

    #print("Nor x ", normalized_x[0,0,105,:])
    #print("Teta cos ", teta_cos[0,0,105,:])

    #density = 0.5*(rho2+rho1 + (rho2-rho1)*torch.tanh(thick*((coord[:,1]/resY).float() - \
    #        (h + ampl*torch.cos(2*math.pi*(coord[:,0]/resX).float()))))).unsqueeze(1)

    density = 0.5*(rho2+rho1 + (rho2-rho1)*torch.tanh(thick*(normalized_y - (h - ampl*torch.cos(teta_cos))))).unsqueeze(1)

    print("bottom",  0.5*(rho2+rho1 + (rho2-rho1)*math.tanh(thick*((1) - \
            (h + ampl*math.cos(2*math.pi*(64/resX)))))))

    print("top",  0.5*(rho2+rho1 + (rho2-rho1)*math.tanh(thick*( - \
            (h + ampl*math.cos(2*math.pi*(64/resX)))))))

    #print("density tensor", density[0,0,0,500:512, 1])
    #print("density tensor", density[0,0,0,0:10, -2])
    print("density", density.shape)
    print("height", h)

    print("Density ", density[0,0,0,:,0])

    batch_dict['density'] = density
    batch_dict['flags'] = flags

    # batch_dict at output = {p, UDiv, flags, density}

