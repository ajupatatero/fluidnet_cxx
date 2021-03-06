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

import lib
import lib.fluid as fluid

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import fluid, MultiScaleNet
from math import inf


# Usage python3 plume.py
# Use python3 plume.py -h for more details

#**************************** Load command line arguments *********************

parser = argparse.ArgumentParser(description='Buoyant plume simulation. \n'
        'Read plumeConfig.yaml for more information', \
        formatter_class= lib.SmartFormatter)
parser.add_argument('--simConf',
        default='plumeConfig.yaml',
        help='R|Simulation yaml config file.\n'
        'Overwrites parameters from trainingConf file.\n'
        'Default: plumeConfig.yaml')
parser.add_argument('--trainingConf',
        default='trainConfig.yaml',
        help='R|Training yaml config file.\n'
        'Default: trainConfig.yaml')
parser.add_argument('--modelDir',
        help='R|Neural network model location.\n'
        'Default: written in simConf file.')
parser.add_argument('--modelFilename',
        help='R|Model name.\n'
        'Default: written in simConf file.')
parser.add_argument('--outputFolder',
        help='R|Folder for sim output.\n'
        'Default: written in simConf file.')
parser.add_argument('--restartSim', action='store_true', default=False,
        help='R|Restarts simulation from checkpoint.\n'
        'Default: written in simConf file.')
parser.add_argument('-sT','--setThreshold',
        help='R|Sets the Divergency Threshold.\n'
        'Default: written in simConf file.', type =float)
parser.add_argument('-delT','--setdt',
        help='R|Sets the dt.\n'
        'Default: written in simConf file.', type =float)

#Cylinder Test
parser.add_argument('--Cylinder',
        default=True,
        help='R|Includes a cylinder in the domain.\n'
        'Default: Trye--ue')


arguments = parser.parse_args()

# Loading a YAML object returns a dict
with open(arguments.simConf, 'r') as f:
    simConf = yaml.load(f, Loader=yaml.FullLoader)
with open(arguments.trainingConf, 'r') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

if not arguments.restartSim:
    restart_sim = simConf['restartSim']
else:
    restart_sim = arguments.restartSim

folder = arguments.outputFolder or simConf['outputFolder']
if (not glob.os.path.exists(folder)):
    glob.os.makedirs(folder)

restart_config_file = glob.os.path.join('/', folder, 'plumeConfig.yaml')
restart_state_file = glob.os.path.join('/', folder, 'restart.pth')
if restart_sim:
    # Check if configPlume.yaml exists in folder
    assert glob.os.path.isfile(restart_config_file), 'YAML config file does not exists for restarting.'
    with open(restart_config_file) as f:
        simConfig = yaml.load(f, Loader=yaml.FullLoader)

simConf['modelDir'] = arguments.modelDir or simConf['modelDir']
assert (glob.os.path.exists(simConf['modelDir'])), 'Directory ' + str(simConf['modelDir']) + ' does not exists'
simConf['modelFilename'] = arguments.modelFilename or simConf['modelFilename']
simConf['modelDirname'] = simConf['modelDir'] + '/' + simConf['modelFilename']
resume = False # For training, at inference set always to false


print('Active CUDA Device: GPU', torch.cuda.current_device())
print()
path = simConf['modelDir']
path_list = path.split(glob.os.sep)
saved_model_name = glob.os.path.join('/', *path_list, path_list[-1] + '_saved.py')
temp_model = glob.os.path.join('lib', path_list[-1] + '_saved_simulate.py')
copyfile(saved_model_name, temp_model)

assert glob.os.path.isfile(temp_model), temp_model  + ' does not exits!'
#importlib.util.spec_from_file_location(name, location, *, loader=None, submodule_search_locations=None)
#A factory function for creating a ModuleSpec instance based on the path to a file. Missing information will be filled in on the spec by making use of loader APIs and by the implication that the module will be file-based.
spec = importlib.util.spec_from_file_location('model_saved', temp_model)
#importlib.util.module_from_spec(spec)
#Create a new module based on spec and spec.loader.create_module.
#If spec.loader.create_module does not return None, then any pre-existing attributes will not be reset. Also, no AttributeError will be raised if triggered while accessing spec or setting an attribute on the module.
#This function is preferred over using types.ModuleType to create a new module as spec is used to set as many import-controlled attributes on the module as possible.
model_saved = importlib.util.module_from_spec(spec)
#exec_module(module)
#An abstract method that executes the module in its own namespace when a module is imported or reloaded. The module should already be initialized when exec_module() is called. When this method exists, create_module() must be defined.
spec.loader.exec_module(model_saved)

try:
    mconf = {}

    mcpath = glob.os.path.join(simConf['modelDir'], simConf['modelFilename'] + '_mconf.pth')
    assert glob.os.path.isfile(mcpath), mcpath  + ' does not exits!'
    mconf.update(torch.load(mcpath))

    print('==> overwriting mconf with user-defined simulation parameters')
    # Overwrite mconf values with user-defined simulation values.
    mconf.update(simConf)

    print('==> loading model')
    mpath = glob.os.path.join(simConf['modelDir'], simConf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
    state = torch.load(mpath)

    print('Data loading: done')

    #********************************** Create the model ***************************
    with torch.no_grad():

        it = 0
        cuda = torch.device('cuda')

        net = lib.FluidNetTGV(mconf, it, dropout=False)
        if torch.cuda.is_available():
            net = net.cuda()

        net.load_state_dict(state['state_dict'])

        #*********************** Simulation parameters **************************
        

        resX = simConf['resX']
        resY = simConf['resY']

        p =       torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        U =       torch.zeros((1,2,1,resY,resX), dtype=torch.float).cuda()
        Ustar =       torch.zeros((1,2,1,resY,resX), dtype=torch.float).cuda()
        flags =   torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        density = torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        div_input =  torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()

        fluid.emptyDomain(flags)
        batch_dict = {}
        batch_dict['p'] = p
        batch_dict['U'] = U
        batch_dict['Ustar'] = Ustar
        batch_dict['flags'] = flags
        batch_dict['density'] = density
        batch_dict['Div_in']= div_input
        #We create a temporary flags for the inflow, in order to avoid affecting the advection
        #flags_i = flags.clone()
        #batch_dict['flags_inflow'] = flags_i

        real_time = simConf['realTimePlot']
        save_vtk = simConf['saveVTK']
        method = simConf['simMethod']
        #it = 0

        dt = arguments.setdt or simConf['dt']
        Outside_Ja = simConf['outside_Ja']
        Threshold_Div = arguments.setThreshold or simConf['threshold_Div']

        max_iter = simConf['maxIter']
        outIter = simConf['statIter']

        rho1 = simConf['injectionDensity']
        rad = simConf['sourceRadius']
        plume_scale = simConf['injectionVelocity']

        #**************************** Initial conditions ***************************


        flags = batch_dict['flags']
        X = torch.arange(0, resX).view(resX).expand((1,1,1,resY,resX)).double()
        Y = torch.arange(0, resY).view(resY,1).expand((1,1,1,resY,resX)).double()

        print("X ", X.shape)
        print("X tensor ", X[0,0,0,0:10,0:10])
        print("Y tensor ", Y[0,0,0,0:10,0:10])

        init_Div = np.cos(2*X*2*3.141592/(resX)) + np.cos(2*Y*2*3.141592/(resY))

        div_input = init_Div.float().cuda()

        sol_P =-0.25*(np.cos(2*X*2*3.141592/(resX)) + np.cos(2*Y*2*3.141592/(resY)))
        sol_output = sol_P.float().cuda()
                  
        #XXX: Create Box2D and Cylinders from YAML config file
        # Uncomment to create Cylinder or Box2D obstacles
        #fluid.createCylinder(batch_dict, centerX=0.5*resX,
        #                                 centerY=0.5*resY,
        #                                 radius=50)
        #fluid.createBox2D(batch_dict, x0=0.5*resX, x1=0.5*resX,
        #                              y0=0.7*resY, y1=0.7*resY)

        # If restarting, overwrite all fields with checkpoint.
        if restart_sim:
            # Check if restart file exists in folder
            assert glob.os.path.isfile(restart_state_file), 'Restart file does not exists.'
            restart_dict = torch.load(restart_state_file)
            batch_dict = restart_dict['batch_dict']
            it = restart_dict['it']
            print('Restarting from checkpoint at it = ' + str(it))

        # Create YAML file in output folder
        with open(restart_config_file, 'w') as outfile:
                yaml.dump(simConf, outfile)

        # Print options for debug
        # Number of array items in summary at beginning and end of each dimension (default = 3).
        torch.set_printoptions(precision=1, edgeitems = 5)

        # Parameters for matplotlib draw
        my_map = cm.jet
        my_map.set_bad('gray')

        skip = 10
        scale = 20
        scale_units = 'xy'
        angles = 'xy'
        headwidth = 0.8#2.5
        headlength = 5#2

        minY = 0
        maxY = resY
        maxY_win = resY
        minX = 0
        maxX = resX
        maxX_win = resX
        X, Y = np.linspace(0, resX-1, num=resX),\
                np.linspace(0, resY-1, num=resY)

        tensor_vel = batch_dict['U'].clone()
        u1 = (torch.zeros_like(torch.squeeze(tensor_vel[:,0]))).cpu().data.numpy()
        v1 = (torch.zeros_like(torch.squeeze(tensor_vel[:,0]))).cpu().data.numpy()

        #Debug
        #print("FLAGS", flags[0,0,0,0:5,40:80])

        # Initialize figure
        if real_time:
            fig = plt.figure(figsize=(20,20))
            gs = gridspec.GridSpec(2,3,
                 wspace=0.5, hspace=0.2)
            fig.show()
            ax_rho = fig.add_subplot(gs[0,0], frameon=False, aspect=1)
            cax_rho = make_axes_locatable(ax_rho).append_axes("right", size="5%", pad="2%")
            ax_velx = fig.add_subplot(gs[0,1], frameon=False, aspect=1)
            cax_velx = make_axes_locatable(ax_velx).append_axes("right", size="5%", pad="2%")
            ax_vely = fig.add_subplot(gs[0,2], frameon=False, aspect=1)
            cax_vely = make_axes_locatable(ax_vely).append_axes("right", size="5%", pad="2%")
            ax_p = fig.add_subplot(gs[1,0], frameon=False, aspect=1)
            cax_p = make_axes_locatable(ax_p).append_axes("right", size="5%", pad="2%")
            ax_div = fig.add_subplot(gs[1,1], frameon=False, aspect=1)
            cax_div = make_axes_locatable(ax_div).append_axes("right", size="5%", pad="2%")
            ax_cut = fig.add_subplot(gs[1,2],frameon=False, aspect="auto")
            cax_cut = make_axes_locatable(ax_cut).append_axes("right", size="5%", pad="2%")
            qx = ax_rho.quiver(X[:maxX_win:skip], Y[:maxY_win:skip],
                u1[minY:maxY:skip,minX:maxX:skip],
                v1[minY:maxY:skip,minX:maxX:skip],
                scale_units = 'height',
                scale=scale,
                #headwidth=headwidth, headlength=headlength,
                color='black')
       
        #Time Vec Declaration
        Time_vec = np.zeros(max_iter)
        Time_Pres = np.zeros(max_iter)
        Jacobi_switch = np.zeros(max_iter)
        Max_Div = np.zeros(max_iter)
        Max_Div_All = np.zeros(max_iter)
        time_big = np.zeros(max_iter)

        # Probe and Plotting
        Probe_U_y = np.zeros(max_iter)
        range_plt = np.arange(max_iter)

        data = torch.cat((div_input, flags), 1)
        # Main loop
        while (it < max_iter):
            #if it < 750:
            #    method = 'jacobi'
            #else:
            method = mconf['simMethod']
            #(mconf, batch_dict, net, sim_method, Time_vec, folder, it, output_div=False)
            start_big = default_timer()

            net.eval()

            p, div, time = net(data,it)

            batch_dict['p'] = p
            batch_dict['Div_in']= div

            end_big = default_timer()
            time_big[it] = (end_big - start_big)


            if (it% outIter == 0):
                print("It = " + str(it))
                tensor_div = fluid.velocityDivergence(batch_dict['U'].clone(),
                        batch_dict['flags'].clone())
                pressure = batch_dict['p'].clone()
                tensor_vel = fluid.getCentered(batch_dict['U'].clone())
                density = batch_dict['density'].clone()
                div = torch.squeeze(tensor_div).cpu().data.numpy()
                np_mask = torch.squeeze(flags.eq(2)).cpu().data.numpy().astype(float)
                rho = torch.squeeze(density).cpu().data.numpy()
                p = torch.squeeze(pressure).cpu().data.numpy()
                img_norm_vel = torch.squeeze(torch.norm(tensor_vel,
                    dim=1, keepdim=True)).cpu().data.numpy()
                img_velx = torch.squeeze(tensor_vel[:,0]).cpu().data.numpy()
                img_vely = torch.squeeze(tensor_vel[:,1]).cpu().data.numpy()
                img_vel_norm = torch.squeeze( \
                        torch.norm(tensor_vel, dim=1, keepdim=True)).cpu().data.numpy()

                img_velx_masked = ma.array(img_velx, mask=np_mask)
                img_vely_masked = ma.array(img_vely, mask=np_mask)
                img_vel_norm_masked = ma.array(img_vel_norm, mask=np_mask)
                ma.set_fill_value(img_velx_masked, np.nan)
                ma.set_fill_value(img_vely_masked, np.nan)
                ma.set_fill_value(img_vel_norm_masked, np.nan)
                img_velx_masked = img_velx_masked.filled()
                img_vely_masked = img_vely_masked.filled()
                img_vel_norm_masked = img_vel_norm_masked.filled()

                filename12 = folder + '/Probe_U_y'
                np.save(filename12,Probe_U_y)

                if save_vtk:
                    px, py = 1580, 950
                    dpi = 100
                    figx = px / dpi
                    figy = py / dpi

                    nx = maxX_win
                    ny = maxY_win
                    nz = 1
                    ncells = nx*ny*nz

                    ratio = nx/ny
                    lx, ly = ratio, 1.0
                    dx, dy = lx/nx, ly/ny

                    # Coordinates
                    x = np.arange(0, lx + 0.1*dx, dx, dtype='float32')
                    y = np.arange(0, ly + 0.1*dy, dy, dtype='float32')
                    z = np.zeros(1, dtype='float32')

                    # Variables
                    div_input = batch_dict['Div_in'][0,0].clone()
                    pressure = batch_dict['p'].clone()
                    b = 1
                    w = pressure.size(4)
                    h = pressure.size(3)
                    d = pressure.size(2)

                    solution = sol_output[0,0]
                    pressure = pressure[0,0]
                    flags = batch_dict['flags'][0,0].clone()

                    # Change shape form (D,H,W) to (W,H,D)
                    div_input.transpose_(0,2).contiguous()
                    pressure.transpose_(0,2).contiguous()
                    flags.transpose_(0,2).contiguous()
                    solution.transpose_(0,2).contiguous()

                    div_input_np = div_input.cpu().data.numpy()
                    pressure_np = pressure.cpu().data.numpy()
                    solution_np = solution.cpu().data.numpy()
                    np_mask = flags.eq(2).cpu().data.numpy().astype(float)
                    pressure_masked = ma.array(pressure_np, mask=np_mask)
                    ma.set_fill_value(pressure_masked, np.nan)
                    pressure_masked = pressure_masked.filled()

                    divergence_input=np.ascontiguousarray(div_input_np[minX:maxX,minY:maxY])
                    solution_input = np.ascontiguousarray(solution_np[minX:maxX,minY:maxY])
                    p = np.ascontiguousarray(pressure_masked[minX:maxX,minY:maxY])
                    filename = folder + '/output_{0:05}'.format(it)
                    vtk.gridToVTK(filename, x, y, z, cellData = {
                        'pressure' : p,
                        'divergence_input': divergence_input,
                        'Solution P': solution_input,
                        })

                restart_dict = {'batch_dict': batch_dict, 'it': it}
                torch.save(restart_dict, restart_state_file)

            # Update iterations
            it += 1

finally:
    # Properly deleting model_saved.py, even when ctrl+C
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)


