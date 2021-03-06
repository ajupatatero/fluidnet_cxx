import glob
import argparse
import yaml

import torch
import torch.autograd
import time
from timeit import default_timer

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

import lib
import lib.fluid as fluid

# Usage python3 RayleighTaylor.py
# Use python3 RayleighTaylor.py -h for more details

#**************************** Load command line arguments *********************

parser = argparse.ArgumentParser(description='Rayleigh Taylor simulation. \n'
        'Read rayleighTaylorConfig.yaml for more information', \
        formatter_class= lib.SmartFormatter)
parser.add_argument('--simConf',
        default='rayleighTaylorConfig.yaml',
        help='R|Simulation yaml config file.\n'
        'Overwrites parameters from trainingConf file.\n'
        'Default: rayleighTaylorConfig.yaml')
parser.add_argument('--trainingConf',
        default='trainConfig.yaml',
        help='R|Training yaml config file.\n'
        'Default: config.yaml')
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

# Create the configuratio, re starting and the average density tensors

restart_config_file = glob.os.path.join('/', folder, 'rayleighTaylorConfig.yaml')
restart_state_file = glob.os.path.join('/', folder, 'restart.pth')
save_dist_file = glob.os.path.join('/', folder, 'growth.npy')
save_rho_file = glob.os.path.join('/', folder, 'avg_density.npy')

if restart_sim:
    # Check if configPlume.yaml exists in folder
    assert glob.os.path.isfile(restart_config_file), 'YAML config file does not exists for restarting.'
    with open(restart_config_file) as f:
        simConfig = yaml.load(f)

# Read from either Config file or from argument parser
simConf['modelDir'] = arguments.modelDir or simConf['modelDir']
assert (glob.os.path.exists(simConf['modelDir'])), 'Directory ' + str(SimConf['modelDir']) + ' does not exists'
simConf['modelFilename'] = arguments.modelFilename or simConf['modelFilename']
simConf['modelDirname'] = simConf['modelDir'] + '/' + simConf['modelFilename']
resume = False # For training, at inference set always to false


# Check cuda device and create a temporal model
print('Active CUDA Device: GPU', torch.cuda.current_device())
print()
path = simConf['modelDir']
path_list = path.split(glob.os.sep)
saved_model_name = glob.os.path.join('/', *path_list, path_list[-1] + '_saved.py')
temp_model = glob.os.path.join('lib', path_list[-1] + '_saved_simulate.py')
copyfile(saved_model_name, temp_model)

assert glob.os.path.isfile(temp_model), temp_model  + ' does not exits!'
spec = importlib.util.spec_from_file_location('model_saved', temp_model)
model_saved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_saved)


try:
    #te = lib.FluidNetDataset(conf, 'te', save_dt=4, resume=resume) # Test instance of custom Dataset
    # Create the mconf dictionary
    mconf = {}
    #conf, mconf = te.createConfDict()

    #cpath = glob.os.path.join(simConf['modelDir'], simConf['modelFilename'] + '_conf.pth')
    mcpath = glob.os.path.join(simConf['modelDir'], simConf['modelFilename'] + '_mconf.pth')
    #assert glob.os.path.isfile(mcpath), cpath  + ' does not exits!'
    assert glob.os.path.isfile(mcpath), mcpath  + ' does not exits!'
    #conf.update(torch.load(cpath))
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

        cuda = torch.device('cuda')

        # Domain Resolution

        resX = simConf['resX']
        resY = simConf['resY']

        #Declare the model variables: p, U, flags density and U star, div_input (which
        # correspond to the injected divergence and non corrected velocity field)

        p =       torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        U =       torch.zeros((1,2,1,resY,resX), dtype=torch.float).cuda()
        Ustar =   torch.zeros((1,2,1,resY,resX), dtype=torch.float).cuda()
        flags =   torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        density = torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()
        div_input =  torch.zeros((1,1,1,resY,resX), dtype=torch.float).cuda()

        # Domain initialization
        fluid.emptyDomain(flags)

        #flags[0,0,0,1:-1,0]= torch.ones((resY-2), dtype=torch.float).cuda()
        #flags[0,0,0,1:-1,-1]=torch.ones((resY-2), dtype=torch.float).cuda()

        # Start dictionary variables and load if results will be plotted, saved and which 
        # solver system will be used. Iter Max and both densities are loaded as well

        batch_dict = {}
        batch_dict['p'] = p
        batch_dict['U'] = U
        batch_dict['flags'] = flags
        batch_dict['density'] = density
        batch_dict['Ustar'] = Ustar
        batch_dict['Div_in']= div_input

        batch_dict['Test_case']= 'RT'

        real_time = simConf['realTimePlot']
        save_vtk = simConf['saveVTK']
        method = simConf['simMethod']
        max_iter = simConf['maxIter']
        outIter = simConf['statIter']

        rho1 = mconf['rho1']
        rho2 = mconf['rho2']

        # Periodicity is hard coded! This could be later on changed

        mconf['periodic-y'] = False
        mconf['periodic-x'] = False

        # Initialize it
        it = 0

        # Initialize and save the model
        net = model_saved.FluidNet(mconf, it, folder, dropout=False)
        if torch.cuda.is_available():
            net = net.cuda()
        net.load_state_dict(state['state_dict'])

        # Initial mean density

        density_a =  batch_dict['density'].clone()
        rho_avg = torch.mean(density_a).item()
        print("Avg rho 0 = " + str(rho_avg))

        # Create Initial Rayleigh Taylor BCs
        print('Creating initial conditions')
        fluid.createRayleighTaylorBCs(batch_dict, mconf, rho1=rho1, rho2=rho2)


        # Creation of Matrix A
        #if mconf['simMethod']=='CG':
        #    A = fluid.createMatrixA(flags)
        #    A_val, I_A, J_A = fluid.CreateCSR(A)
        #    batch_dict['Val']= A_val
        #    batch_dict['IA']= I_A
        #    batch_dict['JA']= J_A

        # Create the A matrix in the beggining!, this will give us the basic pressure field, 
        # which will be substracted at each step

        print("--------------------------------------------------------------")
        print("------------------- A matrix creation ------------------------")
        print("--------------------------------------------------------------")

        A_val, I_A, J_A = fluid.CreateCSR_Direct(flags)

        batch_dict['Val']= A_val
        batch_dict['IA']= I_A
        batch_dict['JA']= J_A


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

        # File with average density
        if not restart_sim:
            avg_density = np.empty((0,2))
            inst_dist = np.empty((0,2))
        else:
            avg_density = np.load(save_rho_file)

        torch.set_printoptions(precision=3, edgeitems = 5)

        my_map = cm.jet
        my_map.set_bad('gray')

        skip = 20
        scale = 12
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

        if real_time:
            fig = plt.figure(figsize=(20,10))
            gs = gridspec.GridSpec(1,5,
                 wspace=0.5, hspace=0.2)

            fig.show()
            fig.canvas.draw()
            ax_rho = fig.add_subplot(gs[0], frameon=False, aspect=1)
            cax_rho = make_axes_locatable(ax_rho).append_axes("right", size="5%", pad="2%")
            ax_velx = fig.add_subplot(gs[1], frameon=False, aspect=1)
            cax_velx = make_axes_locatable(ax_velx).append_axes("right", size="5%", pad="2%")
            ax_vely = fig.add_subplot(gs[2], frameon=False, aspect=1)
            cax_vely = make_axes_locatable(ax_vely).append_axes("right", size="5%", pad="2%")
            ax_p = fig.add_subplot(gs[3], frameon=False, aspect=1)
            cax_p = make_axes_locatable(ax_p).append_axes("right", size="5%", pad="2%")
            ax_div = fig.add_subplot(gs[4], frameon=False, aspect=1)
            cax_div = make_axes_locatable(ax_div).append_axes("right", size="5%", pad="2%")
            qx = ax_rho.quiver(X[:maxX_win:skip], Y[:maxY_win:skip],
                u1[minY:maxY:skip,minX:maxX:skip],
                v1[minY:maxY:skip,minX:maxX:skip],
                scale_units = 'height',
                scale=scale,
                #headwidth=headwidth, headlength=headlength,
                color='black')

        # Vector initialization. These include: 
        # -- Time Vec: Time taken for the Poisson equation,
        # -- Time_Pres: Time taken for the Pressure correction, 
        # -- Time_big: Time taken for the entire simulate iteration (advection + pressure)
        # -- Jacobi_switch (how many Jacobi iterations are made in the Hybrid strategy
        # and Max_Div + Max_Div_All (max divergence for all)

        Time_vec = np.zeros(max_iter)
        Time_Pres = np.zeros(max_iter)
        time_big = np.zeros(max_iter)
        Jacobi_switch = np.zeros(max_iter)
        Max_Div = np.zeros(max_iter)
        Max_Div_All = np.zeros(max_iter)

        while (it < max_iter):
            
            # Load simulation dt, if the hybrid method will be executed and  which will be the threshold
            dt = arguments.setdt or simConf['dt']
            Outside_Ja = simConf['outside_Ja']
            Threshold_Div = arguments.setThreshold or simConf['threshold_Div']


            if it<1:
                method='CG'
            else:
                method=simConf['simMethod']

            #Begin big Time timer!
            start_big = default_timer()

            #lib.simulate(mconf, batch_dict, net, method, it)
            lib.simulate(mconf, batch_dict, net, method, Time_vec, Time_Pres ,Jacobi_switch, Max_Div, Max_Div_All, folder, it,Threshold_Div, dt,Outside_Ja)

            # End lib simulate ... Save this into Time_big
            end_big = default_timer()

            tensor_div = fluid.velocityDivergence(batch_dict['U'].clone(),batch_dict['flags'].clone())

            print("=========================================================================")
            print("")
            print("BIG SIMULATE:--------------------------------->  ", (end_big - start_big))
            print("")
            print("FINAL Div:------------------------------------>  ", abs(tensor_div).max().item())
            print("")
            print("=========================================================================")

            time_big[it] = end_big - start_big

            filename_bigtime = folder + '/Time_big'
            np.save(filename_bigtime,time_big)

            density = batch_dict['density'].clone()
            center_X = resX // 2
            rho_at_center = density[0,0,0,:, center_X]
            h = density.size(3)
            signChange = (rho_at_center[0:h-1] < 0) \
                            .__and__(rho_at_center[1:h] > 0 )
            idx_change = signChange.nonzero()
            idx_change_p = idx_change + 1
            rho_1 = rho_at_center[idx_change]
            rho_2 = rho_at_center[idx_change+1]
            m = rho_1 - rho_2
            interpol = (rho_1/m)
            distance = (idx_change.float() + interpol) - resY // 2
            dt = mconf['dt']
            inst_dist = np.append(inst_dist, [[it*dt, distance]],
                    axis=0)
            np.save(save_dist_file, inst_dist)

            if (it% outIter == 0):
                print("It = " + str(it))
                tensor_div = fluid.velocityDivergence(batch_dict['U'].clone(),
                        batch_dict['flags'].clone())
                print("Div FINAL PRINTING :", (abs(tensor_div).max()).item())
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

                # As there is no source of fluid, the average density should be conserved
                rho_avg = torch.mean(density).item()
                #print("Avg rho = " + str(rho_avg))
                avg_density = np.append(avg_density, [[it, rho_avg]],
                        axis=0)
                np.save(save_rho_file, avg_density)

                img_velx_masked = ma.array(img_velx, mask=np_mask)
                img_vely_masked = ma.array(img_vely, mask=np_mask)
                img_vel_norm_masked = ma.array(img_vel_norm, mask=np_mask)
                ma.set_fill_value(img_velx_masked, np.nan)
                ma.set_fill_value(img_vely_masked, np.nan)
                ma.set_fill_value(img_vel_norm_masked, np.nan)
                img_velx_masked = img_velx_masked.filled()
                img_vely_masked = img_vely_masked.filled()
                img_vel_norm_masked = img_vel_norm_masked.filled()

                if real_time:
                    cax_rho.clear()
                    cax_velx.clear()
                    cax_vely.clear()
                    cax_p.clear()
                    cax_div.clear()
                    fig.suptitle("it = " + str(it), fontsize=16)
                    im0 = ax_rho.imshow(rho[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_rho.set_title('Density')
                    fig.colorbar(im0, cax=cax_rho, format='%.0e')
                    qx.set_UVC(img_velx[minY:maxY:skip,minX:maxX:skip],
                           img_vely[minY:maxY:skip,minX:maxX:skip])

                    im1 = ax_velx.imshow(img_velx[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_velx.set_title('x-velocity')
                    fig.colorbar(im1, cax=cax_velx, format='%.0e')
                    im2 = ax_vely.imshow(img_vely[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_vely.set_title('y-velocity')
                    fig.colorbar(im2, cax=cax_vely, format='%.0e')
                    im3 = ax_p.imshow(p[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_p.set_title('pressure')
                    fig.colorbar(im3, cax=cax_p, format='%.0e')
                    im4 = ax_div.imshow(div[minY:maxY,minX:maxX],
                        cmap=my_map,
                        origin='lower',
                        interpolation='none')
                    ax_div.set_title('divergence')
                    fig.colorbar(im4, cax=cax_div, format='%.0e')

                    fig.canvas.draw()
                    filename = folder + '/output_{0:05}.png'.format(it)
                    fig.savefig(filename)

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
                    div = fluid.velocityDivergence(\
                        batch_dict['U'].clone(), \
                        batch_dict['flags'].clone())[0,0]
                    velstar_div = fluid.velocityDivergence(\
                        batch_dict['Ustar'].clone(), \
                        batch_dict['flags'].clone())[0,0]
                    vel = fluid.getCentered(batch_dict['U'].clone())
                    velstar = fluid.getCentered(batch_dict['Ustar'].clone())
                    density = batch_dict['density'][0,0].clone()
                    pressure = batch_dict['p'][0,0].clone()
                    velX = vel[0,0].clone()
                    velY = vel[0,1].clone()
                    velstarX = velstar[0,0].clone()
                    velstarY = velstar[0,1].clone()
                    flags = batch_dict['flags'][0,0].clone()

                    # Change shape form (D,H,W) to (W,H,D)
                    div.transpose_(0,2).contiguous()
                    div_input.transpose_(0,2).contiguous()
                    velstar_div.transpose_(0,2).contiguous()
                    density.transpose_(0,2).contiguous()
                    pressure.transpose_(0,2).contiguous()
                    velX.transpose_(0,2).contiguous()
                    velY.transpose_(0,2).contiguous()
                    velstarX.transpose_(0,2).contiguous()
                    velstarY.transpose_(0,2).contiguous()
                    flags.transpose_(0,2).contiguous()

                    div_np = div.cpu().data.numpy()
                    div_input_np = div_input.cpu().data.numpy()
                    velstardiv_np = velstar_div.cpu().numpy()
                    density_np = density.cpu().data.numpy()
                    pressure_np = pressure.cpu().data.numpy()
                    velX_np = velX.cpu().data.numpy()
                    velY_np = velY.cpu().data.numpy()
                    velstarX_np = velstarX.cpu().data.numpy()
                    velstarY_np = velstarY.cpu().data.numpy()
                    np_mask = flags.eq(2).cpu().data.numpy().astype(float)
                    pressure_masked = ma.array(pressure_np, mask=np_mask)
                    velx_masked = ma.array(velX_np, mask=np_mask)
                    vely_masked = ma.array(velY_np, mask=np_mask)
                    velstarx_masked = ma.array(velstarX_np, mask=np_mask)
                    velstary_masked = ma.array(velstarY_np, mask=np_mask)
                    ma.set_fill_value(pressure_masked, np.nan)
                    ma.set_fill_value(velx_masked, np.nan)
                    ma.set_fill_value(vely_masked, np.nan)
                    ma.set_fill_value(velstarx_masked, np.nan)
                    ma.set_fill_value(velstary_masked, np.nan)
                    pressure_masked = pressure_masked.filled()
                    velx_masked = velx_masked.filled()
                    vely_masked = vely_masked.filled()
                    velstarx_masked = velstarx_masked.filled()
                    velstary_masked = velstary_masked.filled()

                    divergence = np.ascontiguousarray(div_np[minX:maxX,minY:maxY])
                    divergence_input=np.ascontiguousarray(div_input_np[minX:maxX,minY:maxY])
                    rho = np.ascontiguousarray(density_np[minX:maxX,minY:maxY])
                    p = np.ascontiguousarray(pressure_masked[minX:maxX,minY:maxY])
                    velx = np.ascontiguousarray(velx_masked[minX:maxX,minY:maxY])
                    vely = np.ascontiguousarray(vely_masked[minX:maxX,minY:maxY])
                    velstarx = np.ascontiguousarray(velstarx_masked[minX:maxX,minY:maxY])
                    velstary = np.ascontiguousarray(velstary_masked[minX:maxX,minY:maxY])
                    velstardiv = np.ascontiguousarray(velstardiv_np[minX:maxX,minY:maxY])
                    filename = folder + '/output_{0:05}'.format(it)
                    vtk.gridToVTK(filename, x, y, z, cellData = {
                        'density': rho,
                        'divergence': divergence,
                        'pressure' : p,
                        'ux' : velx,
                        'uy' : vely,
                        'u_star_x' : velstarx,
                        'u_star_y' : velstary,
                        'u_star_div' : velstardiv,
                        'divergence_input': divergence_input,
                        })

                    restart_dict = {'batch_dict': batch_dict, 'it': it}
                    torch.save(restart_dict, restart_state_file)

            it += 1

finally:
    # Delete model_saved.py
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)

