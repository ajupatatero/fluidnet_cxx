import torch
import lib.fluid as fluid
import numpy as np
from timeit import default_timer
import pdb

def setConstVals(batch_dict, p, U, flags, density):
    # apply external BCs.
    # batch_dict at output = {p, UDiv, flags, density, UBC,
    #                         UBCInvMask, densityBC, densityBCInvMask}

    #if 'cylinder' in batch_dict:
    #    # Zero out the U values on the BCs.
    #    U.mul_(batch_dict['InvInletMask'])
    #    # Add back the values we want to specify.
    #    U.add_(batch_dict['UInlet'])
    #    batch_dict['U'] = U.clone()


    if ('UBCInvMask' in batch_dict) and ('UBC' in batch_dict):

        # Zero out the U values on the BCs.
        #U.mul_(batch_dict['UBCInvMask'])

        Mask = batch_dict['densityBCInvMask']
        # Ekhi 19/08/19 Parabolic Velocity profile

        #Inside_cyl = (Mask[0,0,0,1,:].ne(1)).sum() 
        #xdim = U.size(4)
        #centerX = xdim // 2
        #firstX = centerX - (Inside_cyl//2)
        #lastX = centerX + (Inside_cyl//2) + 1

        #X = torch.arange(0, Inside_cyl)
        #normalized_x = (X.float()/np.float(Inside_cyl-1))
        #new_nor = (((normalized_x).unsqueeze(0)).unsqueeze(0)).unsqueeze(0)
        #nor_2 = new_nor.expand_as(U[:,1,:,1:4,firstX:lastX])

        U[:,:,:,1:2,:]=0
        U.mul_(batch_dict['densityBCInvMask'])

        # Add back the values we want to specify.

        ##### THIS LINE MAKES IT SQIUARE OR NOT
        U.add_(batch_dict['UBC'])

        #U[:,1,:,1:4,firstX:lastX]= 6*0.05*(nor_2)*(1-nor_2)
        #U[:,1,:,1:4,firstX:lastX]=0.05
 
        batch_dict['U'] = U.clone()

    if ('densityBCInvMask' in batch_dict) and ('densityBC' in batch_dict):
        Mask = batch_dict['densityBCInvMask']
        #print(" densityBCInvMask ", Mask[0,0,0,0:2,:])
        density[:,:,:,1:4,:]=0
        density.mul_(batch_dict['densityBCInvMask'])
        density.add_(batch_dict['densityBC'])
        batch_dict['density'] = density.clone()

#def simulate(mconf, batch_dict, net, sim_method, output_div=False):
def simulate(mconf, batch_dict, net, sim_method, Time_vec, Time_Pres,Jacobi_switch, Max_Div, Max_Div_All, folder, it, Threshold_Div, dt, Outside_Ja, output_div=False):
    r"""Top level simulation loop.

    Arguments:
        mconf (dict): Model configuration dictionnary.
        batch_dict (dict): Dictionnary of torch Tensors.
            Keys must be 'U', 'flags', 'p', 'density'.
            Simulations are done INPLACE.
        net (nn.Module): convNet model.
        sim_method (string): Options are 'convnet', 'PCG' and 'jacobi'
        output_div (bool, optional): returns just before solving for pressure.
            i.e. leave the state as UDiv and pDiv (before substracting divergence)
        it: only for dubugging, number of iteration

    """

       
    cuda = torch.device('cuda')
    assert sim_method == 'convnet' or sim_method == 'jacobi' or sim_method == 'PCG'\
            or sim_method == 'CG', 'Simulation method \
                not supported. Choose either convnet, PCG, CG or jacobi.'
    
    #dt = arguments.setdt or float(mconf['dt'])
    maccormackStrength = mconf['maccormackStrength']
    sampleOutsideFluid = mconf['sampleOutsideFluid']

    buoyancyScale = mconf['buoyancyScale']
    gravityScale = mconf['gravityScale']

    viscosity = mconf['viscosity']
    assert viscosity >= 0, 'Viscosity must be positive'

    # Get p, U, flags and density from batch.
    p = batch_dict['p']
    U = batch_dict['U']
    Ustar = batch_dict['Ustar']
    flags = batch_dict['flags']
   
    #flags_i = batch_dict['flags_inflow']  

    if (it% 1 == 0):
        U_inx = U.clone()
        Uinter1_cpu = U_inx.cpu()

        filename_inter1 = folder + '/Ux_Before_Advection_{0:05}'.format(it)
        #np.save(filename_inter1,Uinter1_cpu[:,0])

        filename_inter2 = folder + '/Uy_Before_Advection_{0:05}'.format(it)
        #np.save(filename_inter2,Uinter1_cpu[:,1])

        Density_1 = batch_dict['density']
        #print("Density 2/4 ", Density_1[0,0,0,:,0])
        Rho_inx = Density_1.clone()
        Rhointer_cpu = Rho_inx.cpu()
        filename_rhointer1 = folder + '/Rho_Before_Advection_{0:05}'.format(it)
        #np.save(filename_rhointer1,Rhointer_cpu)


    stick = False
    if 'flags_stick' in batch_dict:
        stick = True
        print(" Stick = True ")
        flags_stick = batch_dict['flags_stick']

    # If viscous model, add viscosity
    if (viscosity > 0):
        orig = U.clone()
        fluid.addViscosity(dt, orig, flags, viscosity)

    if 'density' in batch_dict:
        density = batch_dict['density']

        # First advect all scalar fields.
        density = fluid.advectScalar(dt, density, U, flags, \
                method="eulerFluidNet", \
                #method="maccormackFluidNet", \
                boundary_width=1, sample_outside_fluid=sampleOutsideFluid, \
                maccormack_strength=maccormackStrength)

        if mconf['correctScalar']:
            div = fluid.velocityDivergence(U, flags)
            fluid.correctScalar(dt, density, div, flags)
    else:
        density = torch.zeros_like(flags)


    if (it% 1 == 0):
        U_inx = U.clone()
        Uinter1_cpu = U_inx.cpu()

        filename_inter1 = folder + '/Ux_After_Rho_Advection_{0:05}'.format(it)
        #np.save(filename_inter1,Uinter1_cpu[:,0])

        filename_inter2 = folder + '/Uy_After_Rho_Advection_{0:05}'.format(it)
        #np.save(filename_inter2,Uinter1_cpu[:,1])

        Rho_inx = density.clone()
        Rhointer_cpu = Rho_inx.cpu()
        filename_rhointer1 = folder + '/Rho_After_Rho_Advection_{0:05}'.format(it)
        #np.save(filename_rhointer1,Rhointer_cpu)

    flags_only= flags.clone()

    if viscosity == 0:
        # Self-advect velocity if inviscid
        U = fluid.advectVelocity(dt=dt, orig=U, U=U, flags=flags, \
            method="eulerFluidNet", \
            #method="maccormackFluidNet", \
            boundary_width=1, maccormack_strength=maccormackStrength)
    else:
        # Advect viscous velocity field orig by the non-divergent
        # velocity field U.
        U = fluid.advectVelocity(dt=dt, orig=orig, U=U, flags=flags, \
            method="eulerFluidNet", \
            #method="maccormackFluidNet", \
            boundary_width=1, maccormack_strength=maccormackStrength)

    # Set the manual BCs.

    #Add Wall Bc
    #U = fluid.setWallBcs(U, flags)
    #setConstVals(batch_dict, p, U, flags, density)

    #pdb.set_trace() 

    #HERE, no matter the method, we should get the same velocity field in the it 50
    #Print After U
    #folder = '/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/results_plume_Debug/Inflow_channel/BC/Debug'
    
    #Uadvec_cpu = U.cpu()
    #filename_a = folder + '/U1_NN_After_Advection{0:05}'.format(it)
    #np.save(filename_a,Uadvec_cpu)

    if 'density' in batch_dict:
        if buoyancyScale > 0:
            # Add external forces: buoyancy.
            gravity = torch.FloatTensor(3).fill_(0).cuda()
            gravity[0] = mconf['gravityVec']['x']
            gravity[1] = mconf['gravityVec']['y']
            gravity[2] = mconf['gravityVec']['z']
            gravity.mul_(-buoyancyScale)
            rho_star = mconf['operatingDensity']

            print("Applying buoyancy !")
            #Chapuza n3
            U = fluid.addBuoyancy(U, flags, density, gravity, rho_star, dt)
            #U = fluid.addBuoyancy_NewSourceTerm(U, flags, density, gravity, rho_star, dt)
        if gravityScale > 0:
            gravity = torch.FloatTensor(3).fill_(0).cuda()
            gravity[0] = mconf['gravityVec']['x']
            gravity[1] = mconf['gravityVec']['y']
            gravity[2] = mconf['gravityVec']['z']
            # Add external forces: gravity.
            gravity.mul_(-gravityScale)
            U = fluid.addGravity(U, flags, gravity, dt)
            
        print("Done Bouyancy + gravity")


    #Print Before U
    #Uinter_cpu = U.cpu()
    #filename_inter = folder + '/U1_NN_Intermediate{0:05}'.format(it)
    #np.save(filename_inter,Uinter_cpu)

    # 19/09/19 Ekhi. We add the BC Ux = 0 on the right wall for the half domain!
    #wall_zero = torch.zeros_like(U[:,0,:,:,-1])
    #U[:,0,:,:,-1] = wall_zero
    #U[:,0,:,:,-2] = wall_zero

    #U = fluid.setWallBcs(U, flags)

    if sim_method != 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        #U = fluid.setWallBcs(U, flags)
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
                U[:,0,:,:,1] = U_temp[:,0,:,:,U.size(4)-1]
                density[:,0,:,:,1] = density_temp[:,0,:,:,U.size(4)-1]
            if mconf['periodic-y']:

                U[:,1,:,:,-1] = U_temp[:,1,:,:,1]
                U[:,0,:,:,-1] = -U_temp[:,0,:,:,1]
                #U[:,1,:,:,-2] = U_temp[:,1,:,:,2]
                #U[:,0,:,:,-2] = -U_temp[:,0,:,:,2]
                #density[:,:,:,:,-2] = density_temp[:,:,:,:,1]
                #density[:,:,:,:,-3] = density_temp[:,:,:,:,2]

                #U[:,1,:,:,1] = U_temp[:,1,:,:,-2]
                #density[:,0,:,:,1] = density_temp[:,0,:,:,-2]
                #U[:,1,:,:,1] = U_temp[:,1,:,:,-2]
                #U[:,1,:,:,2] = U_temp[:,1,:,:,-3]
                #U[:,0,:,:,2] = U_temp[:,0,:,:,-2]
                #density[:,:,:,:,1] = density_temp[:,:,:,:,-2]
                #density[:,:,:,:,2] = density_temp[:,:,:,:,-3]

    if sim_method == 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
 
                U[:,1,:,:,1] = U_temp[:,1,:,:,U.size(4)-1]
                density[:,0,:,:,1] = density_temp[:,0,:,:,U.size(4)-1]
 
            if mconf['periodic-y']:
 
                print(" Periodic y 0 !")
#                print("U copying ", U_temp[0,1,0,175:200,-2])
#                print("U modifying before ", U[0,1,0,175:200,1])
 
                U[:,1,:,:,-1] = U_temp[:,1,:,:,1]
                U[:,0,:,:,-1] = -U_temp[:,0,:,:,1]

    elif stick:
        fluid.setWallBcsStick(U, flags, flags_stick)

    #if sim_method == 'convnet':
    U = fluid.setWallBcs(U, flags)

    #Special VK
    if batch_dict['VK']:
        print("VK BC")
        BC_V = batch_dict['VK']
        U = fluid.setWallVKBcs(U, flags, BC_V)

    # We add the BC i order to have Divergency in the border!
    setConstVals(batch_dict, p, U, flags, density)

    batch_dict['U'] = U
    #U = fluid.setWallBcs(U, flags)  

    div = fluid.velocityDivergence(U, flags)
    Advected_Div = (abs(div).max()).item()

    #Print Before U
    #Uinter_cpu = U.cpu()
    #filename_inter = folder + '/U1_NN_Intermediate{0:05}'.format(it)
    #np.save(filename_inter,Uinter_cpu)

    #Print Before U
    if (it% 1 == 0):    
        U_inx = U.clone()
        Uinter1_cpu = U_inx.cpu()

        filename_inter1 = folder + '/Ux_After_Advection_{0:05}'.format(it)
        #np.save(filename_inter1,Uinter1_cpu[:,0])

        filename_inter2 = folder + '/Uy_After_Advection_{0:05}'.format(it)
        #np.save(filename_inter2,Uinter1_cpu[:,1])

        Rho_inx = density.clone()
        Rhointer_cpu = Rho_inx.cpu()
        filename_rhointer1 = folder + '/Rho_After_Advection_{0:05}'.format(it)
        #np.save(filename_rhointer1,Rhointer_cpu)


   # Save velocity field after the advection step!

    Ustar = U.clone()
   

    #Print Before U
    #Uinxb = U.clone()
    #Ubef_cpu = Uinxb.cpu()
    #filename_1 = folder + '/U_field/U1_Ja_Bef_update{0:05}'.format(it)
    #np.save(filename_1,Ubef_cpu)


    #Timing for the whole P solving
    start_Pres = default_timer()

    if (sim_method == 'convnet'):
        # fprop the model to perform the pressure projection and velocity calculation.
        # Set wall BCs is performed inside the model, before and after the projection.
        # No need to call it again.
 
        #UDiv = fluid.setWallBcs(UDiv, flags)
        #U = fluid.setWallBcs(U, flags)       
        #Input
        div_input = div.clone()

        batch_dict['Div_in'] = div_input

        #Big P NN
        start_NN = default_timer()

        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event = torch.cuda.Event(enable_timing=True)
        #start_event.record()

        net.eval()
        data = torch.cat((p, U, flags, density), 1)
        p, U, time = net(data, it)

        #end_event.record()
        #torch.cuda.synchronize()  # Wait for the events to be recorded!
        #elapsed_time_ms = start_event.elapsed_time(end_event)

        #print("Elapsed Time ", elapsed_time_ms)

        end_NN = default_timer()

       
        setConstVals(batch_dict, p, U, flags, density)

        #Special VK
        if batch_dict['VK']:
            print("VK BC")
            BC_V = batch_dict['VK']
            U = fluid.setWallVKBcs(U, flags, BC_V)

        U = batch_dict['U']

        print('NN inside end-start : ', end_NN - start_NN)
        
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
 
                U[:,1,:,:,1] = U_temp[:,1,:,:,U.size(4)-1]
                density[:,0,:,:,1] = density_temp[:,0,:,:,U.size(4)-1]
 
            if mconf['periodic-y']:
 
                print(" Periodic y 1 !")
#                print("U copying ", U_temp[0,1,0,175:200,-2])
#                print("U modifying before ", U[0,1,0,175:200,1])
 
                U[:,1,:,:,-1] = U_temp[:,1,:,:,1]
                U[:,0,:,:,-1] = -U_temp[:,0,:,:,1]

    elif (sim_method == 'jacobi'):


        div = fluid.velocityDivergence(U, flags)

        bsz = div.size(0)
        ch = div.size(1)
        d = div.size(2)
        h = div.size(3)
        w = div.size(4)

        #Chapuza Scaling!
        #div = div*20

        div_input= batch_dict['Div_in']

        #h_alt = torch.arange(0, h, dtype=torch.float, device=cuda).view(1,h,1).expand(bsz, ch,d, h, w)
        #div[:,:,:,:256,:]+=(torch.ones_like(div[:,:,:,:256,:])*h_alt[:,:,:,:256,:]-130)/4000
        #div[:,:,:,256:,:]+=(-1.0*torch.ones_like(div[:,:,:,256:,:])*(h_alt[:,:,:,256:,:]-256.0)+126)/4000

        div_input = div.clone()

        is3D = (U.size(2) > 1)
        pTol = mconf['pTol']
        load_file = folder + '/Jacobi_switch_loading.npy'
        #Maxi_Try = np.load(load_file)
        maxIter = mconf['jacobiIter']
        #if it < 20:
        #   maxIter = np.int(Maxi_Try[it]*10)
        #else:
        #   maxIter = np.int(Maxi_Try[it])

        #Print Before U
        #DUinxb = div.clone()
        #DUbef_cpu = DUinxb.cpu()
        #filename_3 = folder + '/U_field/Div_Ja_Bef_update{0:05}'.format(it)
        #np.save(filename_3,DUbef_cpu)

        start = default_timer()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()


        p, residual = fluid.solveLinearSystemJacobi( \
                flags=flags, div=div, is_3d=is3D, p_tol=pTol, \
                max_iter=maxIter)

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded! 
        elapsed_time_ms = start_event.elapsed_time(end_event)

        print("elapsed Jacobi time: ", elapsed_time_ms)

        end = default_timer()
        time=(end - start)
        print("time ", time, "it ", it )


        #Print Pressures
        #P_cpu = p.cpu()       
        #filename = folder + '/P_NN_output_{0:05}'.format(it)
        #np.save(filename,P_cpu)



        # Chapuza n 1
        # We add rho h to the pressure so that the velocity update includes the hydrostatic pressure

        #p_tuned = p.clone()

        #bsz = p.size(0)
        #c = p.size(1)
        #d = p.size(2)
        #h = p.size(3)
        #w =p.size(4)

        #j = torch.arange(0, h, dtype=torch.float, device=cuda).view(1,h,1).expand(bsz,c,d, h, w)
        #gravity_val = gravity[1].float()
        
        #p_tuned = p + density*gravity_val*j*dt

        #p = p_tuned

            #fluid.velocityUpdate_Density(pressure=p, U=U, flags=flags, density=density)
        fluid.velocityUpdate(pressure=p, U=U, flags=flags)

        t_BC_i = default_timer()

        setConstVals(batch_dict, p, U, flags, density)

        #Special VK
        if batch_dict['VK']:
            print("VK BC")
            BC_V = batch_dict['VK']
            U = fluid.setWallVKBcs(U, flags, BC_V)
        #U = fluid.setWallVKBcs(U, flags)
        U = batch_dict['U']

        t_BC_e = default_timer()

        print("BC time: ", t_BC_e - t_BC_i)

    elif (sim_method == 'CG'):

    
        div = fluid.velocityDivergence(U, flags)

        is3D = (U.size(2) > 1)
        pTol = mconf['pTol']
        maxIter_CG = 1000
        pTol_CG = 0.1e-6

        A_val = batch_dict['Val']
        I_A = batch_dict['IA']
        J_A = batch_dict['JA']

        #Input
        div_input = div.clone()
        
        batch_dict['Div_in'] = div_input

        #Timing Test
        start = default_timer()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        residual = fluid.solveLinearSystemCG(flags, p, div, \
            A_val, I_A, J_A,is_3d=is3D, p_tol=pTol_CG, \
            max_iter=maxIter_CG)

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded! 
        elapsed_time_ms = start_event.elapsed_time(end_event)
        
        print("elapsed CG time: ", elapsed_time_ms)

        end = default_timer()
        time=(end - start)
        print("time", time)

        fluid.velocityUpdate(pressure=p, U=U, flags=flags)
        setConstVals(batch_dict, p, U, flags, density)

        #Special VK
        if batch_dict['VK']:
            print("VK BC")
            BC_V = batch_dict['VK']
            U = fluid.setWallVKBcs(U, flags, BC_V)

        U = batch_dict['U']


    elif (sim_method == 'PCG'):
        is3D = (U.size(2) > 1)
        pTol = mconf['pTol']
        maxIter = mconf['jacobiIter']
        maxIter_PCG = 1 
        pTol_PCG = 0.5e-6  


        #Input
        div_input= batch_dict['Div_in']
        div_input = div.clone()

        #Timing Test
        start = default_timer()
        div_one = torch.zeros(5,5).float()
        flags_one = torch.ones(5,5).float()
        flags_one *= 2

        # Inflow 
        inflow = torch.zeros_like(flags)
        inflow_border = torch.zeros_like(flags)

        inflow_bool = False

        if inflow_bool:

            inflow = ((batch_dict['UBC'][:,1,:,:,:]).unsqueeze(1))>0.0001
            inflow_border[0,0,0,1,:]= inflow[0,0,0,1,:]

        #Debug
        print(" ========================================================================")
        print( "IT  ", it)
        print(" ========================================================================")


        """
        div_one = (((div_one.unsqueeze(0)).unsqueeze(0)).unsqueeze(0))
        flags_one = (((flags_one.unsqueeze(0)).unsqueeze(0)).unsqueeze(0))

        zer = flags_one.unsqueeze(0)

        for i in range(0,3):
            
            for j in range(0,3):
                flags_one[0][0][0][i+1][j+1]=  flags_one[0][0][0][i+1][j+1] -1

        U_only3 = torch.stack((U_only2,U_only), dim=1)

        print("U only 3 shape ", U_only3.shape)

        print("U only ", U_only)
        div_new = fluid.velocityDivergence(U_only3, flags_one)
        """

        p, residual = fluid.solveLinearSystemPCG( \
                flags=flags, div=div, inflow = inflow_border, is_3d=is3D, p_tol=pTol_PCG, \
                max_iter=maxIter_PCG)

        end = default_timer()
        time=(end - start)
        print("time", time)

        fluid.velocityUpdate(pressure=p, U=U, flags=flags)

    if sim_method != 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        U = fluid.setWallBcs(U, flags)
        setConstVals(batch_dict, p, U, flags, density)
        U = batch_dict['U']
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
                U[:,1,:,:,1] = U_temp[:,1,:,:,U.size(4)-1]
                density[:,0,:,:,1] = density_temp[:,0,:,:,U.size(4)-1]
            if mconf['periodic-y']:
                #U[:,1,:,:,1] = U_temp[:,1,:,:,-2]
                #density[:,0,:,:,1] = density_temp[:,0,:,:,-2]

                U[:,1,:,:,-1] = U_temp[:,1,:,:,1]
                #U[:,1,:,:,2] = U_temp[:,1,:,:,-3]
                U[:,0,:,:,-1] = -U_temp[:,0,:,:,1]
                #U[:,1,:,:,-2] = U_temp[:,1,:,:,2]
                #U[:,0,:,:,-2] = -U_temp[:,0,:,:,2]
                #density[:,:,:,:,-2] = density_temp[:,:,:,:,1]
                #density[:,:,:,:,-3] = density_temp[:,:,:,:,2]
                #density[:,:,:,:,1] = density_temp[:,:,:,:,-2]
                #density[:,:,:,:,2] = density_temp[:,:,:,:,-3]


    end_Pres = default_timer()
    time_Pressure=(end_Pres - start_Pres)
    print("time Pressure ", time_Pressure)


    #elif stick:
    #    fluid.setWallBcsStick(U, flags, flags_stick)

    # 19/09/19 Ekhi. We add the BC Ux = 0 on the right wall for the half domain!
    #wall_zero = torch.zeros_like(U[:,0,:,:,-1])
    #U[:,0,:,:,-1] = wall_zero
    #U[:,0,:,:,-2] = wall_zero

    div_after  = fluid.velocityDivergence(U, flags)

    #Time Vec Saving
    Time_vec[it] = time
    filename = folder + '/Time'
    #np.save(filename, Time_vec)

    #U = fluid.setWallBcs(U, flags)
    #setConstVals(batch_dict, p, U, flags, density)

    #if it < 200:
    #   Threshold = 3.e-4
    #else:
    #   Threshold = 8.e-4
    Threshold = Threshold_Div

    div_after  = fluid.velocityDivergence(U, flags)

    Max_Div[it] = (abs(div_after).max()).item()

    print(" Div After Advection : ===============> ", Advected_Div)
    print(" Div Max : ===========================> ", Max_Div[it])

    if ( Outside_Ja == True):

        if abs(div_after).max() > Threshold:
     
            print( " Treshold surpassed ========> ")
            
            #Jacobi_switch[it]=1
            Counter = 0

            while abs(div_after).max() > Threshold and Counter< 10000:

                Jacobi_switch[it]+=1

                div = fluid.velocityDivergence(U, flags)
                #print("Beginning ",(abs(div).max()).item())
                is3D = (U.size(2) > 1)
                pTol = mconf['pTol']
                maxIter = 1

                p, residual = fluid.solveLinearSystemJacobi( \
                        flags=flags, div=div, is_3d=is3D, p_tol=pTol, \
                        max_iter=maxIter)

                fluid.velocityUpdate(pressure=p, U=U, flags=flags)

                if 'periodic-x' in mconf and 'periodic-y' in mconf:
                    U_temp = U.clone()
                #U = fluid.setWallBcs(U, flags)
                if 'periodic-x' in mconf and 'periodic-y' in mconf:
                    if mconf['periodic-x']:
                        U[:,1,:,:,1] = U_temp[:,1,:,:,U.size(4)-1]
                    if mconf['periodic-y']:
                        print("hello")
                        U[:,1,:,:,-1] = U_temp[:,1,:,:,1]
                        U[:,0,:,:,-1] = -U_temp[:,0,:,:,1]

                        #U[:,1,:,:,1] = U_temp[:,1,:,:,-2]
                        #density[:,:,:,:,1] = density_temp[:,:,:,:,-2]
                if stick:
                    fluid.setWallBcsStick(U, flags, flags_stick)   

                U = fluid.setWallBcs(U, flags)
                setConstVals(batch_dict, p, U, flags, density)
                U = batch_dict['U']

                Counter +=1
                div_after  = fluid.velocityDivergence(U, flags)
                #print("Ending ",(abs(div_after).max()).item())

                #print(" ----------------------------------------")
                #print("Counter :", Counter)
                #print("Div Max :", (abs(div_after).max()).item())
                 
    #print("Counter :", Counter)

    #end_Pres = default_timer()
    #time_Pressure=(end_Pres - start_Pres)
    #print("time Pressure ", time_Pressure)

    Time_Pres[it] = time_Pressure

    div_final  = fluid.velocityDivergence(U, flags)
    Max_Div_All[it] = (abs(div_final).max()).item()

    print(" ----------------------------------------")
    print("Div FINAL OF ITERATIVE PROCES :", (abs(div_final).max()).item())

    # 19/09/19 Ekhi. We add the BC Ux = 0 on the right wall for the half domain!
    #wall_zero = torch.zeros_like(U[:,0,:,:,-1])
    #U[:,0,:,:,-1] = wall_zero
    #U[:,0,:,:,-2] = wall_zero

    #U = fluid.setWallBcs(U, flags)

    if sim_method != 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:

                U[:,1,:,:,1] = U_temp[:,1,:,:,U.size(4)-1]
                density[:,0,:,:,1] = density_temp[:,0,:,:,U.size(4)-1]

            if mconf['periodic-y']:

                print(" Periodic y !")
#                print("U copying ", U_temp[0,1,0,175:200,-2])
#                print("U modifying before ", U[0,1,0,175:200,1])

                U[:,1,:,:,-1] = U_temp[:,1,:,:,1]
                U[:,0,:,:,-1] = -U_temp[:,0,:,:,1]
                #U[:,1,:,:,-2] = U_temp[:,1,:,:,2]
                #U[:,0,:,:,-2] = -U_temp[:,0,:,:,2]

    if sim_method == 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
 
                U[:,1,:,:,1] = U_temp[:,1,:,:,U.size(4)-1]
                density[:,0,:,:,1] = density_temp[:,0,:,:,U.size(4)-1]
 
            if mconf['periodic-y']:
 
                print(" Periodic y 2 !")
#                print("U copying ", U_temp[0,1,0,175:200,-2])
#                print("U modifying before ", U[0,1,0,175:200,1])
 
                U[:,1,:,:,-1] = U_temp[:,1,:,:,1]
                U[:,0,:,:,-1] = -U_temp[:,0,:,:,1]


#    # Final Check U 
#
#    print("U x 0 ", U[0,0,0,175:200,0])
#    print("U x 1 ", U[0,0,0,175:200,1])
#    print("U x 2 ", U[0,0,0,175:200,2])
#    print("U x -3 ", U[0,0,0,175:200,-3])
#    print("U x -2 ", U[0,0,0,175:200,-2])
#    print("U x -1 ", U[0,0,0,175:200,-1])
#
#    print("U y 0 ", U[0,1,0,175:200,0])
#    print("U y 1 ", U[0,1,0,175:200,1])
#    print("U y 2 ", U[0,1,0,175:200,2])
#    print("U y -3 ", U[0,1,0,175:200,-3])
#    print("U y -2 ", U[0,1,0,175:200,-2])
#    print("U y -1 ", U[0,1,0,175:200,-1])

    batch_dict['Ustar'] = Ustar
    batch_dict['U'] = U
    batch_dict['density'] = density
    batch_dict['p'] = p
    batch_dict['Div_in']= div_input
