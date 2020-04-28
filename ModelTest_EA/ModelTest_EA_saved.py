import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer

from lib import fluid, MultiScaleNet, MS_Analysis
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

class FluidNet(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, mconf, it, folder, dropout=False):
        super(FluidNet, self).__init__()
   
        self.dropout = dropout
        self.mconf = mconf
        self.inDims = mconf['inputDim']
        self.is3D = mconf['is3D']
        self.it = it
        self.folder =folder

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
        #self.multiScale = MultiScaleNet(self.inDims)
        self.multiScale = MS_Analysis(self.inDims)

      
    def forward(self, input_, it,folder):

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

        UDiv_1 = input_[:,1:3].contiguous()

        #Print Before U
        #Uinter_cpu = UDiv_1.cpu()
        #filename_inter = folder + '/U1_NN_Intermediate{0:05}'.format(it)
        #np.save(filename_inter,Uinter_cpu)

        # Flags are always loaded
        if self.is3D:
            flags = input_[:,4].unsqueeze(1)
        else:
            flags = input_[:,3].unsqueeze(1).contiguous()


        if (self.mconf['inputChannels']['pDiv'] or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'pDiv')):
            pDiv = input_[:,0].unsqueeze(1).contiguous()

        if (self.mconf['inputChannels']['UDiv'] or self.mconf['inputChannels']['div'] \
            or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'UDiv')):
            if self.is3D:
                UDiv = input_[:,1:4].contiguous()
            else:
                UDiv = input_[:,1:3].contiguous()

            # Apply setWallBcs to zero out obstacles velocities on the boundary

            ##################################################################
            ###########                  HERE                  ###############
            ##################################################################            

            #UDiv = fluid.setWallBcs(UDiv, flags)

            if self.mconf['inputChannels']['div']:
                div = fluid.velocityDivergence(UDiv, flags)



        #Print Before U
        #Uinter1_cpu = UDiv.cpu()
        #filename_inter1 = folder + '/U1_NN_Intermediate1_{0:05}'.format(it)
        #np.save(filename_inter1,Uinter1_cpu)


        # Apply scale to input
        if self.mconf['normalizeInput']:
            if self.mconf['normalizeInputChan'] == 'UDiv':
                s = self.scale(UDiv)
            elif self.mconf['normalizeInputChan'] == 'pDiv':
                s = self.scale(pDiv)
            elif self.mconf['normalizeInputChan'] == 'div':
                s = self.scale(div)
            else:
                raise Exception('Incorrect normalize input channel.')

            if pDiv is not None:
                pDiv = torch.div(pDiv, s)
            if UDiv is not None:
                UDiv = torch.div(UDiv, s)
            if div is not None:
                div = torch.div(div, s)

        x = torch.FloatTensor(input_.size(0), \
                              self.inDims,    \
                              input_.size(2), \
                              input_.size(3), \
                              input_.size(4)).type_as(input_)

        chan = 0
        if self.mconf['inputChannels']['pDiv']:
            x[:, chan] = pDiv[:,0]
            chan += 1
        elif self.mconf['inputChannels']['UDiv']:
            if self.is3D:
                x[:,chan:(chan+3)] = UDiv
                chan += 3
            else:
                x[:,chan:(chan+2)] = UDiv
                chan += 2
        elif self.mconf['inputChannels']['div']:
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

            set_hook = False            
            #p = self.multiScale(x)

            #MultiGrid Study little Modif
            #p_out = self.multiScale(x)
            #p= p_out[:,0,...].unsqueeze(1) 


            # MultiScale Study
            # We will not just output P fields in the output of layers, but rather
            # All the Feature Maps !
            # Each layer is saved independiently

            if set_hook:
                # Get the corresponding Modules for Analysing!!!
                scale_4= self.multiScale._modules.get('convN_4')
                scale_2= self.multiScale._modules.get('convN_2')
                scale_1= self.multiScale._modules.get('convN_1')

                inside_scale_4 = scale_4._modules.get('encode') 
                inside_scale_2 = scale_2._modules.get('encode')
                inside_scale_1 = scale_1._modules.get('encode')

                # Create a list to save the layer outputs
                output_analysis = {}

                # Definition of saving function

                def get_out(name):
                    def hook(model,input,output):
                        output_analysis[name] = output.data
                    return hook

                # Perform Forward Hooks!

                for i in range(len(inside_scale_4)):    
                    output_4 = inside_scale_4[i].register_forward_hook(get_out('s_4_l_{}'.format(i)))

                for i in range(len(inside_scale_2)):
                    output_2 = inside_scale_2[i].register_forward_hook(get_out('s_2_l_{}'.format(i)))

                for i in range(len(inside_scale_1)):
                    output_1 = inside_scale_1[i].register_forward_hook(get_out('s_1_l_{}'.format(i)))


            # OUTPUTTT

            # Uncomment Saving for Intermediate and network analysis
            #np.save(folder+'/Input_it_{}.npy'.format(it),x.cpu().data.numpy()[0,0,...])

            # Declare variables
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Start recording
            start_event.record()

            # Network it
            p_out = self.multiScale(x)

            # Finish recording
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded! 
            elapsed_time_ms = start_event.elapsed_time(end_event)

            p= p_out[:,0,...].unsqueeze(1)

            # Uncomment Saving for Intermediate and network analysis
            #np.save(folder+'/Big_Scale_it_{}.npy'.format(it),p_out.cpu().data.numpy()[0,0,...])
            #np.save(folder+'/Medium_Scale_it_{}.npy'.format(it),p_out.cpu().data.numpy()[0,1,...])
            #np.save(folder+'/Small_Scale_it_{}.npy'.format(it),p_out.cpu().data.numpy()[0,2,...])


            print_info = False

            if print_info == True:


                print('Printing Small Scale !')
                for i in range(len(inside_scale_4)):
                    print(output_analysis['s_4_l_{}'.format(i)].shape)

                print('Printing Medium Scale !')
                for i in range(len(inside_scale_2)):
                    print(output_analysis['s_2_l_{}'.format(i)].shape)

                print('Printing Big Scale !')
                for i in range(len(inside_scale_1)):
                    print(output_analysis['s_1_l_{}'.format(i)].shape)



            # Final Saving!!!!!!!

            #if it%100 ==5 and it > 21000:
                # Saving

                #folder = '/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/results_VK/MS_Analysis/VK_NN_Re_300_1200_2400'
                #folder = '/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/Results_MultiScale_Analysis/FeatureMaps/Test_01/'

                # Small Scale 1/4
                '''
                s_4_l_0 = output_analysis['s_4_l_0'].cpu().data.numpy()
                s_4_l_2 = output_analysis['s_4_l_2'].cpu().data.numpy()
                s_4_l_5 = output_analysis['s_4_l_5'].cpu().data.numpy()
                s_4_l_7 = output_analysis['s_4_l_7'].cpu().data.numpy()
                s_4_l_9 = output_analysis['s_4_l_9'].cpu().data.numpy()

                np.save(folder+'/It_{}_FM_s_4_l_0.npy'.format(it),s_4_l_0)
                np.save(folder+'/It_{}_FM_s_4_l_2.npy'.format(it),s_4_l_2)
                np.save(folder+'/It_{}_FM_s_4_l_5.npy'.format(it),s_4_l_5)
                np.save(folder+'/It_{}_FM_s_4_l_7.npy'.format(it),s_4_l_7)
                np.save(folder+'/It_{}_FM_s_4_l_9.npy'.format(it),s_4_l_9)

                # Small Scale 1/2

                s_2_l_0 = output_analysis['s_2_l_0'].cpu().data.numpy()
                s_2_l_2 = output_analysis['s_2_l_2'].cpu().data.numpy()
                s_2_l_5 = output_analysis['s_2_l_5'].cpu().data.numpy()
                s_2_l_8 = output_analysis['s_2_l_8'].cpu().data.numpy()
                s_2_l_11 = output_analysis['s_2_l_11'].cpu().data.numpy()
                s_2_l_13 = output_analysis['s_2_l_13'].cpu().data.numpy()
                s_2_l_15 = output_analysis['s_2_l_15'].cpu().data.numpy()

                np.save(folder+'/It_{}_FM_s_2_l_0.npy'.format(it),s_2_l_0)
                np.save(folder+'/It_{}_FM_s_2_l_2.npy'.format(it),s_2_l_2)
                np.save(folder+'/It_{}_FM_s_2_l_5.npy'.format(it),s_2_l_5)
                np.save(folder+'/It_{}_FM_s_2_l_8.npy'.format(it),s_2_l_8)
                np.save(folder+'/It_{}_FM_s_2_l_11.npy'.format(it),s_2_l_11)
                np.save(folder+'/It_{}_FM_s_2_l_13.npy'.format(it),s_2_l_13)
                np.save(folder+'/It_{}_FM_s_2_l_15.npy'.format(it),s_2_l_15)

                '''
                # Full Scale 

                #s_1_l_0 = output_analysis['s_1_l_0'].cpu().data.numpy()
                #s_1_l_2 = output_analysis['s_1_l_2'].cpu().data.numpy()
                #s_1_l_5 = output_analysis['s_1_l_5'].cpu().data.numpy()
                #s_1_l_7 = output_analysis['s_1_l_7'].cpu().data.numpy()
                #s_1_l_8 = output_analysis['s_1_l_8'].cpu().data.numpy()
                #s_1_l_11 = output_analysis['s_1_l_11'].cpu().data.numpy()
                #s_1_l_13 = output_analysis['s_1_l_13'].cpu().data.numpy()
                #s_1_l_15 = output_analysis['s_1_l_15'].cpu().data.numpy()

                #np.save(folder+'/It_{}_FM_s_1_l_0.npy'.format(it),s_1_l_0)
                #np.save(folder+'/It_{}_FM_s_1_l_2.npy'.format(it),s_1_l_2)
                #np.save(folder+'/It_{}_FM_s_1_l_5.npy'.format(it),s_1_l_5)
                #np.save(folder+'/It_{}_FM_s_1_l_7.npy'.format(it),s_1_l_7)
                #np.save(folder+'/It_{}_FM_s_1_l_8.npy'.format(it),s_1_l_8)
                #np.save(folder+'/It_{}_FM_s_1_l_11.npy'.format(it),s_1_l_11)
                #np.save(folder+'/It_{}_FM_s_1_l_13.npy'.format(it),s_1_l_13)
                #np.save(folder+'/It_{}_FM_s_1_l_15.npy'.format(it),s_1_l_15)

            if set_hook:
                output_4.remove()
                output_2.remove()
                output_1.remove()

                output_analysis.clear()
                           

            #print("p shape ", p.shape)

            # Print and save different P fields!
            #folder = '/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/Results_MultiScale_Analysis'
            #folder='/scratch/daep/e.ajuria/FluidNet/Fluid_EA/fluidnet_cxx/Multigird_test'
            if it % 100 ==5:

                fig,axes = plt.subplots(figsize=(45,15),nrows = 1, ncols =3 )
                fig.subplots_adjust(hspace=0.4, wspace=0.4)

                fig.suptitle(' P -- Iteration : {} '.format(it), fontsize = 45)

                tt=axes[0].imshow(p_out.cpu().data.numpy()[0,0,...], origin='lower')
                axes[0].set_title('Big Scale', fontsize = 30)
                axes[0].axis('off')
                fig.colorbar(tt, ax =axes[0])

                ll=axes[1].imshow(p_out.cpu().data.numpy()[0,1,...], origin='lower')
                axes[1].set_title('Intermediate Scale', fontsize = 30)
                axes[1].axis('off')
                fig.colorbar(ll, ax =axes[1])

                pp=axes[2].imshow(p_out.cpu().data.numpy()[0,2,...],origin='lower')
                axes[2].set_title('Small Scale', fontsize = 30)
                axes[2].axis('off')
                fig.colorbar(pp, ax =axes[2])

                savefile = folder + '/Different_Scale_P_{}_it.png'.format(it)
                plt.savefig(savefile)
                plt.close()

                fig,axes = plt.subplots(figsize=(15,15) )
                fig.subplots_adjust(hspace=0.4, wspace=0.4)

                fig.suptitle(' P -- Big Iteration : {} '.format(it), fontsize = 45)

                ll=axes.imshow(p_out.cpu().data.numpy()[0,0,...], origin='lower')
                axes.set_title('Big Scale', fontsize = 30)
                axes.axis('off')
                fig.colorbar(ll, ax =axes)

                savefile = folder + '/Big_Scale_P_{}_it.png'.format(it)
                plt.savefig(savefile)
                plt.close()


                fig,axes = plt.subplots(figsize=(15,15) )
                fig.subplots_adjust(hspace=0.4, wspace=0.4)

                fig.suptitle(' P -- Middle Iteration : {} '.format(it), fontsize = 45)

                ll=axes.imshow(p_out.cpu().data.numpy()[0,1,...], origin='lower')
                axes.set_title('Intermediate Scale', fontsize = 30)
                axes.axis('off')
                fig.colorbar(ll, ax =axes)

                savefile = folder + '/Medium_Scale_P_{}_it.png'.format(it)
                plt.savefig(savefile)
                plt.close()

                fig,axes = plt.subplots(figsize=(15,15) )
                fig.subplots_adjust(hspace=0.4, wspace=0.4)

                fig.suptitle(' P -- Small Iteration : {} '.format(it), fontsize = 45)

                ll=axes.imshow(p_out.cpu().data.numpy()[0,2,...], origin='lower')
                axes.set_title('Small Scale', fontsize = 30)
                axes.axis('off')
                fig.colorbar(ll, ax =axes)

                savefile = folder + '/Small_Scale_P_{}_it.png'.format(it)
                plt.savefig(savefile)
                plt.close()

                

            print("elapsed time: ", elapsed_time_ms)
    
            time= elapsed_time_ms


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
    
     
        # Correct U = UDiv - grad(p)
        # flags is the one with Manta's values, not occupancy in [0,1]
        fluid.velocityUpdate(pressure=p, U=UDiv, flags=flags)

      
        # We now UNDO the scale factor we applied on the input.
        if self.mconf['normalizeInput']:
            p = torch.mul(p,s)  # Applies p' = *= scale
            UDiv = torch.mul(UDiv,s)

        # Set BCs after velocity update.
        #UDiv = fluid.setWallBcs(UDiv, flags)



        return p, UDiv, time


