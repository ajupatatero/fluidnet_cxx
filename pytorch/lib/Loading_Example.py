import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import glob
from load_manta_data import loadMantaFile

#cur_scene = idx // self.step_per_scene
#cur_timestep = (idx % (self.step_per_scene)) * self.save_dt
#data_file = glob.os.path.join(self.base_dir, '{0:06d}'.format(cur_scene), \
#			      '{0:06d}.bin'.format(cur_timestep))
#data_div_file = glob.os.path.join(self.base_dir, '{0:06d}'.format(cur_scene), \
#			      '{0:06d}_divergent.bin'.format(cur_timestep))
#assert glob.os.path.isfile(data_file), 'Data file ' + data_file +  ' does not exists'
#assert glob.os.path.isfile(data_div_file), 'Data file does not exists'
#p, U, flags, density, is3D = self.pr_loader(data_file)
#pDiv, UDiv, flagsDiv, densityDiv, is3DDiv = self.pr_loader(data_div_file)

#assert is3D == is3DDiv, '3D flag is inconsistent!'
#assert torch.equal(flags, flagsDiv), 'Flags are not equal for idx ' + str(idx)


p, U, flags, density, is3D = loadMantaFile('/scratch/daep/e.ajuria/FluidNet/OriginalFluid/FluidNet/data/datasets/output_current_model_sphere/tr/000064/000064.bin')
p1, U1, flags1, density1, is3D1 = loadMantaFile('/scratch/daep/e.ajuria/FluidNet/OriginalFluid/FluidNet/data/datasets/output_current_model_sphere/tr/000016/000016.bin')
p2, U2, flags2, density2, is3D2 = loadMantaFile('/scratch/daep/e.ajuria/FluidNet/OriginalFluid/FluidNet/data/datasets/output_current_model_sphere/tr/000000/000000.bin')

torch_file = torch.load('/scratch/daep/e.ajuria/FluidNet/OriginalFluid/FluidNet/data/datasets/output_current_model_sphere/tr/000064/000064_pyTen.pt')
torch_file1 = torch.load('/scratch/daep/e.ajuria/FluidNet/OriginalFluid/FluidNet/data/datasets/output_current_model_sphere/tr/000016/000016_pyTen.pt')
torch_file2 = torch.load('/scratch/daep/e.ajuria/FluidNet/OriginalFluid/FluidNet/data/datasets/output_current_model_sphere/tr/000000/000000_pyTen.pt')

data = torch_file[0,0:5]
#target = torch_file[0,5:9]
#p = target[0].unsqueeze(0).unsqueeze(0)
#U = target[1:3].unsqueeze(0)
#density = target[3].unsqueeze(0).unsqueeze(0)
flags_p = data[3].unsqueeze(0).unsqueeze(0)

data1 = torch_file1[0,0:5]
flags_p1 = data1[3].unsqueeze(0).unsqueeze(0)

data2 = torch_file2[0,0:5]
flags_p2 = data2[3].unsqueeze(0).unsqueeze(0)

print("")
print("INITIAL BIN FILE")
print("")
F0 = flags[0,0,0]-1
F1 = flags1[0,0,0]-1
F2 = flags2[0,0,0]-1

print("Sum of Flags",torch.sum(F0), F0)
print("Sum of Flags",torch.sum(F1), F1)
print("Sum of Flags",torch.sum(F2), F2)

print("")
print("Preprocessed PT FILE")
print("")

G0 = flags_p[0,0,0]-1
G1 = flags_p1[0,0,0]-1
G2 = flags_p2[0,0,0]-1

print("Sum of Flags",torch.sum(G0), G0)
print("Sum of Flags",torch.sum(G1), G1)
print("Sum of Flags",torch.sum(G2), G2)

#data = torch.cat([pDiv, UDiv, flagsDiv, densityDiv, p, U, density], 1)
#data1 = torch.cat([pDiv1, UDiv1, flagsDiv1, densityDiv1, p1, U1, density1], 1)
#data2 = torch.cat([pDiv2, UDiv2, flagsDiv2, densityDiv2, p2, U2, density2], 1)

#torch.save(data, save_file)

