# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:15:17 2018

@author: Oren Solomon
"""
""" Modules """
"""========================================================================="""
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

import sys
sys.path.append('../')

from scipy.io import loadmat,savemat
from classes.Player import Player
#from network.UnfoldedNet3dC import UnfoldedNet3dC,to_var
from network.UnfoldedNet3dC_MSElayer import UnfoldedNet3dC,to_var
#from tools.mat2gif import mat2gif
from tools.eval_Unfolded import process_patch

from network.ResNet3dC import ResNet3dC

#import matplotlib 
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

#from DataSet_Unfolded import ImageDataset
from classes.Dataset import Converter

import time
import datetime
import pickle

"""Settings"""
"""=========================================================================""" 
#Model file
# Sim
mfile = 'Results/Res3dC_sim50epoch_sim_Res3dC_Model_Tr2400_epoch50_lr2.00e-03.pkl'


"""Network Settings: Remember to change the parameters when you change model!"""
# For data loading
prefix         = 'sim' #invivo,sim_pm,sim
shape_dset     = (32,32,40)

gpu=True #if gpu=True, the ResNet will use more parameters
#Directory of input data and its size

m,n,time=32,32,20 #size of data

#directory of datasets
d_invivo = '../../../Data/Invivo/' #
d_simpm  = '../../Ultrasound_805/data/Sim_PM/'
d_sim    = '../../../Data/Sim/'

#Saving Mat
saveMat=True
save_mat_dir='../../../Data/for_eval/test/MSE_test1__ResNet_sim_50epochs.mat'

"""========================================================================="""
"""                         Load pretrained model                           """
"""========================================================================="""
print('Loading Network...', end = "")
#Converter
form_in={'pre':'concat','shape':[-1,1,m,n,time*2]}
form_out={'pre':'concat','shape':[m,n,time*2]}
convert=Converter()

#Load the model
#device='cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
if mfile[-3:]=='pkl':
    model=ResNet3dC(gpu)
    state_dict=torch.load(mfile,map_location=device)
    model.load_state_dict(state_dict)
else:
    model=torch.load(mfile)
print('done.')

"""========================================================================="""
"""                               Load data                                 """
"""========================================================================="""
print('Loading phase...')
print('----------------')

#Dataset, converter and player
data_dir={'invivo':d_invivo,'sim_pm':d_simpm,'sim':d_sim}[prefix]
conter=Converter()
player=Player()
formshow={'pre':'concat','shape':(32,32,20)}
formlist=[]
for i in range(6):
    formlist.append(formshow)
minloss=np.inf

#validation - MSE calculations are performed on the validation set
val_dataset = ImageDataset(round(ValInstances),shape_dset, 
                         train=1,data_dir=data_dir)
val_loader  = data.DataLoader(val_dataset,batch_size=ValBatchSize,shuffle=False)
print('Finished loading.\n')

"""========================================================================="""
"""          Compute the MSE as a function of increasing layers             """
"""========================================================================="""
floss = nn.MSELoss() 

# Reset loss 
loss_val_mean_S = np.zeros(1)    

# Load all patches and apply them to the network
kk = 1;
for _,(Lv,Sv,Dv) in enumerate(val_loader):
    #t_start = time.time()
    print('Calculating MSE for batch %d / %d ... ' %(kk, ValInstances/ValBatchSize), end="")
    for jj in range(ValBatchSize):
        inputsv    = to_var(Dv[jj],CalInGPU)   # "jj"th picture
        targets_Sv = to_var(Sv[jj],CalInGPU)     
        
        # Forward
        with torch.no_grad():
            [Sum]        = convert.np2torch([inputsv.data.numpy()],[form_in])
            pred         = model(Sum) 
            [predmv,Sum] = convert.torch2np([pred,Sum],[form_out,form_out])            
            Out          = to_var(torch.from_numpy(predmv),CalInGPU)            
            
        # Current loss
        loss_val_S       = floss(pred, targets_Sv)
        loss_val_mean_S += loss_val_S.item()
                        
    # Batch counter
    kk += 1        
    #t_end = time.time()
    #print('time = %f [sec]' %(t_end - t_start))      
      
# Divide by total number of patches
loss_val_mean_S    = loss_val_mean_S/ValInstances

# Save
savemat(save_mat_dir,{'loss_val_mean_S':loss_val_mean_S})



   

            
    
    
    
    
    
    
    
    
    
    
    