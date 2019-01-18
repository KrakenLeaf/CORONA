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

#import matplotlib 
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

from DataSet_Unfolded import ImageDataset
from classes.Dataset import Converter

import time
import datetime
import pickle

"""Settings"""
"""=========================================================================""" 
#Model file
# Sim
#mfile = '../saved/Unf_coefS1d8_coefL0d4_sim10epoch_2400Tr.pkl' # 10 epochs, trained on SIM
#mfile = '../Unfolded/Results/Unf_10lay_sim/Unfolded_10lay_sim50epoch_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'

# Rat1 -> Rat2: 50 sim / 20 invivo
#mfile = '../saved/coefL0d4_coefS1d8_siminvivo10epoch.pkl' # 10 epochs on SIM, 10 epochs on In-vivo
#mfile = '../Unfolded/Results/GoodResults/OLD/Unfolded_sim_in_vivo_20epochs/Oren_after_sim_invivo_Unfolded_Model_al0.50_Tr2400_epoch20_lr2.00e-03.pkl'

# Rat2-> Rat1 (converse): 50 sim / 20 invivo
#mfile = '../saved/Unf_converse_invivo_Unfolded_Model_al0.50_Tr2400_epoch10_lr2.00e-03.pkl'
#mfile = '../Unfolded/Results/GoodResults/OLD/Unfolded_sim_in_vivo_20epochs_converse/Oren_after_sim_converse_invivo_Unfolded_Model_al0.50_Tr2400_epoch20_lr2.00e-03.pkl'

# Unfolded trained only on sim for 50 epochs
# 10 layers
#mfile = '../Unfolded/Results/Unf_10lay_sim/Unfolded_10lay_sim50epoch_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'
# 9 layers
#mfile = '../Unfolded/Results/Unf_9lay_sim/sim_9lay_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'
# 8 layers
#mfile = '../Unfolded/Results/Unf_8lay_sim/Unfolded_8lay_sim50epoch_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'
# 6 layers
#mfile = '../Unfolded/Results/Unf_6lay_sim/sim_6lay_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'
# 5 layers
#mfile = '../Unfolded/Results/Unf_5lay_sim/sim_5lay_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'
# 4 layers
#mfile = '../Unfolded/Results/Unf_4lay_sim/sim_4lay_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'
# 3 layers
#mfile = '../Unfolded/Results/Unf_3lay_sim/sim_3lay_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'
# 2 layers
#mfile = '../Unfolded/Results/Unf_2lay_sim/sim_2lay_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'
# 1 layers
mfile = '../Unfolded/Results/Unf_1lay_sim/sim_1lay_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'



"""Network Settings: Remember to change the parameters when you change model!"""
params_net={'layers':1,
            'kernel':[(5,1)]*3+[(3,1)]*7,
            'coef_L':0.4,
            'coef_S':1.8,
            'CalInGPU':False}

#The area to be processed
#arange=[50,120,50,200,0,100] # Vertical | Horizontal | Depth

# For data loading
prefix         = 'sim' #invivo,sim_pm,sim
shape_dset     = (32,32,40)

TrainInstances = 2400 # Size of training dataset
ValInstances   = 800
BatchSize      = 40
ValBatchSize   = 40
ALPHA          = 0.5
num_epochs     = 10; 50; #200num_epochs     = 5; #200
frame=10
#directory of datasets
d_invivo = '../../../Data/Invivo/' #
d_simpm  = '../../Ultrasound_805/data/Sim_PM/'
d_sim    = '../../../Data/Sim/'

#Whether to calculate in GPU
CalInGPU=params_net['CalInGPU']


#Saving Mat
saveMat=True
save_mat_dir='../../../Data/for_eval/test/MSE_test1__4lay_sim_50epochs.mat'

"""========================================================================="""
"""                         Load pretrained model                           """
"""========================================================================="""
print('Loading Network...', end = "")
#device='cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
if mfile[-3:]=='pkl':
    net=UnfoldedNet3dC(params_net)
    state_dict=torch.load(mfile,map_location=device)
    #state_dict=delitems(state_dict)
    net.load_state_dict(state_dict)  
else:
    net=torch.load(mfile)
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
## Define a class of trancated unfolded network
#class TrancUnf(nn.Module):
#    def __init__(self, NumLayers):
#        super(TrancUnf, self).__init__()
#        features = list(net.features)[:1]
#        self.features = nn.Modulelist(features).eval()
#        self.Nlayers  = NumLayers
#        
#    def forward(self, x):
#        results = []
#        for ii, model in enumerate(self.features):
#            x = model(x)
 
# Loss function
floss = nn.MSELoss() 

# For each sequential layer in the network
#for ii in range(params_net['layers']):
#print('Calculating MSE for layers 0:%d...' %(ii), end="")
#Tnet = nn.Sequential(net.sig, net.relu, net.filter[:ii+1])  
#Tnet = net.filter[:ii+1]

#Arrays for recording datas
outputs_S = to_var(torch.zeros([params_net['layers'],32,32,40]),CalInGPU)
outputs_L = to_var(torch.zeros([params_net['layers'],32,32,40]),CalInGPU)
lossmean_val_vec = np.zeros((num_epochs,)) 

# Reset loss 
loss_val_mean_S = np.zeros(params_net['layers'])    
loss_val_mean_L = np.zeros(params_net['layers']) 
loss_val_mean_comb = np.zeros(params_net['layers']) 

# Load all patches and apply them to the network
kk = 1;
for _,(Lv,Sv,Dv) in enumerate(val_loader):
    t_start = time.time()
    print('Calculating MSE for batch %d / %d ... ' %(kk, ValInstances/ValBatchSize), end="")
    for jj in range(ValBatchSize):
        inputsv    = to_var(Dv[jj],CalInGPU)   # "jj"th picture
        targets_Lv = to_var(Lv[jj],CalInGPU)
        targets_Sv = to_var(Sv[jj],CalInGPU)
        
        # Forward
        outputs_Lv, outputs_Sv = net(inputsv) 
        
        # Current loss
        loss_val_S = np.zeros(params_net['layers']) 
        loss_val_L = np.zeros(params_net['layers'])
        for ii in range(params_net['layers']):
            # S
            loss_val_S[ii] = floss(outputs_Sv[ii],targets_Sv)
            loss_val_mean_S[ii] += loss_val_S[ii].item()
            
            # L
            loss_val_L[ii] = floss(outputs_Lv[ii],targets_Lv)
            loss_val_mean_L[ii] += loss_val_L[ii].item()
            
            # SUM
            loss_val_mean_comb[ii] = ALPHA*loss_val_mean_L[ii] + (1-ALPHA)*loss_val_mean_S[ii]           
            
    # Batch counter
    kk += 1        
    t_end = time.time()
    print('time = %f [sec]' %(t_end - t_start))      
      
# Divide by total number of patches
loss_val_mean_S    = loss_val_mean_S/ValInstances
loss_val_mean_L    = loss_val_mean_L/ValInstances
loss_val_mean_comb = loss_val_mean_comb/ValInstances 

# Save
savemat(save_mat_dir,{'loss_val_mean_S':loss_val_mean_S, 'loss_val_mean_L':loss_val_mean_L, 'loss_val_mean_comb':loss_val_mean_comb})



   

            
    
    
    
    
    
    
    
    
    
    
    