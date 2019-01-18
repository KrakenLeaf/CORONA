# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:15:17 2018

@author: Yi Zhang
"""

import torch
import numpy as np

import sys
sys.path.append('../')
from scipy.io import loadmat,savemat
from classes.Player import Player
from network.ResNet3dC import ResNet3dC
#from tools.mat2gif import mat2gif
from tools.eval_Res3dC import process_patch
  
"""Settings"""
"""=========================================================================""" 
#Model file

# 10 epochs on sim
mfile = '../Res3dC/Results/GoodResults/Res_sim10epochs/Res3dC_sim10_sim_Res3dC_Model_Tr2400_epoch10_lr2.00e-03.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
gpu=True #if gpu=True, the ResNet will use more parameters

#The area to be processed
arange=[50,120,50,200,0,100]

#directory of input data, ground truth of S
data_dir='../Data/for_eval/resources/D_test2.npz'
Sfile='../Data/for_eval/resources/S_fista_test2.mat'
#Saving GIF
saveGif=False
save_gif_dir='../Data/for_eval/gifs_Res3dC/invivo_patch.gif'
note='db'
cmap='hot'
#Saving Mat
saveMat=True
save_mat_dir='../Data/for_eval/results_Res3dC/siminvivo_50_20epochs_p3.mat'
"""========================================================================="""

#Load model
#device='cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
if mfile[-3:]=='pkl':
    net=ResNet3dC(gpu)
    state_dict=torch.load(mfile,map_location=device)
    #state_dict=delitems(state_dict)
    net.load_state_dict(state_dict)    
else:
    net=torch.load(mfile)

#Processing
D,Sp=process_patch(net,data_dir,arange)
brow,erow,bcol,ecol,bt,et=arange
if Sfile[-3:]=='mat':
    S=loadmat(Sfile)['vid'][brow:erow,bcol:ecol,bt:et]
else:
    S=np.load(Sfile)['arr_0'][brow:erow,bcol:ecol,bt:et]

#Display
player=Player()
player.play([D,Sp,S],cmap=cmap,note=note,tit=['Input','Prediction','Fista'])        
    
#save as gif
if saveGif:
    mat2gif([D,Sp,S],save_gif_dir,cmap=cmap,note=note,
            tit=['Input','Prediction','Fista'],play=False)     
#save the matrices  
if saveMat:
    savemat(save_mat_dir,{'D':D,'S':S,'Sp':Sp})
    
    