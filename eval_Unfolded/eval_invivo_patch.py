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
from network.UnfoldedNet3dC import UnfoldedNet3dC
#from tools.mat2gif import mat2gif
from tools.eval_Unfolded import process_patch
 
"""Settings"""
"""=========================================================================""" 
#Model file
# Unfoledd trained only on sim for 50 epochs
mfile = '../Unfolded/Results/Unf_10lay_sim/Unfolded_10lay_sim50epoch_sim_Unfolded_Model_al0.50_Tr2400_epoch50_lr2.00e-03.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
params_net={'layers':10,
            'kernel':[(5,1)]*3+[(3,1)]*7,
            'coef_L':0.4,
            'coef_S':1.8,
            'CalInGPU':False}
#The area to be processed
arange=[50,120,50,200,0,100] # Vertical | Horizontal | Depth
#arange=[80,112,100,132,0,50]

#directory of input data, ground truth of S, ground truth of L
# ---------------------------------------------------------------------------------
# Rat1 -> Rat2
data_dir='../../../Data/for_eval/resources/D_val.mat'
Sfile='../../../Data/for_eval/resources/S_fista_val.mat'
Lfile='../../../Data/for_eval/resources/L_fista_val.mat'

#Saving GIF
saveGif=False
save_gif_dir='../Data/for_eval/gifs_Unfolded/invivo_patch.gif'
note='db'
cmap='hot'
#Saving Mat
saveMat=True
save_mat_dir='../Data/for_eval/test/save_name.mat'
"""========================================================================="""

#Load model
#device='cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
if mfile[-3:]=='pkl':
    net=UnfoldedNet3dC(params_net)
    state_dict=torch.load(mfile,map_location=device)
    #state_dict=delitems(state_dict)
    net.load_state_dict(state_dict)  
else:
    net=torch.load(mfile)

#Processing
D,Sp,Lp=process_patch(net,data_dir,arange)
brow,erow,bcol,ecol,bt,et=arange
if Sfile[-3:]=='mat':
    S=loadmat(Sfile)['vid'][brow:erow,bcol:ecol,bt:et]
else:
    S=np.load(Sfile)['arr_0'][brow:erow,bcol:ecol,bt:et]    
    
if Lfile[-3:]=='mat':
    L=loadmat(Lfile)['vid'][brow:erow,bcol:ecol,bt:et]
else:
    L=np.load(Lfile)['arr_0'][brow:erow,bcol:ecol,bt:et]
    
#Display
player=Player()
player.play([D,S,L,None,Sp,Lp],cmap=cmap,note=note,
            tit=['Input','Fista S','Fista L',None,'Prediction S','Prediction L'])        
    
#save as gif
if saveGif:
    mat2gif([D,S,L,None,Sp,Lp],save_gif_dir,cmap=cmap,note=note,
            tit=['Input','Fista S','Fista L',None,'Prediction S','Prediction L'])      
#save the matrices  
if saveMat:
    savemat(save_mat_dir,{'D':D,'S':S,'L':L,'Sp':Sp,'Lp':Lp})
    
    