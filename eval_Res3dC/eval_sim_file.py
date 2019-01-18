# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 15:10:02 2018

@author: Yi Zhang
"""

import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat
sys.path.append('../')
from classes.Player import Player
from classes.Dataset import Converter
from network.ResNet3dC import ResNet3dC
#from tools.mat2gif import mat2gif

"""Settings"""
"""========================================================================="""
#Model file
# 10 epochs on sim
mfile = '../Res3dC/Results/GoodResults/Res_sim10epochs/Res3dC_sim10_sim_Res3dC_Model_Tr2400_epoch10_lr2.00e-03.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
gpu=True #if gpu=True, the ResNet will use more parameters
#Directory of input data and its size
data_dir='../../../Data/for_eval/resources/sim_patch_128x128x50.npz'
m,n,time=128,128,50 #size of data
#Save gif
saveGif=False
save_gif_dir='../Data/for_eval/gifs_Res3dC/sim_file.gif'
cmap='hot'
note='abs'
#Save matrix
saveMat=True
save_mat_dir='../Data/for_eval/results_Res3dC/ResNet_sim10epochs.mat'
"""========================================================================="""

#Converter
form_in={'pre':'concat','shape':[-1,1,m,n,time*2]}
form_out={'pre':'concat','shape':[m,n,time]}
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

#Processing
with torch.no_grad():
    data=np.load(data_dir)
    Sum,Bubbles=data['arr_0'],data['arr_1']
    [Sum]=convert.np2torch([Sum],[form_in])
    pred=model(Sum)
[predmv,Sum]=convert.torch2np([pred,Sum],[form_out,form_out])

#Display
plt.close('all')
player=Player()
player.play([Sum,predmv,Bubbles],note=note,cmap=cmap)

#Save gif
if saveGif:
    mat2gif([Sum,predmv,Bubbles],save_gif_dir,
            note=note,cmap=cmap,tit=['Input','Prediction','Ground Truth'])
#Save matrix
if saveMat:
    savemat(save_mat_dir,{'D':Sum,'S':Bubbles,'Sp':predmv})