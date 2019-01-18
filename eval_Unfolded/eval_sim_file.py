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
from network.UnfoldedNet3dC import UnfoldedNet3dC
#from tools.mat2gif import mat2gif

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
#Directory of input data and its size
data_dir='../Data/for_eval/resources/sim_patch_128x128x50.npz'
m,n,time=128,128,50 #size of data
#Save gif
saveGif=False
save_gif_dir='../Data/for_eval/gifs_Unfolded/sim_file.gif'
cmap='hot'
note='abs'
#Save matrix
saveMat=True
save_mat_dir='../Data/for_eval/results_Unfolded/Unf_10epochs_sim_only_applied_2_sim.mat'
"""========================================================================="""

#Converter
form_in={'pre':'concat','shape':[m,n,time*2]}
form_out={'pre':'concat','shape':[m,n,time]}
convert=Converter()

#Load the model
#device='cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
if mfile[-3:]=='pkl':
    model=UnfoldedNet3dC(params_net)
    state_dict=torch.load(mfile,map_location=device)
    model.load_state_dict(state_dict)
else:
    model=torch.load(mfile)

#Processing
with torch.no_grad():
    data=np.load(data_dir)
    Sum,Bubbles,Tissue=data['arr_0'],data['arr_1'],data['arr_2']
    [Sum]=convert.np2torch([Sum],[form_in])
    predL,predS=model(Sum)
[predL,predS,Sum]=convert.torch2np([predL,predS,Sum],[form_out,form_out,form_out])

#Display
plt.close('all')
player=Player()
player.play([Sum,Bubbles,Tissue,None,predS,predL],note=note,cmap=cmap,
            tit=['Input','Ground Truth S','Ground Truth L',
                 None,'Prediction S','Prediction L'])

#Save gif
if saveGif:
    mat2gif([Sum,Bubbles,Tissue,None,predS,predL],save_gif_dir,
            note=note,cmap=cmap,
            tit=['Input','Ground Truth S','Ground Truth L',
                 None,'Prediction S','Prediction L'])
#Save matrix
if saveMat:
    savemat(save_mat_dir,{'D':Sum,'S':Bubbles,'L':Tissue,
                          'Sp':predS,'Lp':predL})