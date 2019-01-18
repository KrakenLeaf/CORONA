# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:35:50 2018

@author: Yi Zhang
"""

import sys
import torch
import matplotlib.pyplot as plt
sys.path.append('../')
from classes.Player import Player
from classes.Dataset import Converter
from network.ResNet3dC import ResNet3dC
from SimPlatform.Parameters import params_default
from SimPlatform.Simulator import Simulator
from tools.mat2gif import mat2gif

"""Settings"""
"""========================================================================="""
#Model file
#mfile='../saved/Res3dC_invivo10epoch.pkl'
mfile='../saved/Res3dC_sim10epoch_invivo10epoch.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
gpu=False #if gpu=True, the ResNet will use more parameters
#Settings for generating data
m,n,time=128,128,50
#Saving gif
saveGif=True
save_gif_dir='../../../Data/for_eval/gifs_Res3dC/sim_gen.gif'
cmap='hot'
note='abs'
"""========================================================================="""

#Converter
form_in={'pre':'concat','shape':[-1,1,m,n,time*2]}
form_out={'pre':'concat','shape':[m,n,time]}
convert=Converter()
#Generating data
params=params_default
params['shape']=(m,n)
simtor=Simulator(params)
Sum,Bubbles,Tissue=simtor.generate(T=time)
player=Player()

#device='cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
if mfile[-3:]=='pkl':
    model=ResNet3dC(gpu)
    state_dict=torch.load(mfile,map_location=device)
    #state_dict=delitems()
    model.load_state_dict(state_dict)
else:
    model=torch.load(mfile)

with torch.no_grad():
    [Sum]=convert.np2torch([Sum],[form_in])
    pred=model(Sum)
[predmv,Sum]=convert.torch2np([pred,Sum],[form_out,form_out])

plt.close('all')
player.play([Sum,predmv,Bubbles],note=note,cmap=cmap)

if saveGif:
    mat2gif([Sum,predmv,Bubbles],save_gif_dir,
            note=note,cmap=cmap,tit=['Input','Prediction','Ground Truth'])
