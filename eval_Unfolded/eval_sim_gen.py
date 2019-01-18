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
from network.UnfoldedNet3dC import UnfoldedNet3dC
from SimPlatform.Parameters import params_default
from SimPlatform.Simulator import Simulator
from tools.mat2gif import mat2gif

"""Settings"""
"""========================================================================="""
#Model file
#mfile='../saved/Unf_coefS1d8_coefL0d4_sim10epoch_2400Tr.pkl'
#mfile='../saved/coefL0d4_coefS1d8_siminvivo10epoch.pkl'
mfile='../saved/Unf_coefS1d8_coefL0d4_sim10epoch_2400Tr.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
params_net={'layers':10,
            'kernel':[(5,1)]*3+[(3,1)]*7,
            'coef_L':0.4,
            'coef_S':1.8,
            'CalInGPU':True}
#Settings for generating data
m,n,time=128,128,50
#Saving gif
saveGif=True
save_gif_dir='../../../Data/for_eval/gifs_Unfolded/sim_gen.gif'
cmap='hot'
note='abs'
"""========================================================================="""

#Converter
form_in={'pre':'concat','shape':[m,n,time*2]}
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
    model=UnfoldedNet3dC(params_net)
    state_dict=torch.load(mfile,map_location=device)
    #state_dict=delitems()
    model.load_state_dict(state_dict)
else:
    model=torch.load(mfile)

with torch.no_grad():
    [Sum]=convert.np2torch([Sum],[form_in])
    predL,predS=model(Sum)
[predL,predS,Sum]=convert.torch2np([predL,predS,Sum],[form_out,form_out,form_out])

plt.close('all')
player.play([Sum,Bubbles,Tissue,None,predS,predL],note=note,cmap=cmap,
            tit=['Input','Ground Truth S','Ground Truth L',
                 None,'Prediction S','Prediction L'])

if saveGif:
    mat2gif([Sum,Bubbles,Tissue,None,predS,predL],save_gif_dir,
            note=note,cmap=cmap,
            tit=['Input','Ground Truth S','Ground Truth L',
                 None,'Prediction S','Prediction L'])
