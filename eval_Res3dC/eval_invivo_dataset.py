# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:48:50 2018

@author: Yi Zhang
"""

import sys
import torch
from scipy.io import loadmat

sys.path.append('../')
from classes.Player import Player
from network.ResNet3dC import ResNet3dC
from tools.eval_Res3dC import process_dataset

"""Settings"""
"""========================================================================="""
#Model file
mfile='../saved/Res3dC_invivo10epoch.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
gpu=True #if gpu=True, the ResNet will use more parameters

#Directory of input dataset and the name of the set to be processed
data_dir='../../../Data/Invivo/'
setname='test2'
#Directory of output patches
patch_dir='../../../Data/for_eval/output_patches/'
#whether to watch the first input and output
display=True
note='abs' #db or abs
minDB=None #if note==db, set the minimum of dB
cmap='hot'
"""========================================================================="""

#load model
net=ResNet3dC(gpu)

#device='cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
if mfile[-3:]=='pkl':
    state_dict=torch.load(mfile,map_location=device)
    #state_dict=delitems(state_dict)
    net.load_state_dict(state_dict)
else:
    net=torch.load(mfile)

#process        
process_dataset(net,data_dir,setname,patch_dir)
#display
if display:
    startnum={'train':0,'val':2400,'test1':3200,'test2':4000}[setname]
    Dname='Patch' if setname=='test2' else 'patch_180'
    D=loadmat('%s/D_data/%s/D%d.mat'%(data_dir,setname,startnum))[Dname]
    Si=loadmat('%s/data0.mat'%(patch_dir))['data']
    D=D.reshape([32,32,20])
    Si=Si.reshape([32,32,20])
    
    player=Player()
    player.play([D,Si],tit=['Input','Prediction'],
                cmap=cmap,note=note,minDB=minDB)
    
