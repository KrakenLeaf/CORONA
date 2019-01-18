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
from network.UnfoldedNet3dC import UnfoldedNet3dC
from tools.eval_Unfolded import process_dataset

"""Settings"""
"""========================================================================="""
#Model file
# 50 epochs on SIM, 20 on In-vivo
#mfile='../Unfolded/Results/GoodResults/Unfolded_sim_in_vivo_20epochs_converse/Oren_after_sim_converse_invivo_Unfolded_Model_al0.50_Tr2400_epoch20_lr2.00e-03.pkl'

# 10 epochs on SIM, 10 on In-vivo
mfile = '../Unfolded/Results/GoodResults/Unf_coefL0.4_coef1.8_siminvivo10epoch/invivoCPU_Unfolded,ALPHA0.5000_Learningrate0.010000_NumEpochs10_TrainInstances2400.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
params_net={'layers':10,
            'kernel':[(5,1)]*3+[(3,1)]*7,
            'coef_L':0.4,
            'coef_S':1.8,
            'CalInGPU':False}
#Directory of input dataset and the name of the set to be processed
data_dir='../../../Data/Invivo/'
setname='test2' #'test2'
#Directory of output patches
patch_dir='../../../Data/for_eval/siminvivo_10epochs/'
#whether to watch the first input and output
display=True
note='abs' #db or abs
minDB=None #if note==db, set the minimum of dB
cmap='hot'
"""========================================================================="""

#load model
net=UnfoldedNet3dC(params_net)

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
    Li=loadmat('%s/dataL0.mat'%(patch_dir))['data']
    D=D.reshape([32,32,20])
    Si=Si.reshape([32,32,20])
    Li=Li.reshape([32,32,20])
    
    player=Player()
    player.play([D,Si,Li],tit=['Input','Prediction S','Prediction_L'],
                cmap=cmap,note=note,minDB=minDB)
    
