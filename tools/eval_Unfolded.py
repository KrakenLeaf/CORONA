# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:51:37 2018

@author: Yi Zhang
"""

from scipy.io import savemat,loadmat
import torch.utils.data as data
import torch
import numpy as np

import sys
sys.path.append('../')
from classes.Dataset import Converter
from Unfolded.DataSet_Unfolded import ImageDataset

def process_dataset(net,data_dir,setname,patch_dir):  
    """
    the function will process datas from one dataset and save all the 
    output patches to the specified directory
    
    Args:
        net: the model used for processing data
        data_dir: the directory of dataset
        setname: train/val/test1/test2
        patch_dir: the directory of output patches
    """
    Instances=800
    BatchSize=100
    
    shape_dset=(32,32,40)
    trainID={'train':0,'val':1,'test1':2,'test2':3}[setname]
    formout={'pre':'concat','shape':(1024,20)}
    
    dataset=ImageDataset(round(Instances), shape_dset, train=trainID,data_dir=data_dir)
    loader =data.DataLoader(dataset, batch_size=BatchSize, shuffle=False)
    conter=Converter()
    
    numf=0
    with torch.no_grad():
        print('Total iterations: %d'%int(Instances/BatchSize))
        for niter,(_,_,D) in enumerate(loader):
            print('iteration %d'%niter)
            for ii in range(BatchSize):
                outL,outS=net(D[ii])
                [outL,outS]=conter.torch2np([outL,outS],[formout,formout])
                savemat('%s/data%d'%(patch_dir,numf),{'data':outS})
                savemat('%s/dataL%d'%(patch_dir,numf),{'data':outL})
                numf+=1
                
def process_patch(net,data_dir,arange,matname=None):
    """
    the function will process one patch from the data in the specified file
    
    Args:
        net: the model
        data_dir: the directory of data file
        matname: if data file is .mat file, the name of matrix is needed
    """
    brow,erow,bcol,ecol,bt,et=arange

    formin={'pre':'concat','shape':((erow-brow),(ecol-bcol),(et-bt)*2)}
    formout={'pre':'concat','shape':(erow-brow,ecol-bcol,et-bt)}
    
    conter=Converter()
    if data_dir[-3:]=='npz':
        D=np.load(data_dir)['arr_0'][brow:erow,bcol:ecol,bt:et]
    else:
        D=loadmat(data_dir)['vid'][brow:erow,bcol:ecol,bt:et]
    D=D/np.max(np.abs(D))
    [D1]=conter.np2torch([D],[formin])
    
    with torch.no_grad():
        Lp,Sp=net(D1)
    [Sp,Lp]=conter.torch2np([Sp,Lp],[formout,formout])
    
    return D,Sp,Lp

