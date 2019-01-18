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
from Res3dC.DataSet_3dC import ImageDataset

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
                outS=net(D[ii][None,None])
                [outS]=conter.torch2np([outS],[formout])
                savemat('%s/data%d'%(patch_dir,numf),{'data':outS})
                numf+=1
                
def delitems(dic):
    delword={"bn1R.num_batches_tracked", "bn1I.num_batches_tracked",
             "bn2R.num_batches_tracked", "bn2I.num_batches_tracked", 
             "res3.bn1.num_batches_tracked", "res3.bn2.num_batches_tracked", 
             "res4.bn1.num_batches_tracked", "res4.bn2.num_batches_tracked"}
    for k in list(dic.keys()):
        if k in delword:
            del dic[k]
    return dic

def process_patch(net,data_dir,arange,matname=None):
    """
    the function will process one patch from the data in the specified file
    
    Args:
        net: the model
        data_dir: the directory of data file
        matname: if data file is .mat file, the name of matrix is needed
    """
    brow,erow,bcol,ecol,bt,et=arange

    formin={'pre':'concat','shape':(-1,1,(erow-brow),(ecol-bcol),(et-bt)*2)}
    formout={'pre':'concat','shape':(erow-brow,ecol-bcol,et-bt)}
    
    conter=Converter()
    if data_dir[-3:]=='npz':
        D=np.load(data_dir)['arr_0'][brow:erow,bcol:ecol,bt:et]
    else:
        D=loadmat(data_dir)[matname][brow:erow,bcol:ecol,bt:et]
    [D1]=conter.np2torch([D],[formin])
    
    with torch.no_grad():
        Sp=net(D1)
    [Sp]=conter.torch2np([Sp],[formout])
    
    return D,Sp

