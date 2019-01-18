# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 08:49:43 2018

@author: Yi Zhang
"""

import numpy as np
import sys
sys.path.append('../')
from scipy.io import loadmat,savemat
from classes.Player import Player

def switchrat():
    """Settings"""
    """====================================================================="""
    old_dir='../../../Data/Invivo/'
    new_dir='../../../Data/Invivo_converse/'
    """====================================================================="""
    
    #use the data from val,test1,test2 as new training data
    for dset in ['val','test1','test2']:
        numstart={'val':2400,'test1':3200,'test2':4000}[dset]
        Dname,Sname,Lname=['Patch','S_est_f','L_est_f'] if dset=='test2'\
                            else ['patch_180','patch_180','patch_180']
        
        print('old dataset: %s, new dataset: %s'%(dset,'train'))
        for ii in np.arange(numstart,numstart+800):
            if (ii+1)%100==0:
                print('  file number: %d to %d'%(ii,ii+100))
            D=loadmat('%s/D_data/%s/D%d.mat'%(old_dir,dset,ii))[Dname]
            S=loadmat('%s/fista/%s/S_fista%d.mat'%(old_dir,dset,ii))[Sname]
            L=loadmat('%s/fista/%s/L_fista%d.mat'%(old_dir,dset,ii))[Lname]
            
            savemat('%s/D_data/train/D%d.mat'%(new_dir,ii-2400),{'patch_180':D})
            savemat('%s/fista/train/S_fista%d.mat'%(new_dir,ii-2400),{'patch_180':S})
            savemat('%s/fista/train/L_fista%d.mat'%(new_dir,ii-2400),{'patch_180':L})
            
    #use the data from train as val, test1, test2 data
    for numstart in [0,800,1600]:
        dset_new={0:'val',800:'test1',1600:'test2'}[numstart]
        Dname_new,Sname_new,Lname_new=['Patch','S_est_f','L_est_f']\
                                if dset_new=='test2' else ['patch_180','patch_180','patch_180']
        
        print('old dataset: %s, new dataset: %s'%('train',dset_new))
        for ii in np.arange(numstart,numstart+800):
            if (ii+1)%100==0:
                print('  file number: %d to %d'%(ii,ii+100))
            D=loadmat('%s/D_data/train/D%d.mat'%(old_dir,ii))['patch_180']
            S=loadmat('%s/fista/train/S_fista%d.mat'%(old_dir,ii))['patch_180']
            L=loadmat('%s/fista/train/L_fista%d.mat'%(old_dir,ii))['patch_180']
            
            savemat('%s/D_data/%s/D%d.mat'%(new_dir,dset_new,ii+2400),{Dname_new:D})
            savemat('%s/fista/%s/S_fista%d.mat'%(new_dir,dset_new,ii+2400),{Sname_new:S})
            savemat('%s/fista/%s/L_fista%d.mat'%(new_dir,dset_new,ii+2400),{Lname_new:L})
        
def check():
    """Settings"""
    """====================================================================="""
    new_dir='../../../Data/invivo_converse/'
    setname='test1'
    filenum=700 #less than 2400 if setname=='train', else less than 800
    """====================================================================="""
    
    startnum={'train':0,'val':2400,'test1':3200,'test2':4000}[setname]
    Dname,Sname,Lname=['Patch','S_est_f','L_est_f'] if setname=='test2'\
                    else ['patch_180','patch_180','patch_180']
    filenum=filenum+startnum
    player=Player()
    
    D=loadmat('%s/D_data/%s/D%d.mat'%(new_dir,setname,filenum))[Dname]
    S=loadmat('%s/fista/%s/S_fista%d.mat'%(new_dir,setname,filenum))[Sname]
    L=loadmat('%s/fista/%s/L_fista%d.mat'%(new_dir,setname,filenum))[Lname]
    D=D.reshape([32,32,20])
    S=S.reshape([32,32,20])
    L=L.reshape([32,32,20])
    
    player.play([D,S,L])
    
#switchrat()
check()