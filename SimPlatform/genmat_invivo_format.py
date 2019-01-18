# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:53:01 2018

@author: Yi Zhang
"""

from scipy.io import savemat
from SimPlatform.Simulator import Simulator
from SimPlatform.Parameters import params_default
import sys
sys.path.append('../')
from classes.Player import Player

folder='../../../Data/Sim1/'
setname='val'
numInst=16

Dname,Sname,Lname=['patch_180','patch_180','patch_180']\
                  if setname!='test2' else ['Patch','S_est_f','L_est_f']
#the start number of .mat file                      
numstart={'train':0, 'val':2400, 'test1':3200, 'test2':4000}[setname] 

params=params_default
shape=(128,128)
T=20
params['shape']=shape
rIter=int(shape[0]/32)
cIter=int(shape[1]/32)
player=Player()

nIter=int(numInst/shape[0]/shape[1]/T*32*32*20)

print('total iterations and instances: %d, %d'%(nIter,numInst))
numf=numstart
for i in range(nIter):
    print('current iteration: %d, file number: %d to %d'%(i,numf,numf+rIter*cIter))
    simtor=Simulator(params)
    Sum,Bubbles,Tissue=simtor.generate(T)
    for rr in range(rIter):
        for cc in range(cIter):
            D=Sum[rr*32:(rr+1)*32,cc*32:(cc+1)*32,0:20]
            S=Bubbles[rr*32:(rr+1)*32,cc*32:(cc+1)*32,0:20]
            L=Tissue[rr*32:(rr+1)*32,cc*32:(cc+1)*32,0:20]
            
            savemat(folder+'D_data/%s/D%d.mat'%(setname,numf),{Dname:D.reshape([32*32,20])})
            savemat(folder+'fista/%s/S_fista%d.mat'%(setname,numf),{Sname:S.reshape([32*32,20])})
            savemat(folder+'fista/%s/L_fista%d.mat'%(setname,numf),{Lname:L.reshape([32*32,20])})
            numf+=1
            
player.play([D,S,L],cmap='hot')
            
            

