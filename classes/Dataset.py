# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:29:11 2018

@author: Yi Zhang
"""

import numpy as np
import os
import torch
from scipy.io import loadmat
import random
            
class Dataset:
    def __init__(self,folder=None,shuffle=None,prefix=None,shownum=10):
        self._folder=None
        if not folder is None:
            self._folder=folder
            self._flist=os.listdir(folder)
        if not prefix is None:    
            tmp=[]
            for s in self._flist:
                if s.startswith(prefix):
                    tmp.append(s)
            self._flist=tmp
        self._flist.sort()
        if not shuffle is None:
            random.shuffle(self._flist,shuffle)
            print(self._flist[0:shownum])
            
    def getnpz(self,fileind,arrind,folder=None):
        if folder is None:
            folder=self._folder
            flist=self._flist
        else:
            flist=os.listdir(folder)
        data=np.load(folder+flist[fileind],mmap_mode='r+')
        dlist=[]
        l=0
        for x in data:
            if l in arrind:
                dlist.append(data[x])
            l=l+1
        return dlist
    
    def getmat(self,fileind,arrind,folder=None):
        if folder is None:
            folder=self._folder
            flist=self._flist
        else:
            flist=os.listdir(folder)
        data=loadmat(folder+flist[fileind])
        dlist=[]
        for x in arrind:
            dlist.append(data[x])
        return dlist

                
class Converter:
    def __init__(self):
        """
        Preprocessing:
        1.concat: inv=False, concatenate real and imaginary parts in axis=-1
            inv=True, depart two parts in -1 axis into real and imaginary parts
        2.ch2: inv=False, concatenate real and imaginary parts in a new axis=0
            inv=True, depart two parts in 1 axis into real and imaginary parts
        3.stack: inv=False, stack real and imaginary parts in axis=-1
            inv=True, depart two parts in -1 axis into real and imaginary parts
        3.None: pass
        """        
        self.pre={'concat':self.concat,'ch2':self.ch2,'stack':self.stack}
    
    def np2torch(self,xnp,formlist):
        dlist=[]
        for i,x in enumerate(xnp):
            if ('pre' in formlist[i]) and (formlist[i]['pre'] in self.pre):
                x=self.pre[formlist[i]['pre']](x,inv=False)
            x=x.reshape(formlist[i]['shape'])
            x=torch.tensor(x,dtype=torch.float32)
            dlist.append(x)
        return dlist
    
    def torch2np(self,xtorch,formlist):
        dlist=[]
        for i,x in enumerate(xtorch):
            if torch.cuda.is_available():
                x=x.cpu().detach().numpy()
            else:
                x=x.detach().numpy()
            if ('pre' in formlist[i]) and (formlist[i]['pre'] in self.pre):
                x=self.pre[formlist[i]['pre']](x,inv=True)
            x=x.reshape(formlist[i]['shape'])            
            dlist.append(x)
        return dlist
            
    def stack(self,x,inv):
        if not inv:
            #not inv, np2torch
            x=np.swapaxes(x,0,-1)
            xr=x.real
            xi=x.imag
            size=list(x.shape)
            size[0]*=2
            size=tuple(size)
            z=np.zeros(size)
            z[0::2]=xr
            z[1::2]=xi
            z=np.swapaxes(z,0,-1)
        else:
            #inv, torch2np
            #size=[B,C,H,W,(T)],numpy
            x=np.swapaxes(x,0,-1)
            xr=x[0::2]
            xi=x[1::2]
            z=xr+1j*xi
            z=np.swapaxes(z,0,-1)
            
        return z
    
    def concat(self,x,inv):
        if not inv:
            #not inv, np2torch
            z=np.concatenate((x.real,x.imag),axis=-1)
        else:
            #inv, torch2np
            #size=[B,C,H,W,(T)],numpy
            x=np.swapaxes(x,0,-1)
            n=x.shape[0]
            nh=int(n/2)
            xr=x[0:nh]
            xi=x[nh:n]
            z=xr+1j*xi
            z=np.swapaxes(z,0,-1)
            
        return z
    
    def ch2(self,x,inv):
        pass
            