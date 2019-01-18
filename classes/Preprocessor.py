# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:17:20 2018

@author: Yi Zhang
"""

import numpy as np

class Preprocessor:
    def cSVT(self,D,rank,getS1=False):
        """
        S: output of cSVT
        S1: complementary of S
        """
        u,s,vh=np.linalg.svd(D)
        s1=s[0:rank]
        s[0:rank]=0
        m,n=D.shape
        S=np.zeros(D.shape,dtype=np.complex128)
            
        for i in np.arange(rank,min(m,n)):
            S[:,i]=u[:,i]*s[i]
        S=S@vh
        
        if getS1:
            S1=np.zeros(D.shape,dtype=np.complex128)
            for i in range(rank):
                S1[:,i]=u[:,i]*s1[i]
            S1=S1@vh
            return S,S1
        
        return S
    
    def cSVT1(self,D):
        shape=D.shape
        D=D.reshape([shape[0]*shape[1],shape[2]])
        u,s,vh=np.linalg.svd(D)
        s[0]=0
        m,n=D.shape
        S=np.zeros(D.shape,dtype=np.complex128)
            
        for i in np.arange(1,min(m,n)):
            S[:,i]=u[:,i]*s[i]
        S=S@vh
                
        return S.reshape(shape)
    
    def block_process(self,data,proc,params):
        """
        stride should be less than half of block size
        drop should be less than half of block size
        """
        shape=np.array(data.shape)
        Out=np.zeros(shape,dtype=data.dtype)
        bshape=np.array(params['bshape'])
        dim=len(bshape)
        stride=np.array(params['stride'])
        drop=np.array(params['drop'])
        bbeg=np.zeros([dim]).astype(np.int)
        bend=np.zeros([dim]).astype(np.int)+np.array(bshape)
        
        while(True):
            #drop part of the block
            bend=np.minimum(shape,bend)
            dropbeg=(bbeg!=0).astype(np.int)
            dropend=(bend!=shape).astype(np.int)
            beg=bbeg+dropbeg*drop
            end=bend-dropend*drop
            br=beg-bbeg
            er=end-bbeg
            
            if dim==2:
                tmp=proc(data[bbeg[0]:bend[0],bbeg[1]:bend[1]])
                Out[beg[0]:end[0],beg[1]:end[1]]=tmp[br[0]:er[0],br[1]:er[1]]
                #print(tmp[br[0]:er[0],br[1]:er[1]])
                #print(Out)
            elif dim==3:
                tmp=proc(data[bbeg[0]:bend[0],bbeg[1]:bend[1],bbeg[2]:bend[2]])
                Out[beg[0]:end[0],beg[1]:end[1],beg[2]:end[2]]=tmp[br[0]:er[0],br[1]:er[1],br[2]:er[2]]
    
            #change the block
            for ii in range(dim):
                bbeg[ii]+=stride[ii]
                if (bbeg<shape).all() and  not(bend[ii]==shape[ii]):
                    break
                bbeg[0:ii+1]=0
            
            bend=bbeg+bshape
            if not bbeg.any():
                break
            
        return Out
        
