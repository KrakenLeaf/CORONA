# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:10:58 2018

@author: Yi Zhang
"""

import numpy as np
from numpy.random import uniform,normal,rand,randint
from scipy.signal import convolve2d as conv2d
from SimPlatform.Parameters import params_default
from SimPlatform.Functions import Envelope

"""Parameters for ultrasonic device"""
        
"""
1.pixel: size of pixel of screen. unit=mm, pixel[0] corresponds to y dimension,
         pixel[1] corresponds to x dimension.
2.psf: the standard variation of the point spread function. psf[0] corresponds to
        y dimension, psf[1] corresponds to x dimension.
"""

"""Parameters for tissue"""
        
"""
1.ev: envelope of amplitude
2.sdv_phi: standard variation for initial phase of tissue, unit: degree
3.sdv_lp: the standard variation of 2d gaussian kernel as the low pass filter of amplitude,
        unit mm
4.kerflow: the size of probability kernel to represent flowing of tissue cells
5.sdv_df: the standard variation of the change of flowing kernel
6.minf: the minimum value of probability in flowing filter
7.block: the size of block to share a common flow pattern
8.numk: number of kernel to simulate the pattern of flow
"""

class ZoneTissue:
    def __init__(self,params=params_default):
        self.pixel=params['pixel']       
        self.shape=params['shape']
        
        self.ev=Envelope(params).GMenvelope()
        self.sdv_phi=params['sdv_phi']
        self.sdv_lp=params['sdv_lp']
        self.kerflow=params['kerflow']
        self.sdv_df=params['sdv_df']
        self.minf=params['minf']
        self.block=params['block']
        self.numk=params['numk']
        #self.maxAmpt=params['maxAmpt']
            
        AM=normal(0,1,self.shape)+1j*normal(0,1,self.shape)
        
        #convolve gaussian and normal
        mlp=max(int(3*self.sdv_lp/self.pixel[0]),int(3*self.sdv_lp/self.pixel[1]))
        x=np.arange(-mlp,mlp+1)*self.pixel[1]
        y=np.arange(-mlp,mlp+1)*self.pixel[0]
        x1,y1=np.meshgrid(x,y)
        lpfilter=np.exp(-x1**2/2/self.sdv_lp**2-y1**2/2/self.sdv_lp**2)
        #get amplitude of the tissue
        A=np.abs(conv2d(AM*self.ev,lpfilter,mode='same'))
        
        #generate phase field for tissue
        phi=uniform(0,2*np.pi)
        #phi=0
        dp=np.random.normal(0,self.sdv_phi,A.shape)*np.pi/180
        self.template=A*np.exp(1j*(phi+dp))
       #self.template=self.template*self.maxAmpt/np.max(np.abs(self.template))
        
        #generate the flowing filter
        self.flowfilt=np.zeros([self.numk,self.kerflow,self.kerflow])
        for i in range(self.numk):
            tmp=rand(self.kerflow,self.kerflow)
            tmp=tmp/np.sum(tmp)
            self.flowfilt[i]=tmp
        
    def refresh(self):
        df=normal(0,self.sdv_df,[self.numk,self.kerflow,self.kerflow])
        tmp=np.maximum(self.minf,df+self.flowfilt)
        for i in range(self.numk):           
            self.flowfilt[i]=tmp[i]/np.sum(tmp[i])
        
    def image(self):            
        B=np.zeros([self.numk,self.shape[0],self.shape[1]],dtype=np.complex128)
        for i in range(self.numk):
            B[i]=conv2d(self.template,self.flowfilt[i],mode='same')
        
        A=np.zeros([self.shape[0],self.shape[1]],dtype=np.complex128)
        for i in np.arange(0,self.shape[0],self.block):
            for j in np.arange(0,self.shape[1],self.block):
                rend=min(i+self.block,self.shape[0]-1)
                cend=min(j+self.block,self.shape[1]-1)
                cho=randint(self.numk)
                A[i:rend,j:cend]=B[cho,i:rend,j:cend]
        return A