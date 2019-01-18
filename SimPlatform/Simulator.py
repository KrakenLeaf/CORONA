# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:45:43 2018

@author: Yi Zhang
"""

import numpy as np
from scipy.signal import convolve2d
from SimPlatform.ZoneTissue import ZoneTissue
from SimPlatform.ZoneBubbles import ZoneBubbles
from SimPlatform.Parameters import params_default
from scipy.special import gamma

"""Parameters for ultrasonic device"""
        
"""
1.pixel: size of pixel of screen. unit=mm, pixel[0] corresponds to y dimension,
         pixel[1] corresponds to x dimension.
2.psf: the standard variation of the point spread function. psf[0] corresponds to
        y dimension, psf[1] corresponds to x dimension.
"""
        
class Simulator:
    def __init__(self,params=params_default):
        #global settings
        self.pixel=params['pixel'] #size of pixel, unit: mm
        self.shape=params['shape'] 
        self.ampbubbles=params['ampbubbles']
        self.amptissue=params['amptissue']
        self.sigman=params['ampnoise']
        self.sigman=self.sigman/np.sqrt(2)/gamma(1.5)
        
        #bubbles settings
        self.zTissue=ZoneTissue(params)
        self.zBubbles=ZoneBubbles(params)
        #psf
        s1,s2=params['psf']
        s1=s1/self.pixel[0]
        s2=s2/self.pixel[1]
        fshape=[int(10*s1),int(10*s2)]
        xc,yc=(fshape[1]-1)/2,(fshape[0]-1)/2
        xv,yv=np.meshgrid(np.arange(fshape[1])-xc,np.arange(fshape[0])-yc)
        self.filter=np.exp(-(xv**2/2/s2**2+yv**2/2/s1**2))
        self.filter=self.filter.astype(np.complex128)
        
    def generate(self,T=20):
        Tissue=np.zeros(self.shape+tuple([T]),dtype=np.complex128)
        Bubbles=np.zeros(self.shape+tuple([T]),dtype=np.complex128)
        noi=np.zeros(self.shape+tuple([T]),dtype=np.complex128)
        #phi=np.random.uniform(0,2*np.pi)
        #lmd=2*np.pi*15e6/34e4*self.pixel[0]
        #_,yv=np.meshgrid(np.arange(self.shape[1]),np.arange(self.shape[0]))
        #phase0=np.exp(1j*phi)*np.exp(1j*lmd*yv)
        #phase=np.zeros(self.shape+tuple([T]),dtype=np.complex128)
        
        for t in range(T):
            Bubbles[:,:,t]=self.zBubbles.image()
            Tissue[:,:,t]=self.zTissue.image()
            noi[:,:,t]=np.random.normal(0,self.sigman,self.shape)+\
                        1j*np.random.normal(0,self.sigman,self.shape)
            #phase[:,:,t]=phase0
            
            self.zBubbles.refresh()
            self.zTissue.refresh()
            
        #control the amplitude of bubbles, tissue and noise
        tmpb=Bubbles[Bubbles!=0]
        if not (np.sum(np.abs(Bubbles))==0):
            Bubbles=Bubbles*self.ampbubbles/np.mean(np.abs(tmpb))
        AMtissue=np.random.uniform(0.6,1,[1])
        Tissue=Tissue*self.amptissue/np.max(np.abs(Tissue))*AMtissue
        
        #point spread function
        for i in range(T):
            Bubbles[:,:,i]=self.psf(Bubbles[:,:,i])
            Tissue[:,:,i]=self.psf(Tissue[:,:,i])
            noi[:,:,i]=self.psf(noi[:,:,i])
        Sum=Bubbles+Tissue+noi
            
        return Sum,Bubbles,Tissue
    
    def psf(self,A):
        return convolve2d(A,self.filter,mode='same')
        
        
        
        
        