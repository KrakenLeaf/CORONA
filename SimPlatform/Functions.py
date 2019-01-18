# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:26:41 2018

@author: Yi Zhang
"""

import numpy as np
from numpy.random import normal,uniform,rand,randint,choice
from SimPlatform.Parameters import params_default

"""Parameters for envelope of amplitude of tissue"""
         
"""
 1.sigrange: standard variation for two-dimensional gaussian as the amplitude envelop of tissue,
         unit mm. sig=[sig1min,sig1max,sig2min,sig2max]. sig1 corresponds to y dimension,
         sig2 corresponds to x dimension.
 2.num: number of the max areas of tissue
"""        

class Envelope:
    def __init__(self,params=params_default):
        self.pixel=params['pixel']
        self.shape=params['shape']
    
    def GMenvelope(self,params=params_default):
        """
        generate a mixed gaussian matrix as envelope of amplitude for tissue
        """
        self.sigrange=params['sigrange']
        self.num=params['numev']
    
        ev={}
        m={}
        sig={}
        loc_ev={}
        theta_ev={}
        for i in range(self.num):
            sig[i]=[uniform(self.sigrange[0],self.sigrange[1]),
                   uniform(self.sigrange[2],self.sigrange[3])]
            
            m[i]=max(int(3*sig[i][0]/self.pixel[0]),int(3*sig[i][1]/self.pixel[1]))
            xv=np.arange(-m[i],m[i]+1)*self.pixel[1]
            yv=np.arange(-m[i],m[i]+1)*self.pixel[0]
            xv,yv=np.meshgrid(xv,yv)
            
            theta_ev[i]=uniform(0,2*np.pi)
            xv1=xv*np.cos(theta_ev[i])-yv*np.sin(theta_ev[i])
            yv1=xv*np.sin(theta_ev[i])+yv*np.cos(theta_ev[i])
            
            ev[i]=np.exp(-xv1**2/2/sig[i][0]**2-yv1**2/2/sig[i][1])
            loc_ev[i]=(randint(self.shape[0]),randint(self.shape[1]))
        
        #set location of envelope
        A=np.zeros(self.shape)
        for i in range(self.num):
            y1,y2,x1,x2=[max(0,loc_ev[i][0]-m[i]),min(self.shape[0],loc_ev[i][0]+m[i]+1),
                        max(0,loc_ev[i][1]-m[i]),min(self.shape[1],loc_ev[i][1]+m[i]+1)]
                        
            ye1,ye2,xe1,xe2=[y1-loc_ev[i][0]+m[i],y2-loc_ev[i][0]+m[i],
                            x1-loc_ev[i][1]+m[i],x2-loc_ev[i][1]+m[i]]
                            
            A[y1:y2,x1:x2]+=ev[i][ye1:ye2,xe1:xe2]
        return A