# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:18:56 2018

@author: Yi Zhang
"""

import numpy as np
from SimPlatform.Parameters import params_default

class ZoneBubbles:
    def __init__(self,params=params_default):
        self.pixel=params['pixel']
        self.shape=params['shape'][0:2]
        self.dt=params['dt']
        
        self.xrange=np.array([-0.5,self.shape[1]-0.5])*self.pixel[0]
        self.yrange=np.array([-0.5,self.shape[0]-0.5])*self.pixel[1]
        self.vmean=params['vmean']
        self.p_birth=params['p_birth']
        self.p_death=params['p_death']
        self.muv=params['muv']
        self.sigmav=params['sigmav']
        self.sigmaa=params['sigmaa']
        self.sigmaA=params['sigmaA']
        self.bubbleAM=params['bubbleAM']
        self.sdv_phib=params['sdv_phib']
        
        #location of bubbles
        S=self.shape[0]*self.shape[1]*self.pixel[0]*self.pixel[1]
        num=int(S*np.random.rand()*params['density']/1e2) #the unit of density: cm^(-2)
        self.x=np.random.uniform(-0.5,self.shape[1]-0.5,[num])*self.pixel[0]
        self.y=np.random.uniform(-0.5,self.shape[0]-0.5,[num])*self.pixel[1]
        
        #velocity of bubbles
        v=self.vmean*np.random.normal(self.muv,self.sigmav,[num])
        v[v<0]=0
        theta=np.random.uniform(0,2*np.pi,[num])
        self.vx=v*np.cos(theta)
        self.vy=v*np.sin(theta)
        
        #acceleration of bubbles
        self.ax=np.random.normal(0,self.sigmaa,[num])
        self.ay=np.random.normal(0,self.sigmaa,[num])
        
        #complex amplitude of bubbles
        self.A=np.random.normal(0,self.sigmaA,[num])+1j*np.random.normal(0,self.sigmaA,[num])
        #self.A=np.abs(self. A)*np.exp(1j*np.random.normal(0,self.sdv_phib,[num])*np.pi/180)
        
    def refresh(self):
        """Change the location, velocity, acceleration and Complex Amplitude of bubbles,
        delete the bubbles that are out of the zone, give birth to new bubbles and vanish 
        part of the old bubbles"""
        #refresh location
        self.x=self.x+self.vx*self.dt
        self.y=self.y+self.vy*self.dt
        #refresh velocity
        self.vx=self.vx+self.ax*self.dt
        self.vy=self.vy+self.ay*self.dt
        direct=np.random.uniform(-30,30,self.vx.shape)*np.pi/180
        self.vx=self.vx*np.cos(direct)-self.vy*np.sin(direct)
        self.vy=self.vx*np.sin(direct)+self.vy*np.cos(direct)
        #refresh acceleration
        self.ax=np.random.normal(0,self.sigmaa,self.x.shape)
        self.ay=np.random.normal(0,self.sigmaa,self.x.shape)
        #refresh amplitude
        AM=np.random.uniform(1-self.bubbleAM,1+self.bubbleAM,self.A.shape)
        self.A*=AM
        #delete the bubbles that are out of the zone
        flag1=np.bitwise_and(self.x>=self.xrange[0],self.x<=self.xrange[1])
        flag2=np.bitwise_and(self.y>=self.yrange[0],self.y<=self.yrange[1])
        flag=np.bitwise_and(flag1,flag2)
        self.x=self.x[flag]
        self.y=self.y[flag]
        self.vx=self.vx[flag]
        self.vy=self.vy[flag]
        self.ax=self.ax[flag]
        self.ay=self.ay[flag]
        self.A=self.A[flag]
        
    def image(self):
        locx=np.round(self.x/self.pixel[0]).astype(np.int)
        locy=np.round(self.y/self.pixel[1]).astype(np.int)
        A=np.zeros(self.shape,dtype=np.complex128)
        
        A[tuple(locy),tuple(locx)]=self.A
        
        return A

        