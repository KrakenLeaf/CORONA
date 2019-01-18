# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 23:29:00 2018

@author: Yi Zhang
"""

import torch
import torch.nn as nn

class Conv3dC(nn.Module):
    """
    Conv block for 3d complex computation
    """
    def __init__(self,Cin,Cout,kernel,stride,padding):
        """
        Args:
            Cin: number of input panes
            Cout: number of output panes
            kernel: (w1,w2), w1 for X and Y dimension, w2 for T dimension
            stride: (s1,s2), s1 for X and Y dimension, s2 for T dimension
            padding: (p1,p2), p1 for X and Y dimension, p2 for T dimension
        """
        super(Conv3dC,self).__init__()
        
        w1,w2=kernel
        s1,s2=stride
        p1,p2=padding
        self.convR=nn.Conv3d(Cin,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
        self.convI=nn.Conv3d(Cin,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2))
    
    def forward(self,xR,xI):
        xR,xI=self.convR(xR)-self.convI(xI),self.convR(xI)+self.convI(xR)

        return xR,xI
    
class ResBlock3dC(nn.Module):
    """
    Res Block for 3d complex computation
    1. suppose Cin==Cout
    """
    def __init__(self,Cin,Cout):
        super(ResBlock3dC,self).__init__()
        
        w1,w2=3,3
        s1,s2=1,1
        p1,p2=1,1
        
        self.relu=nn.ReLU()
        self.convR1=nn.Conv3d(Cin,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2),
                              bias=False)
        self.convI1=nn.Conv3d(Cin,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2),
                              bias=False)
        self.bn1=nn.BatchNorm3d(Cout)
        self.convR2=nn.Conv3d(Cout,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2),
                              bias=False)
        self.convI2=nn.Conv3d(Cout,Cout,(w1,w1,w2),(s1,s1,s2),(p1,p1,p2),
                              bias=False)
        self.bn2=nn.BatchNorm3d(Cout)
        
    def forward(self,xR,xI):
        yR,yI=self.convR1(xR)+self.convI1(xI),self.convR1(xI)+self.convI1(xR)
        yR=self.relu(self.bn1(yR))
        yI=self.relu(self.bn1(yI))
        yR,yI=self.convR2(yR)+self.convI2(yI),self.convR2(yI)+self.convI2(yR)
        yR=self.bn2(yR)
        yI=self.bn2(yI)
        yR+=xR
        yI+=xI
        
        return self.relu(yR),self.relu(yI)

class ResNet3dC(nn.Module):
    """
    ResNet for 3d computation
    1. input: x should be numpy.ndarray type, 
              size=[batchsize,channels=1,H,W,T],complex data type
    
    2. output: y, size=[batchsize,channels=1,H,W,T],complex data type
    """
    def __init__(self,gpu=True):
        super(ResNet3dC,self).__init__()
        
        if gpu:
            #GPU version     
            c =[ 1,16, 8, 8, 8, 1]
            w1=[ 0, 5, 3, 3, 3, 3]
            w2=[ 0, 5, 3, 3, 3, 3]
            s1=[ 0, 1, 1, 1, 1, 1]
            s2=[ 0, 1, 1, 1, 1, 1]
            p1=[ 0, 2, 1, 1, 1, 1]
            p2=[ 0, 2, 1, 1, 1, 1]

        else:
            c =[ 1,16, 4, 4, 4, 1]
            w1=[ 0, 3, 3, 3, 3, 3]
            w2=[ 0, 3, 3, 3, 3, 3]
            s1=[ 0, 1, 1, 1, 1, 1]
            s2=[ 0, 1, 1, 1, 1, 1]
            p1=[ 0, 1, 1, 1, 1, 1]
            p2=[ 0, 1, 1, 1, 1, 1]        
        
        self.relu=nn.ReLU()
        self.conv1=Conv3dC(c[0],c[1],(w1[1],w2[1]),
                             (s1[1],s2[1]),(p1[1],p2[1]))        
        self.bn1R=nn.BatchNorm3d(c[1])
        self.bn1I=nn.BatchNorm3d(c[1])
        self.conv2=Conv3dC(c[1],c[2],(w1[2],w2[2]),
                             (s1[2],s2[2]),(p1[2],p2[2]))
        self.bn2R=nn.BatchNorm3d(c[2])
        self.bn2I=nn.BatchNorm3d(c[2])
        self.res3=ResBlock3dC(c[2],c[3])
        self.res4=ResBlock3dC(c[3],c[4])

        self.conv5=Conv3dC(c[4],c[5],(w1[5],w2[5]),
                             (s1[5],s2[5]),(p1[5],p2[5]))        
        
    def forward(self,x):
        T2=x.shape[-1]
        T=int(T2/2)
        xR=x[:,:,:,:,0:T]
        xI=x[:,:,:,:,T:T2]
        
        xR,xI=self.conv1(xR,xI)
        xR=self.relu(self.bn1R(xR))
        xI=self.relu(self.bn1I(xI))
        xR,xI=self.conv2(xR,xI)
        xR=self.relu(self.bn2R(xR))
        xI=self.relu(self.bn2I(xI))
        
        xR,xI=self.res3(xR,xI)
        xR,xI=self.res4(xR,xI)
        
        xR,xI=self.conv5(xR,xI)
        
        x=torch.cat((xR,xI),-1)
        
        return x
        