# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:48:39 2018

@author: Yi Zhang
"""

import numpy as np
import sys
sys.path.append('../')
from classes.Player import Player
from SimPlatform.ZoneTissue import ZoneTissue
from SimPlatform.ZoneBubbles import ZoneBubbles
from SimPlatform.Simulator import Simulator
from SimPlatform.Functions import Envelope
from SimPlatform.Parameters import params_default

player=Player()

# =============================================================================
# #zTissue=ZoneTissue()
# zTissue=ZoneTissue()
# 
# T=30
# shape=params_default['shape']
# Tissue=np.zeros([shape[0],shape[1],T],dtype=np.complex128)
# 
# for i in range(T):
#     Tissue[:,:,i]=zTissue.image()
#     zTissue.refresh()
#     
# player.play([Tissue])
# =============================================================================
params=params_default
params['shape']=(128,128)
sim=Simulator(params)
T=50

Sum,Bubbles,Tissues=sim.generate(T)

# =============================================================================
# Sum=Sum[0:32,0:32,:]
# Bubbles=Bubbles[0:32,0:32,:]
# Tissues=Tissues[0:32,0:32,:]
# 
# =============================================================================
player.play([Sum,Bubbles,Tissues],cmap='hot',note='abs')
#player.play([Sum],cmap='hot')
ang=np.unwrap(np.angle(Sum)*180/np.pi)


