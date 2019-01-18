# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:55:12 2018

@author: Yi Zhang
"""

from Player import Player
import numpy as np
import matplotlib.pyplot as plt

player=Player()
plt.close('all')

for i in range(100):
    x=np.random.rand(3,4)
    y=np.random.rand(3,4)
    z=np.random.rand(3,4)
    w=np.random.rand(3,4)
    player.plotmat([x],supt='%d'%i)#,tit=['a','b','c','d'])
    plt.pause(0.1)

x1=np.random.rand(3,4,10)
y1=np.random.rand(3,4,10)
player.play([x1,y1],supt='A',tit=['a','b'])

# =============================================================================
# player.play([np.random.rand(3,4,100)])
# =============================================================================
