# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:57:37 2018

@author: Yi Zhang
"""

"""Parameters for ultrasonic device"""
        
"""
1.pixel: size of pixel of screen. unit=mm, pixel[0] corresponds to y dimension,
         pixel[1] corresponds to x dimension.
2.psf: the standard variation of the point spread function. psf[0] corresponds to
        y dimension, psf[1] corresponds to x dimension.
3.shape: shape of the screen
4.dt: duration of each frame
5.amptissue: mean amplitude of tissue
6.c2: mean amplitude
"""        
dt=0.01
params_default={        
        'pixel':(0.12,0.12),'psf':(0.14,0.32),'shape':(100,100),
        'dt':dt, 'amptissue':1, 'ampbubbles':3, 'ampnoise':0.1
        }

"""Parameters for tissue"""
        
"""
1.ev: envelope of the amplitude of tissue
2.sdv_phi: standard variation for initial phase of tissue, unit: degree
3.sdv_lp: the standard variation of 2d gaussian kernel as the low pass filter of amplitude,
        unit mm
4.kerflow: the size of probability kernel to represent flowing of tissue cells
5.sdv_df: the standard variation of the change of flowing kernel
6.minf: the minimum value of probability in flowing filter
7.block: the size of block to share a common flow pattern
8.numk: number of kernel to simulate the pattern of flow
"""
        
params_tissue={        
        'sdv_phi':15, 'sdv_lp':0.2, 'kerflow':3, 'sdv_df':0.3/3, 'minf':0.1,
        'block':4, 'numk':4
        }

"""Parameters for bubbles"""
        
"""
1.vmean: mean velocity of the bubbles. unit=mm/s
2.density: maximum density of bubbles. unit=cm^(-2)
3.muv: expectation of the envelope of velocity
4.sigmav: standard variation of the envelope of velocity
"""

params_bubbles={
        #bubbles
        'vmean':0.24/dt,'density':130,'p_birth':0.3,'p_death':0.3,
        'maxBirth':3,'maxDeath':3, 'sdv_phib':5,
        'muv':1,'sigmav':1,'sigmaa':0.05*(0.12/dt)/dt,'sigmaA':1,'bubbleAM':0.1
        }

params_ev={
        'sigrange':[15*0.14,30*0.14,15*0.16,30*0.16], 'numev':5
        } 

params_default.update(params_tissue)
params_default.update(params_bubbles)
params_default.update(params_ev)
