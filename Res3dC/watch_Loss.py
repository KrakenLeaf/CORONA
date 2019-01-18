# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:07:37 2018

@author: Yi Zhang
"""

import pickle

#fig_trace='./Results/invivoCPU_Lossmean_lr1.0e-03_epoch10.fig.pickle'
fig_trace='./Results/GoodResults/invivo10epoch_batch40/invivoCPU_Lossmean_lr1.0e-03_epoch10.fig.pickle'
fig_trace=pickle.load(open(fig_trace, 'rb'))

fig_trace.show() # Show the figure, edit it, etc.!

data = fig_trace.axes[0].lines[0].get_data()