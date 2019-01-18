# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:41:33 2018

@author: Yi Zhang
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio

def mat2gif(mvlist,save_dir,note='abs',cmap='hot',minDB=-50,
            tit=None,supt=None,play=False):
    if play:
        plt.ion()
    else:
        plt.ioff()

    subp={1:[1,1],2:[1,2],3:[1,3],4:[2,2],5:[2,3],6:[2,3]}
    p1,p2=subp[len(mvlist)]
    T=mvlist[0].shape[-1]
    fig,ax=plt.subplots(p1,p2) 
    if p1*p2==1:
        ax=np.array([ax]) 
    ax=ax.reshape([-1])
    images=[]
        
    #init
    vmin,vmax=[minDB,0] if note=='db' else [0,1]
    
    print('saving movie as GIF...')
    for i in range(len(mvlist)):
        US=mvlist[i]
        if US is None:
            continue
        if US.dtype is torch.float32:
            US=US.detach().numpy().squeeze()
        
        US=np.abs(US)
        if np.sum(np.abs(US))!=0:
            US=US/np.max(US)
        if note=='db':
            US[US<10**(minDB/20)]=10**(minDB/20)
            US=20*np.log10(US)
        mvlist[i]=US
    
    for t in range(T):  
        for i in range(len(mvlist)):
            if mvlist[i] is None:
                continue
            ax[i].clear()
            ax[i].imshow(mvlist[i][:,:,t],cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
            if not(tit is None):
                ax[i].set_title(tit[i])
        if supt==None:
            supt=''
        fig.suptitle('%dth Frame '%(t+1)+supt)
        #plt.show()
        print('%d frame saved'%t)
        
        fig.savefig('../../../Data/Temp/frame%d.png'%t)
        if play:
            plt.pause(0.1)
                
    for t in range(T):
        images.append(imageio.imread('../../../Data/Temp/frame%d.png'%t))
    imageio.mimsave(save_dir, images)
                
    