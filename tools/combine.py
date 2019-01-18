# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:20:03 2018

@author: Yi Zhang
"""

import numpy as np
from scipy.signal import tukey
from scipy.io import loadmat
import scipy.io as sio

def combineSim():
    data_dir='./SimResults/OutputPatches/'
    save_dir='./SimResults/SimOutput'
    numf=0
    result=np.zeros([128,128,20],dtype=np.complex)
    for rr in range(4):
        for cc in range(4):
            print('row=%d, colum=%d'%(rr,cc))
            pred=np.load(data_dir+'patch%d.npz'%numf)['arr_0']
            result[rr*32:(rr+1)*32,cc*32:(cc+1)*32,:]=pred
            numf+=1
    np.savez(save_dir,result)
    
def combineInvivo(data_dir=None,save_dir=None,filename=None,matname=None,start=None):
    if data_dir is None:
        #data_dir = '../../../Data/for_eval/Oren_test_invivo2_unf_converse'
        data_dir = '../../../Data/Invivo_converse/fista/test2' # FISTA
    if save_dir is None:
        #save_dir = '../../../Data/for_eval/Oren_test_invivo2_unf_movie_converse'
        save_dir = '../../../Data/Invivo_converse/fista/test2_movie'
    if filename is None:
       #filename = 'dataL' #'dataL'
        filename = 'S_fista'
    if matname is None:
        #matname  = 'data'
        matname = 'S_est_f' #patch_180 S_est_f
    PatchMov2=np.zeros([1024,160,100],dtype=np.complex128)
    vid=np.zeros([169,256,100],dtype=np.complex)
    
    Dimensions={'X':32,'Y':32}
    ImageSize={'X':169,'Y':256}
    Overlap={'X':16,'Y':16}
    Params={'Type':'combine'}
    
    h=0
    if start is None:
        start=4000
    
    for jj in range(160):
        print('iteration: %d/160'%(jj+1))
        for kk in range(5):
            #Patch=np.load(data_dir+'patch%d.npz'%(h+start))['arr_0']
            Patch=loadmat('%s/%s%d.mat'%(data_dir,filename,h+start))[matname]
            PatchMov2[:,jj,kk*20:(kk+1)*20]=Patch
            h+=1
    
    for ii in range(100):
        vid[:,:,ii],_=PatchManipulator_V2(PatchMov2[:,:,ii],Dimensions,Overlap,ImageSize,Params)
    #np.savez(save_dir,vid)
        
    # Save as mat file
    adict        = {}
    adict['vid'] = vid
    sio.savemat(save_dir+'/S_fista_test2.mat', adict)
    
    return vid
        
def PatchManipulator_V2(ImageBlock,Dimensions,Overlap,ImageSize,Params):
    #Step1: Preliminaries
    #Full image dimensions
    I_M=ImageSize['X']
    I_N=ImageSize['Y']
    
    #Patch dimensions
    M=Dimensions['X']
    N=Dimensions['Y']
    
    #Patch centers
    dM=int(np.floor(M/2)) #pixels around the center cM
    cM=dM+M%2-1
    dN=int(np.floor(N/2)) #pixels around the center cN
    cN=dN+N%2-1
    
    #Create center patch grids-disregard edges
    Mvec=np.arange(cM,I_M,M-Overlap['X'])
    Nvec=np.arange(cN,I_N,N-Overlap['Y'])
    
    #Add additional center in case of "overslip"
    if (I_M-1-Mvec[-1]>M-dM-M%2):
        Mvec=list(Mvec).append(Mvec[-1]+M-Overlap['X'])
    if (I_N-1-Nvec[-1]>N-dN-N%2):
        Nvec=list(Nvec).append(Nvec[-1]+N-Overlap['Y'])
        
    if Params['Type'].lower()=='extract':
        pass
    elif Params['Type'].lower()=='combine':
        #Initialization
        PatchMat=np.zeros([Mvec[-1]+dM+1,Nvec[-1]+dN+1],dtype=np.complex128)
        
        #Apodization
        ApodType='tukey'
        ApodParam={}
        ApodParam['N']=0.95 #0.5 #0.5 for 32 windows, 0.95 for 16 windows
        if ApodType.lower()=='hamming':
            pass
        elif ApodType.lower()=='triang':
            pass
        elif ApodType.lower()=='tukey':
            Apodx=tukey(Dimensions['X'],alpha=ApodParam['N'])
            Apody=tukey(Dimensions['Y'],alpha=ApodParam['N'])
            Apod=Apodx.reshape([-1,1])*Apody.reshape([1,-1])
        elif ApodType.lower()=='cosine':
            pass
        
        kk=0
        for ii in range(Mvec.size):
            for jj in range(Nvec.size):
                #Apodization
                Temp=ImageBlock[:,kk].reshape([M,N]).T*Apod
                #Add current patch in the correct place
                PatchMat[Mvec[ii]-dM+(M-1)%2:Mvec[ii]+dM+1,
                         Nvec[jj]-dN+(N-1)%2:Nvec[jj]+dN+1]+=Temp
                kk+=1
                
        PatchMat=PatchMat[0:I_M,0:I_N]
        RowEndIndex=[]
    
    return PatchMat,RowEndIndex
    
    
    
            
    