import torch
import numpy as np
import torch.utils.data as data
from scipy.io import loadmat

def preprocess(L,S,D):
    A=max(np.max(np.abs(L)),np.max(np.abs(S)),np.max(np.abs(D)))   
    if A==0:
        A=1
    L=L/A
    S=S/A
    D=D/A
    L=np.concatenate((L.real,L.imag),axis=-1)
    S=np.concatenate((S.real,S.imag),axis=-1)
    D=np.concatenate((D.real,D.imag),axis=-1)
    return L,S,D
    
class ImageDataset(data.Dataset):

    #DATA_DIR = '/home/orens/Projects/LS/Matlab/fista_patch'
    #DATA_DIR = '/home/orens/Projects/LS'
    DATA_DIR='/home/yizhang/Ultrasound_816/data/SimMv'
    #DATA_DIR='D:/SpyderProject/Ultrasound/data/real_data/'
    #DATA_DIR='../data/Sim_PM/'

    def __init__(self, NumInstances, shape, train, transform=None, data_dir=None):
        data_dir = self.DATA_DIR if data_dir is None else data_dir
        self.shape=shape

        # dummy image loader
        images_L = torch.zeros(tuple([NumInstances])+self.shape)
        images_S = torch.zeros(tuple([NumInstances])+self.shape)
        images_D = torch.zeros(tuple([NumInstances])+self.shape)
        
        #   --  TRAIN  --  RAT 1
        if train is 0:
            for n in range(NumInstances):
                if np.mod(n, 600) == 0: print('loading train set %s' % (n))
                L=loadmat(data_dir + '/fista/train/L_fista%s.mat' % (n))['patch_180']
                S=loadmat(data_dir + '/fista/train/S_fista%s.mat' % (n))['patch_180']
                D=loadmat(data_dir + '/D_data/train/D%s.mat' % (n))['patch_180']
                L,S,D=preprocess(L,S,D)
                
                images_L[n] = torch.from_numpy(L.reshape(self.shape))
                images_S[n] = torch.from_numpy(S.reshape(self.shape))
                images_D[n] = torch.from_numpy(D.reshape(self.shape))
        
        #   --  VALIDATION -- RAT 2, 100 frames 
        if train is 1:
            IndParam = 2400
            for n in range(IndParam, IndParam + NumInstances):
                if np.mod(n - IndParam, 200) == 0: print('loading validation set %s' % (n - IndParam))
                L=loadmat(data_dir + '/fista/val/L_fista%s.mat' % (n))['patch_180']
                S=loadmat(data_dir + '/fista/val/S_fista%s.mat' % (n))['patch_180']
                D=loadmat(data_dir + '/D_data/val/D%s.mat' % (n))['patch_180']
                L,S,D=preprocess(L,S,D)
                
                images_L[n-IndParam] = torch.from_numpy(L.reshape(self.shape))
                images_S[n-IndParam] = torch.from_numpy(S.reshape(self.shape))
                images_D[n-IndParam] = torch.from_numpy(D.reshape(self.shape))
        
        if train is 2:
            IndParam = 3200
            for n in range(IndParam, IndParam + NumInstances):
                if np.mod(n - IndParam, 200) == 0: print('loading test1 set %s' % (n - IndParam))
                L=loadmat(data_dir + '/fista/test1/L_fista%s.mat' % (n))['patch_180']
                S=loadmat(data_dir + '/fista/test1/S_fista%s.mat' % (n))['patch_180']
                D=loadmat(data_dir + '/D_data/test1/D%s.mat' % (n))['patch_180']
                L,S,D=preprocess(L,S,D)
                
                images_L[n-IndParam] = torch.from_numpy(L.reshape(self.shape))
                images_S[n-IndParam] = torch.from_numpy(S.reshape(self.shape))
                images_D[n-IndParam] = torch.from_numpy(D.reshape(self.shape))

        #   --  TEST 2 --  RAT 2, 100 frames
        if train is 3:
        #   --  TEST 1  -- RAT 2, 100 frames
            IndParam = 4000
            for n in range(IndParam, IndParam + NumInstances):
                if np.mod(n - IndParam, 200) == 0: print('loading test2 set %s' % (n - IndParam))
                L=loadmat(data_dir + '/fista/test2/L_fista%s.mat' % (n))['L_est_f']
                S=loadmat(data_dir + '/fista/test2/S_fista%s.mat' % (n))['S_est_f']
                D=loadmat(data_dir + '/D_data/test2/D%s.mat' % (n))['Patch']
                L,S,D=preprocess(L,S,D)
                
                images_L[n-IndParam] = torch.from_numpy(L.reshape(self.shape))
                images_S[n-IndParam] = torch.from_numpy(S.reshape(self.shape))
                images_D[n-IndParam] = torch.from_numpy(D.reshape(self.shape))
        
        self.transform = transform
        self.images_L = images_L
        self.images_S = images_S
        self.images_D = images_D

    def __getitem__(self, index):

        L = self.images_L[index]
        S = self.images_S[index]
        D = self.images_D[index]


        return L, S, D

    def __len__(self):

        return len(self.images_L)

