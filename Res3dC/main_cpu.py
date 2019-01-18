# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:01:40 2018

@author: Yi Zhang
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.utils.data as data

import sys
sys.path.append('../')

from DataSet_3dC import ImageDataset
from network.ResNet3dC import ResNet3dC
from classes.Dataset import Converter
from classes.Player import Player

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pickle

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

"""Settings"""
"""========================================================================="""
#Name and choice of training set
ProjectName='test'
prefix='invivo' #invivo,sim_pm,sim
#Load model
loadmodel=False
mfile='../saved/Res3dC_invivo10epoch.pkl'

"""Network Settings: Remember to change the parameters when you change model!"""
gpu=True #if gpu=True, the ResNet will use more parameters
#Whether to plot predictions during training and frequency
plot=True #for cpu, it's always needed to plot
plotT=1
#seed
seed=123
torch.manual_seed(seed)
#parameters for training
lr_list=[1e-3] #list of learning rate
TrainInstances = 4 # Size of training dataset
ValInstances   = 4
BatchSize      = 4
ValBatchSize   = 4
num_epochs     = 3; #200
frame=10
#directory of datasets
d_invivo='../../../Data/Invivo_converse/'
d_simpm='../../Ultrasound_805/data/Sim_PM/'
d_sim='../../Ultrasound_805/data/Sim_noPM/'
"""========================================================================="""

#Dataset, converter and player
data_dir={'invivo':d_invivo,'sim_pm':d_simpm,'sim':d_sim}[prefix]
conter=Converter()
player=Player()
formshow={'pre':'concat','shape':(32,32,20)}
formlist=[]
for i in range(6):
    formlist.append(formshow)
minloss=np.inf
#Logs
log=open('./Results/%s_%sCPU_Res3dC_Log_Tr%s_epoch%s_lr%.2e.txt'\
         %(ProjectName,prefix,TrainInstances,num_epochs,lr_list[0]),'w')
print('Project Name: %s\n'%ProjectName)
log.write('Project Name: %s\n\n'%ProjectName)
#Loading data
print('Loading phase...')
print('----------------')
log.write('Loading phase...\n')
log.write('----------------\n')
shape_dset=(32,32,40)
# training
train_dataset=ImageDataset(round(TrainInstances),shape_dset,
                           train=0,data_dir=data_dir)
train_loader=data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
# validation
val_dataset=ImageDataset(round(ValInstances),shape_dset,
                         train=1,data_dir=data_dir)
val_loader=data.DataLoader(val_dataset,batch_size=ValBatchSize,shuffle=True)
print('Finished loading.\n')
log.write('Finished loading.\n\n');

# Training
for learning_rate in lr_list:
    #Construct network
    print('Configuring network...')
    log.write('Configuring network...\n')
    if not loadmodel:
        net=ResNet3dC(gpu)
    else:
        if mfile[-3:]=='pkl':
            net=ResNet3dC(gpu)
            state_dict=torch.load(mfile,map_location='cpu')
            net.load_state_dict(state_dict)
        else:
            net=torch.load(mfile)

    if torch.cuda.is_available():
        net=net.cuda()

    #Loss and optimizer
    floss=nn.MSELoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)

    #Array for recording datas
    outputs_S=to_var(torch.zeros([1,1,32,32,40]))
    lossmean_vec=[]
    lossmean_val_vec=[]
    
    #Training
    print('Training the model over %d samples, with learning rate %.6f\n'\
          %(TrainInstances,learning_rate))
    log.write('Training the model over %d samples, with learning rate %.6f\n\n'\
              %(TrainInstances,learning_rate))
    inst=0 #calculate the instances we have processed since last plot    
    loss_mean=0
    # Run over each epoch
    for epoch in range(num_epochs):       
        for niter,(_,S,D) in enumerate(train_loader):
            #print time
            ts=time.time()
            st=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            print('\n'+st)
            log.write('\n'+st+'\n')

            # set the gradients to zero at the beginning of each epoch            
            optimizer.zero_grad()  
            for ii in range(BatchSize):
                inputs=to_var(D[ii])   # "ii"th picture
                targets_S=to_var(S[ii])

                outputs_S=net(inputs[None,None])  # Forward
                loss=floss(outputs_S.squeeze(), targets_S)  # Current loss
                loss_mean+=loss.item()
                inst+=1
                loss.backward()
            
            optimizer.step()
            
            #Validate, rint loss and plot prediction in each iteration
            if (niter+1)%plotT==0:
                #calculate validation loss
                loss_val_mean=0
                with torch.no_grad():
                    for _,(_,Sv,Dv) in enumerate(val_loader): 
                        for jj in range(ValBatchSize):
                            inputsv=to_var(Dv[jj])   # "jj"th picture
                            targets_Sv=to_var(Sv[jj])
            
                            outputs_Sv=net(inputsv[None,None])  # Forward
                            loss_val=floss(outputs_Sv.squeeze(),targets_Sv)  # Current loss
                            loss_val_mean+=loss_val.item()
                loss_val_mean=loss_val_mean/ValInstances
                loss_mean=loss_mean/inst
                
                lossmean_vec.append(loss_mean)
                lossmean_val_vec.append(loss_val_mean)
                #Print loss
                print('[iteration%d/%d,epoch%d/%d],lossmean:%.3e,lossmean_val:%.3e'\
                      %(niter+1,int(TrainInstances/BatchSize),epoch+1,num_epochs,
                      loss_mean,loss_val_mean))
                log.write('[iteration%d/%d,epoch%d/%d],lossmean:%.3e,lossmean_val:%.3e\n'\
                      %(niter+1,int(TrainInstances/BatchSize),epoch+1,num_epochs,
                      loss_mean,loss_val_mean))
                
                #Save the model in each iteration
                if True or loss_val_mean<minloss: 
                    print('saved at [iteration%d/%d,epoch%d/%d]'\
                          %(niter+1,int(TrainInstances/BatchSize),
                            epoch+1,num_epochs))
                    log.write('saved at [iteration%d/%d,epoch%d/%d]\n'\
                          %(niter+1,int(TrainInstances/BatchSize),
                            epoch+1,num_epochs))
                    torch.save(net.state_dict(), 
                               "Results/%s_%sCPU_Res3dC_Model_Tr%s_epoch%s_lr%.2e.pkl"\
                               %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate))
                    minloss=min(loss_val_mean,minloss)   
                
                #Observe results
                if plot and ((epoch+1)%plotT==0 or epoch==0):
                    [xtr,ytr,ptr,xval,yval,pval]=conter.torch2np([D[ii],S[ii],outputs_S,
                                                                  Dv[jj],Sv[jj],outputs_Sv],
                                                                  formlist)
                    fig=player.plotmat([xtr[:,:,frame],ytr[:,:,frame],ptr[:,:,frame],
                                    xval[:,:,frame],yval[:,:,frame],pval[:,:,frame]],
                                    tit=['xtr','ytr','ptr','xval','yval','pval'],
                                    supt='{iteration%d/%d,epoch%d/%d}'\
                                    %(niter+1,int(TrainInstances/BatchSize),epoch+1,num_epochs))
                    plt.pause(0.1)
                    
                inst=0
                loss_mean=0
                
                            
    """Save logs, prediction, loss figure, loss data, model and settings """
    #Graphs
    #Save the prediction figure
    plt.savefig("Results/%s_%sCPU_Res3dC_Pred_Tr%s_epoch%s_lr%.2e.png"\
                %(ProjectName,prefix,TrainInstances,
                  num_epochs,learning_rate))

    #MSE
    fig=plt.figure()
    epochs_vec=np.arange(0,num_epochs,1)
    iteration=(np.arange(len(lossmean_vec))*plotT).astype(np.int)
    plt.semilogy(iteration,lossmean_vec, '-*',label='loss')
    plt.semilogy(iteration,lossmean_val_vec, '-*',label='loss_val')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.title("MSE")
    plt.legend()
    #Save png, pickle, data of MSE
    plt.savefig("Results/%s_%sCPU_Res3dC_LossPng_Tr%s_epoch%s_lr%.2e.png"\
                %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate))
    pickle.dump(fig,open("Results/%s_%sCPU_Res3dC_LossFig_Tr%s_epoch%s_lr%.2e.fig.pickle"\
                          %(ProjectName,prefix,TrainInstances,
                            num_epochs,learning_rate),'wb'))
    np.savez('Results/%s_%sCPU_Res3dC_LossData_Tr%s_epoch%s_lr%.2e'\
             %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate),
             lossmean_vec,lossmean_val_vec)
    
    #Save settings of training
    params={'ProjectName':ProjectName,
            'prefix':prefix,
            'mfile':mfile if loadmodel else None,
            'gpu':gpu,
            'lr_list':lr_list,
            'data_dir':data_dir,
            'shape':shape_dset,
            'TrainInstances':TrainInstances,
            'ValInstances':ValInstances,
            'BatchSize':BatchSize,
            'ValBatchSize':ValBatchSize,
            'num_epochs':num_epochs,
            'frame':frame}
    file=open('Results/%s_%sCPU_Res3dC_Settings_Tr%s_epoch%s_lr%.2e.txt'\
              %(ProjectName,prefix,TrainInstances,num_epochs,learning_rate),'w')
    file.write('Training Settings:\n\t')
    for k,v in params.items():
        file.write(k+'='+str(v)+'\n')
        file.write('\t')
    file.close()
    
    #Print min loss
    print('\nmin Loss is %f'%minloss)
    log.write('\nmin Loss is %f\n'%minloss)
    log.close()
