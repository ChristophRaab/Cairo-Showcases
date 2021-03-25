from __future__ import print_function, division
from numpy.lib.utils import source
import sys
from torch._C import device
sys.path.append("../../jukebox/")
sys.path.append("../jukebox/")
from torch.optim import optimizer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
import time
import copy
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50,resnet34
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader
import numpy as np
#from dda.adversarial.asan.loss import Entropy_Regularization
from itertools import cycle
from pytorch_metric_learning import miners, losses
np.random.seed(0)
torch.manual_seed(0)
from data_generation.helpers import init_weights,create_feature_extractor, train_resnet,feat_extract
from data_generation.center_loss import CenterLoss, slda
from sklearn.cluster import k_means
cuda = "cuda:1"


bottleneck_dim = 64
lr = 0.1
log_interval = 500
batch_size = 32
mode = "dann"
num_workers = 0
cuda = "cuda:1"
num_epochs = 50
print("clustering")
data = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_data.npy')
label = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_labels.npy')
data = (data - np.mean(data,0))/ np.std(data,0) 
l = data.shape[0]

dataset = TensorDataset(torch.tensor(data),torch.tensor(label)) # create your datset
extract_loader = DataLoader(dataset,shuffle=False,num_workers=num_workers,batch_size=86)
extract_loader_size = len(extract_loader)
dataset_size = len(dataset)
num_classes = len(torch.unique(dataset.tensors[1]))


features_extractor = nn.Sequential(create_feature_extractor(),nn.Flatten(),nn.Linear(2048,bottleneck_dim)).to(cuda)
classifier = nn.Sequential(nn.Linear(bottleneck_dim,num_classes)).to(cuda)
classifier.apply(init_weights)



features_extractor[-1].apply(init_weights)
optimizer = optim.Adam(
    [{'params': features_extractor[:-1].parameters(),"lr_mult":1,'decay_mult':2},
     {'params': features_extractor[-1:].parameters(),"lr_mult":10,'decay_mult':2},
     {'params': classifier.parameters(),"lr_mult":10,'decay_mult':2}],
     lr=lr,weight_decay=0.0005)
# scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=5   , gamma=0.1)
def kmeans(fes):
    with torch.no_grad():
        idx = torch.randint(0,fes.shape[0],(num_classes,1))
        clusters = fes[idx,:].squeeze(1)
        mask = torch.ones(fes.shape[0],dtype=torch.bool) 
        mask[idx] = False
        samples = fes[mask,:]

        for i in range(10):
            dist = torch.cdist(clusters,samples)
            assign = torch.argmin(dist,0)
            for j in range(clusters.shape[0]):
                if samples[assign==j,:].shape[0] != 0:
                    clusters[j,:] = torch.mean(samples[assign==j,:],dim=0) 
                else: 
                    new_idx = torch.randint(0,clusters.shape[0],(1,1))
                    clusters[j,:] = clusters[new_idx,:] + torch.randn(clusters[new_idx,:].shape).to(cuda)*0.001
        dist = torch.cdist(clusters,fes)
        ky = torch.argmin(dist,0)
        weight = 1 / torch.unique(ky,return_counts=True)[1]
        return ky,weight


# Start epoch: 
features_extractor.train(),classifier.train()
best_acc = 100
best_model = copy.deepcopy(features_extractor)   
for i in range(num_epochs):
    #Cluster all data, no shuffeling
    with torch.set_grad_enabled(False):
        features = torch.empty(0,bottleneck_dim,device=cuda)
        for (xs,ys) in extract_loader:
            batch= feat_extract(xs,features_extractor,cuda)
            features = torch.cat([features,batch],dim=0)
        

        features = features.detach().cpu().numpy()
        _,ky,_=k_means(features,num_classes)
        ky =torch.Tensor(ky).to(cuda).long()
        weight = 1 / torch.unique(ky,return_counts=True)[1]
        # ky,weight = kmeans(features)
        # features = features.detach().cpu().numpy()

    #Assign dataset with pseudo labels

    train_dataset = TensorDataset(torch.tensor(data,device=cuda),ky)
    train_loader = DataLoader(train_dataset,shuffle=True,num_workers=num_workers,batch_size=batch_size)
  
    with torch.set_grad_enabled(True):
        avg_loss = avg_acc  = 0.0
        for (xs,ys) in train_loader:
            optimizer,features_extractor,loss_func,avg_loss,avg_acc= train_resnet(xs,ys,features_extractor,classifier,optimizer,avg_loss,avg_acc,cuda)
            # scheduler.step()
        
        if best_acc > avg_loss/extract_loader_size:
            best_acc,best_model= avg_loss/extract_loader_size, copy.deepcopy([features_extractor])
        print("Progress " + str(i) + " Mean Training Loss: "+str(round((avg_loss/extract_loader_size).item(),3))+ " Acc "+str(round((avg_acc/dataset_size).item(),3)))
print(best_acc)
features_extractor =best_model[0]
features_extractor.eval(),classifier.eval()
data = torch.empty(0, bottleneck_dim).to(cuda)
for (xs,ys) in extract_loader:
    with torch.set_grad_enabled(False):
        features_extractor.eval()
        batch= feat_extract(xs,features_extractor,cuda)
        data = torch.cat([data,batch],dim=0)
print(data.shape)
data.detach().cpu().numpy().tofile("data_generation/deep_clustering.bytes")