from __future__ import print_function, division
from numpy.lib.utils import source
import sys
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
from data_generation.helpers import init_weights,train_resnet,create_feature_extractor,feat_extract,train_resnet_metric
from data_generation.center_loss import CenterLoss, slda


bottleneck_dim = 256
lr = 0.001
log_interval = 500
batch_size = 86
mode = "dann"
num_workers = 0
cuda = "cuda:1"
num_epochs = 50


data = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_data.npy')
label = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_labels.npy')

l = data.shape[0]

train_dataset = TensorDataset(torch.tensor(data),torch.tensor(label)) # create your datset

train_loader = DataLoader(train_dataset,shuffle=True,num_workers=num_workers,batch_size=batch_size)
extract_loader = DataLoader(train_dataset,shuffle=False,num_workers=num_workers,batch_size=batch_size)

train_loader_size,extract_loader_size = len(train_loader),len(extract_loader)
train_dataset_size = len(train_dataset)
num_classes = len(torch.unique(train_dataset.tensors[1]))

features_extractor = nn.Sequential(create_feature_extractor(),nn.Flatten(),nn.Linear(2048,bottleneck_dim)).to(cuda)
classifier = nn.Sequential(nn.Linear(bottleneck_dim,num_classes)).to(cuda)
classifier.apply(init_weights)
metricl = losses.NCALoss()

# loss_func = losses.NormalizedSoftmaxLoss(num_classes=num_classes,embedding_size=bottleneck_dim)
# loss_func = loss_func.to(cuda)
features_extractor[-1].apply(init_weights)
optimizer = optim.Adam(
    [{'params': features_extractor[:-1].parameters(),"lr_mult":1,'decay_mult':2},
     {'params': features_extractor[-1:].parameters(),"lr_mult":10,'decay_mult':2},
     {'params': classifier.parameters(),"lr_mult":10,'decay_mult':2}],
     lr=lr,weight_decay=0.0005)

# summary(features_extractor, (3, 224, 224))
# summary(classifier,(72,256))


features_extractor.train(),classifier.train()
for i in range(num_epochs):
    with torch.set_grad_enabled(True):
        avg_loss = avg_acc  = 0.0
        for (xs,ys) in train_loader:
            
            # optimizer,features_extractor,loss_func,avg_loss,avg_acc= train_resnet_metric(xs,ys,features_extractor,classifier,metricl,optimizer,avg_loss,avg_acc,cuda)
            optimizer,features_extractor,loss_func,avg_loss,avg_acc= train_resnet(xs,ys,features_extractor,classifier,optimizer,avg_loss,avg_acc,cuda)
        print("Progress " + str(i) + " Mean Training Loss: "+str(round((avg_loss/train_loader_size).item(),3))+ " Acc "+str(round((avg_acc/train_dataset_size).item(),3)))

features_extractor.eval(),classifier.eval()
data = torch.empty(0, bottleneck_dim).to(cuda)
for (xs,ys) in extract_loader:
    with torch.set_grad_enabled(False):
        features_extractor.eval()
        batch= feat_extract(xs,features_extractor,cuda)
        data = torch.cat([data,batch],dim=0)


# data.detach().cpu().numpy().tofile("data_generation/resnet_metric.bytes")
data.detach().cpu().numpy().tofile("data_generation/resnet.bytes")