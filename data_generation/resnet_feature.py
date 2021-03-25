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

from data_generation.center_loss import CenterLoss, slda

def RSL(source,target,k=1):
    ns, nt = source.size(0), target.size(0)
    d = source.size(1)

    # Compute singular values of layers output
    d, u,v = torch.svd(source)
    r, l,s = torch.svd(target)

    # Control sign of singular vectors in backprob.
    d,v,r,s= torch.abs(d),torch.abs(v),torch.abs(r),torch.abs(s)


    u_k = u[:-k]

    #BSS with Spectral loss
    u_n = torch.pow(u[-k:],2)
    u_new = torch.cat([u_k,u_n])

    # Compute Spectral loss
    loss = torch.norm(u_new-l)
    loss = loss / ns

    return loss

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

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

def create_feature_extractor():
    model_resnet = models.resnet50(pretrained=True,)
    conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    bn1 = model_resnet.bn1
    relu = model_resnet.relu
    maxpool = model_resnet.maxpool
    layer1 = model_resnet.layer1
    layer2 = model_resnet.layer2
    layer3 = model_resnet.layer3
    layer4 = model_resnet.layer4
    avgpool = model_resnet.avgpool
    feature_layers = nn.Sequential(conv1, bn1, relu, maxpool, \
                            layer1, layer2, layer3, layer4, avgpool)
    return feature_layers

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

def train_resnet(xs,ys,features_extractor,classifier,optimizer,avg_loss,avg_acc):

    xs,ys = xs.float().to(cuda),ys.to(cuda)

    features_extractor.train(),classifier.train()
    optimizer.zero_grad() 

    fes = features_extractor(xs)
    ls = classifier(fes)
    classifier_loss = nn.CrossEntropyLoss()(ls,ys)
    loss = classifier_loss
    loss.backward()
    optimizer.step()

    _,preds = nn.Softmax(1)(ls).detach().max(1)
    avg_loss = avg_loss + loss
    avg_acc  = avg_acc + (preds == ys).sum()
    return optimizer,features_extractor,classifier,avg_loss,avg_acc

    
def train_resnet_metric(xs,ys,features_extractor,classifier,metricl,optimizer,avg_loss,avg_acc):

    xs,ys = xs.float().to(cuda),ys.to(cuda)

    features_extractor.train(),classifier.train()
    optimizer.zero_grad() 

    fes = features_extractor(xs)
    ls = classifier(fes)
    
    loss = nn.CrossEntropyLoss()(ls,ys) + metricl(fes,ys)
    loss.backward()
    optimizer.step()
    
    _,preds = nn.Softmax(1)(ls).detach().max(1)
    avg_loss = avg_loss + loss
    avg_acc  = avg_acc + (preds == ys).sum()
    return optimizer,features_extractor,classifier,avg_loss,avg_acc

def train_deep_clustering(xs,ys,features_extractor,classifier,optimizer,avg_loss,avg_acc):
    ys.detach()
    xs = xs.float().to(cuda)

    features_extractor.train(),loss_func.train()
    optimizer.zero_grad() 

    fes = features_extractor(xs)
    ls = classifier(fes,ys)

    with torch.no_grad():
        idx = torch.randint(0,fes.shape[0],(num_classes,1))
        clusters = fes[idx,:].squeeze(1)
        mask = torch.ones(fes.shape[0],dtype=torch.bool) 
        mask[idx] = False
        samples = fes[mask,:]

        for i in range(50):
            dist = torch.cdist(clusters,samples)
            assign = torch.argmin(dist,0)
            for j in range(clusters.shape[0]):
                if samples[assign==j,:].shape[0] != 0:
                    clusters[j,:] = torch.mean(samples[assign==j,:],dim=0) 
                else: 
                    new_idx = torch.randint(0,clusters.shape[0],(1,1))
                    clusters[j,:] = clusters[new_idx,:] + torch.randn(clusters[new_idx,:].shape).to(cuda)*0.001
        dist = torch.cdist(clusters,fes)
        ys = torch.argmin(dist,0)
        ys_weight = 1 / torch.unique(ys,return_counts=True)[1]
        weight = ys_weight[ys]
    if ys_weight.shape[0] == num_classes:
        classifier_loss =  torch.sum( weight * nn.CrossEntropyLoss(reduction='none')(ls,ys))/ torch.sum(weight).detach().item()
    else:
        classifier_loss = nn.CrossEntropyLoss()(ls,ys)
    loss = classifier_loss
    loss.backward()
    optimizer.step()

    _,preds = nn.Softmax(1)(ls).detach().max(1)
    avg_loss = avg_loss + loss
    avg_acc  = avg_acc + (preds == ys).sum()
    return optimizer,features_extractor,classifier,avg_loss,avg_acc

def feat_extract(x,features_extractor):
    x = x.float().to(cuda)
    features_extractor.eval(),loss_func.eval()
    return features_extractor(x)

features_extractor.train(),classifier.train()
for i in range(num_epochs):
    with torch.set_grad_enabled(True):
        avg_loss = avg_acc  = 0.0
        for (xs,ys) in train_loader:
            
            # optimizer,features_extractor,loss_func,avg_loss,avg_acc= train_resnet_metric(xs,ys,features_extractor,classifier,metricl,optimizer,avg_loss,avg_acc)
            optimizer,features_extractor,loss_func,avg_loss,avg_acc= train_resnet(xs,ys,features_extractor,classifier,optimizer,avg_loss,avg_acc)
        print("Progress " + str(i) + " Mean Training Loss: "+str(round((avg_loss/train_loader_size).item(),3))+ " Acc "+str(round((avg_acc/train_dataset_size).item(),3)))

features_extractor.eval(),classifier.eval()
data = torch.empty(0, bottleneck_dim).to(cuda)
for (xs,ys) in extract_loader:
    with torch.set_grad_enabled(False):
        batch= feat_extract(xs,features_extractor)
        data = torch.cat([data,batch],dim=0)


# data.detach().cpu().numpy().tofile("data_generation/resnet_metric.bytes")
data.detach().cpu().numpy().tofile("data_generation/resnet.bytes")