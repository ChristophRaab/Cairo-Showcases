from __future__ import print_function, division
from numpy.lib.utils import source
import sys
from torch.optim import optimizer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
import time
import copy
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50

from torch.utils.data import DataLoader
import numpy as np
#from dda.adversarial.asan.loss import Entropy_Regularization
from itertools import cycle
from pytorch_metric_learning import miners, losses
np.random.seed(0)
torch.manual_seed(0)

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
batch_size = 16
mode = "dann"
num_workers = 0
cuda = "cuda:0"
num_epochs = 50


data = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_data.npy')
label = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_labels.npy')

l = data.shape[0]

indices = np.random.permutation(data.shape[0])
sidx, tidx = indices[:int(data.shape[0]*0.7)], indices[int(data.shape[0]*0.7):]
source_data, target_data = torch.tensor(data[sidx,:]), torch.tensor(data[tidx,:])
source_label, target_label = torch.tensor(label[sidx]), torch.tensor(label[tidx])


source_dataset = TensorDataset(source_data,source_label) # create your datset
target_dataset = TensorDataset(target_data,target_label) # create your datset


source_loader = DataLoader(source_dataset,shuffle=True,num_workers=num_workers,batch_size=batch_size,drop_last=True)
target_loader = DataLoader(target_dataset,shuffle=True,num_workers=num_workers,batch_size=batch_size,drop_last=True)
validation_loader = DataLoader(target_dataset,shuffle=True,num_workers=num_workers,batch_size=batch_size,drop_last=False)

source_loader_size,target_loader_size,validation_loader_size = len(source_loader),len(target_loader),len(validation_loader)
source_dataset_size, target_dataset_size = len(source_dataset),len(target_dataset)
num_classes = len(torch.unique(source_dataset.tensors[1]))

def create_feature_extractor():
    model_resnet = models.resnet50(pretrained=True)
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
    
features_extractor = nn.Sequential(create_feature_extractor(),nn.Flatten(),nn.Linear(2048,bottleneck_dim)).cuda()
classifier = nn.Sequential(nn.Linear(bottleneck_dim,num_classes)).cuda()
classifier.apply(init_weights)
features_extractor[-1].apply(init_weights)
s_sn = nn.BatchNorm1d(bottleneck_dim).cuda()
t_sn = nn.BatchNorm1d(bottleneck_dim).cuda()

optimizer = optim.SGD(
    [{'params': features_extractor[1:-1].parameters(),"lr_mult":1,'decay_mult':2},
     {'params': features_extractor[-1].parameters(),"lr_mult":10,'decay_mult':2},
    {'params': s_sn.parameters(),"lr_mult":10,'decay_mult':2},
    {'params': t_sn.parameters(),"lr_mult":10,'decay_mult':2},
     {'params': classifier.parameters(),"lr_mult":10,'decay_mult':2}],
     lr=lr,nesterov=True,momentum=0.9,weight_decay=0.0005)

# summary(features_extractor, (3, 224, 224))
# summary(classifier,(72,256))
def train(xs,ys,xt,yt,features_extractor,classifier,optimizer,s_sn,t_sn,avg_loss,avg_acc):

    xs,ys,xt,yt = xs.float().cuda(),ys.cuda(),xt.float().cuda(),yt.cuda()

    features_extractor.train(),classifier.train()
    optimizer.zero_grad() 

    fes = features_extractor(xs)
    fet = features_extractor(xt)
    fes = s_sn(fes)
    fet = t_sn(fet)
    ls = classifier(fes)
    lt = classifier(fet)
    
    
    classifier_loss = nn.CrossEntropyLoss()(ls,ys)
    loss = classifier_loss
    loss.backward()
    optimizer.step()

    _,preds = nn.Softmax(1)(ls).detach().max(1)
    avg_loss = avg_loss + loss
    avg_acc  = avg_acc + (preds == ys).sum()
    return optimizer,features_extractor,classifier,avg_loss,avg_acc

def eval(xt,yt,features_extractor,classifier,t_sn,best_acc,vavg_loss,vavg_acc):
    xt,yt = xt.float().cuda(),yt.cuda()
    features_extractor.eval(),classifier.eval()

    lt = classifier(t_sn(features_extractor(xt)))

    _,preds = nn.Softmax(1)(lt).max(1)
    classifier_loss = nn.CrossEntropyLoss()(lt,yt)
    
    loss = classifier_loss

    vavg_loss = vavg_loss + loss
    vavg_acc  = vavg_acc + (preds == yt).sum()
    best_acc = vavg_acc if vavg_acc > best_acc else best_acc
    return vavg_loss,vavg_acc

def feat_extract(x,features_extractor,classifier,sn,):
    x = x.float().cuda()
    features_extractor.eval(),classifier.eval()
    return sn(features_extractor(x))

best_acc = 0
best_model = copy.deepcopy([features_extractor,classifier,s_sn,t_sn])
for i in range(num_epochs):
    with torch.set_grad_enabled(True):
        avg_loss = avg_acc = cls_loss = avg_dc = dc_loss = classifier_loss = discriminator_loss = loss = 0.0
        training_list = zip(source_loader, cycle(target_loader)) if len(source_loader) > len(target_loader) else zip(cycle(source_loader), target_loader)
        for (xs,ys),(xt,yt) in training_list:
            
            optimizer,features_extractor,classifier,avg_loss,avg_acc= train(xs,ys,xt,yt,features_extractor,classifier,optimizer,s_sn,t_sn,avg_loss,avg_acc)

    if i % 5 == 0:
        with torch.set_grad_enabled(False):
            vavg_loss,vavg_acc = 0.0,0.0
            for xt,yt in validation_loader:
                
                vavg_loss,vavg_acc = eval(xt,yt,features_extractor,classifier,t_sn,best_acc,vavg_loss,vavg_acc)
            vavg_acc = vavg_acc/target_dataset_size
            best_acc = vavg_acc if vavg_acc > best_acc else best_acc
            best_model = copy.deepcopy([features_extractor,classifier,s_sn,t_sn]) if vavg_acc > best_acc else best_model
        print("Progress " + str(i) +  " Mean Validation Loss: "+str(round((vavg_loss/target_loader_size).item(),3))+ " Acc "+str(round((vavg_acc).item(),3))
                + " --- Mean Training Loss: "+str(round((avg_loss/source_loader_size).item(),3))+ " Acc "+str(round((avg_acc/source_dataset_size).item(),3)))
print(best_acc)
