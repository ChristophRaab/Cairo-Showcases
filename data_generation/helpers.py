from data_generation.center_loss import CenterLoss, slda
import torch
from torch import nn
from torchvision import models
from torch.nn.functional import one_hot
import numpy as np 
from pytorch_metric_learning import losses
import copy
def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer



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

def create_spectral_feature_extractor():
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

def features_dataset(loader,feature_extractor,args):
    with torch.no_grad():
        feature_extractor.eval()
        data = torch.empty(0, args.bottleneck_dim).to(args.cuda)
        for (x,_) in loader:
            x = x.float().to(args.cuda)
            batch = feature_extractor(x)
            data = torch.cat([data,batch],dim=0)
    return data 

def kmeans(fes,num_cluster,iters=10):
    idx = torch.randint(0,fes.shape[0],(num_cluster,1))
    clusters = fes[idx,:].squeeze(1)
    mask = torch.ones(fes.shape[0],dtype=torch.bool) 
    mask[idx] = False
    samples = fes[mask,:]

    for i in range(iters):
        dist = torch.cdist(clusters,samples)
        assign = torch.argmin(dist,0)
        for j in range(clusters.shape[0]):
            if samples[assign==j,:].shape[0] != 0:
                clusters[j,:] = torch.mean(samples[assign==j,:],dim=0) 
            else: 
                new_idx = torch.randint(0,clusters.shape[0],(1,1))
                clusters[j,:] = clusters[new_idx,:] + torch.randn(clusters[new_idx,:].shape).to(fes.device)*0.001
    dist = torch.cdist(clusters,fes)
    ky = torch.argmin(dist,0)
    weight = 1 / torch.unique(ky,return_counts=True)[1]
    return ky,weight

def train_resnet(xs,ys,i,features_extractor,classifier,optimizer,avg_loss,avg_acc,cuda,weight=None):
    # optimizer = inv_lr_scheduler(optimizer,i,gamma=0.001,power=0.75)
    xs,ys = xs.float().to(cuda),ys.to(cuda)

    features_extractor.train(),classifier.train()
    optimizer.zero_grad() 

    fes = features_extractor(xs)
    
    ls = classifier(fes)
    if weight is None:
        classifier_loss = nn.CrossEntropyLoss()(ls,ys)
    else:
        classifier_loss = nn.CrossEntropyLoss()(ls,ys) + losses.NCALoss()(fes,ys)
    loss = classifier_loss # 
    loss.backward()
    optimizer.step()

    _,preds = nn.Softmax(1)(ls).detach().max(1)
    avg_loss = avg_loss + loss
    avg_acc  = avg_acc + (preds == ys).sum()
    # print("Progress " + str(i) + " Mean Validation Loss: "+str(round(avg_loss.item(),3))+ " Acc "+str(round(avg_acc.item(),3)))
    return optimizer,features_extractor,classifier,avg_loss,avg_acc

def validation_epoch(validation_loader,features_extractor,classifier,i,dataset_size,loader_size,best_model,best_acc,args):
    with torch.set_grad_enabled(False):
        avg_loss = avg_acc  = 0.0
        for (x,y) in validation_loader:
            avg_loss,avg_acc = eval_resnet(x,y,features_extractor,classifier,avg_loss,avg_acc,args.cuda)
    best_acc = print_learning(i,avg_acc,dataset_size,avg_loss,loader_size,best_acc)
    best_model = copy.deepcopy([features_extractor]) if avg_acc > best_acc else best_model
    return best_acc,best_model

def eval_resnet(xs,ys,features_extractor,classifier,avg_loss,avg_acc,cuda,weight=None):
    xs,ys = xs.float().to(cuda),ys.to(cuda)

    features_extractor.eval(),classifier.eval()


    fes = features_extractor(xs)
    ls = classifier(fes)
    if weight is None:
        classifier_loss = nn.CrossEntropyLoss()(ls,ys)
    else:
        classifier_loss = nn.CrossEntropyLoss(weight)(ls,ys)
    loss = classifier_loss

    _,preds = nn.Softmax(1)(ls).detach().max(1)
    avg_loss = avg_loss + loss
    avg_acc  = avg_acc + (preds == ys).sum()
    return avg_loss,avg_acc

def train_resnet_metric(xs,ys,i,features_extractor,classifier,metricl,optimizer,avg_loss,avg_acc,cuda,weight):
    # optimizer = inv_lr_scheduler(optimizer,i,gamma=0.001,power=0.75)
    xs,ys = xs.float().to(cuda),ys.to(cuda)

    features_extractor.train(),classifier.train()
    optimizer.zero_grad() 

    fes = features_extractor(xs)
    ls = classifier(fes)
    
    loss = nn.CrossEntropyLoss()(ls,ys) + losses.TripletMarginLoss()(fes,ys)
    loss.backward()
    optimizer.step()
    
    _,preds = nn.Softmax(1)(ls).detach().max(1)
    avg_loss = avg_loss + loss
    avg_acc  = avg_acc + (preds == ys).sum()
    return optimizer,features_extractor,classifier,avg_loss,avg_acc

def print_learning(i,avg_acc,dataset_size,avg_loss,loader_size,best_acc):
    avg_acc = avg_acc/dataset_size
    avg_loss = avg_loss/loader_size
    best_acc = avg_acc if avg_acc > best_acc else best_acc
    print("Progress " + str(i) + " Mean Validaiton Loss: "+str(round(avg_loss.item(),3))+ " Acc "+str(round(avg_acc.item(),3)))
    return best_acc

def entropy(p):
    p = torch.nn.functional.softmax(p,dim=1)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))