import torch
from torch import nn
from torchvision import models


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

def features_dataset(loader,feature_extractor,args):
    with torch.no_grad():
        feature_extractor.eval()
        data = torch.empty(0, args.bottleneck_dim).to(args.cuda)
        for (x,_) in loader:
            with torch.set_grad_enabled(False):
                feature_extractor.eval()
                x = x.float().to(args.cuda)
                batch = feature_extractor(x)
                data = torch.cat([data,batch],dim=0)
    return data 

def train_resnet(xs,ys,features_extractor,classifier,optimizer,avg_loss,avg_acc,cuda,weight=None):

    xs,ys = xs.float().to(cuda),ys.to(cuda)

    features_extractor.train(),classifier.train()
    optimizer.zero_grad() 

    fes = features_extractor(xs)
    ls = classifier(fes)
    if weight is None:
        classifier_loss = nn.CrossEntropyLoss()(ls,ys)
    else:
        classifier_loss = nn.CrossEntropyLoss(weight)(ls,ys)
    loss = classifier_loss
    loss.backward()
    optimizer.step()

    _,preds = nn.Softmax(1)(ls).detach().max(1)
    avg_loss = avg_loss + loss
    avg_acc  = avg_acc + (preds == ys).sum()
    return optimizer,features_extractor,classifier,avg_loss,avg_acc


def train_resnet_metric(xs,ys,features_extractor,classifier,metricl,optimizer,avg_loss,avg_acc,cuda):

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

def print_learning(i,avg_acc,dataset_size,avg_loss,loader_size,best_acc):
    avg_acc = avg_acc/dataset_size
    avg_loss = avg_loss/loader_size
    best_acc = avg_acc if avg_acc > best_acc else best_acc
    print("Progress " + str(i) + " Mean Training Loss: "+str(round(avg_loss.item(),3))+ " Acc "+str(round(avg_acc.item(),3)))