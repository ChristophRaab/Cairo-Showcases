from __future__ import print_function, division
from typing import no_type_check_decorator
from numpy.lib.utils import source
import sys
from sklearn.utils import validation
from torch.functional import Tensor
sys.path.append("../../Jukebox/")
sys.path.append("../Jukebox/")
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
from data_generation.helpers import eval_resnet, features_dataset,init_weights,train_resnet,create_spectral_feature_extractor,train_resnet_metric,print_learning,kmeans, validation_epoch
from data_generation.center_loss import CenterLoss, slda
import argparse
from sklearn import cluster,manifold,decomposition
from sklearn.manifold import TSNE


def generate_labels(x,n_clusters):
    x = np.reshape(x,(x.shape[0],-1))
    sc = cluster.KMeans(n_clusters=n_clusters,n_jobs=-1)
    l =  sc.fit_predict(x)
    return l.astype(np.long)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def make_parser():
    parser = argparse.ArgumentParser(description='Mozart Feature Generation Adversarial')
    parser.add_argument('--cuda', type=str, default='cuda:3', help="GPU Device ID")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of data loader workers")
    parser.add_argument('--bottleneck_dim', type=int, default=3, help="Bottleneck Dimension of Feature Extractor")
    parser.add_argument('--lr', type=float, default=0.001, help="Feature Extractor and Classifier LR")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size of Train loaders")
    parser.add_argument('--num_epochs', type=int, default=100, help="Training Epochs")
    parser.add_argument('--eval_epoch', type=int, default=5, help="Evaluation Cycle")
    parser.add_argument('--mode', type=str, default="supervised", help="Network Mode" )
    parser.add_argument('--path', type=str, default="/home/raabc/Jukebox/data_generation/", help="Network Mode" )

    return parser.parse_args()

def train_representation(args):

    data = np.load(args.path+'/classicalmusic_data.npy')

    # if args.mode == "unsupervised":
    #     label = generate_labels(data,15)
    # else:
    label = np.load(args.path+'/classicalmusic_labels.npy')

    l = data.shape[0]
    ratio = 0.9
    indices = np.random.permutation(data.shape[0])
    sidx, tidx = indices[:int(data.shape[0]*ratio)], indices[int(data.shape[0]*ratio):]
    train_data, validation_data = torch.tensor(data[sidx,:]), torch.tensor(data[tidx,:])
    train_label, vallidation_label = torch.tensor(label[sidx]), torch.tensor(label[tidx])


    train_dataset = TensorDataset(train_data,train_label) # create your datset
    validation_dataset = TensorDataset(validation_data,vallidation_label) # create your datset
    complete_dataset = TensorDataset(torch.tensor(data),torch.tensor(label))

    train_loader = DataLoader(train_dataset,shuffle=True,num_workers=0,batch_size=args.batch_size)
    validation_loader = DataLoader(validation_dataset,shuffle=True,num_workers=0,batch_size=args.batch_size)
    extract_loader = DataLoader(complete_dataset,shuffle=False,num_workers=0,batch_size=args.batch_size)

    train_loader_size,extract_loader_size,validation_loader_size = len(train_loader),len(extract_loader),len(validation_loader)
    train_dataset_size,validation_dataset_size = len(train_dataset),len(validation_dataset)
    num_classes = len(torch.unique(train_dataset.tensors[1]))

    features_extractor = nn.Sequential(create_spectral_feature_extractor(),nn.Flatten(),nn.Linear(2048,args.bottleneck_dim),nn.BatchNorm1d(args.bottleneck_dim)).to(args.cuda)
    classifier = nn.Sequential(nn.Dropout(),nn.Linear(args.bottleneck_dim,num_classes)).to(args.cuda)
    classifier.apply(init_weights)

    features_extractor[-2].apply(init_weights)
    optimizer = optim.Adam(
        [{'params': features_extractor[:-2].parameters(),"lr_mult":1,'decay_mult':2},
        {'params': features_extractor[-2:].parameters(),"lr":10*args.lr,"lr_mult":10,'decay_mult':2},
        {'params': classifier.parameters(),"lr":10*args.lr,'decay_mult':2}],
        lr=args.lr,weight_decay=0.0005)

    features_extractor.train(),classifier.train()
    best_model = copy.deepcopy([features_extractor])
    best_acc = 0
    if args.mode != "unsupervised":
        for i in range(args.num_epochs):
            with torch.set_grad_enabled(True):
                avg_loss = avg_acc  = 0.0
                for (xs,ys) in train_loader:
                    optimizer,features_extractor,classifier,avg_loss,avg_acc= train_resnet(xs,ys,i,features_extractor,classifier,optimizer,avg_loss,avg_acc,args.cuda)
            if i % 5 == 0:
                best_acc,best_model = validation_epoch(validation_loader,features_extractor,classifier,i,validation_dataset_size,validation_loader_size,best_model,best_acc,args)
    else:
        for i in range(args.num_epochs):
            with torch.set_grad_enabled(False):
                features = torch.empty(0,args.bottleneck_dim,device=args.cuda)
                features = features_dataset(extract_loader,features_extractor,args)

                features = features.detach().cpu().numpy()
                ky = cluster.KMeans(n_clusters=num_classes).fit_predict(features)
                ky =torch.Tensor(ky).to(args.cuda).long()
                weight = 1 / torch.unique(ky,return_counts=True)[1]

            cluster_dataset = TensorDataset(torch.tensor(data,device=args.cuda),ky)
            cluster_loader = DataLoader(cluster_dataset,shuffle=True,num_workers=0,batch_size=args.batch_size)

            train_iters = 5 if ky.unique().size(0) == 15 else 1
            for j in range(train_iters):
                avg_loss = avg_acc  = 0.0
                for (xs,ys) in cluster_loader:
                    with torch.set_grad_enabled(True):
                        optimizer,features_extractor,classifier,avg_loss,avg_acc= train_resnet(xs,ys,i,features_extractor,classifier,optimizer,avg_loss,avg_acc,args.cuda,weight)
            if i % 5 == 0:
                best_acc,best_model = validation_epoch(cluster_loader,features_extractor,classifier,i,len(cluster_dataset),len(cluster_loader),best_model,best_acc,args)
    
    print("Best:"+str(best_acc))
    data = features_dataset(extract_loader,best_model[0],args)
    data.detach().cpu().numpy().tofile("../data_storage/resnet_"+args.mode+".bytes")


if __name__ == "__main__":

    args = make_parser()
    train_representation(args)

#train_representation.py --mode unsupervised --cuda cuda:1 --num_epochs 2000 --batch_size 256 
#train_representation.py --mode supervised --cuda cuda:1 --num_epochs 300 --batch_size 64 
#train_representation.py --mode metric --cuda cuda:1 --num_epochs 300 --batch_size 64 
