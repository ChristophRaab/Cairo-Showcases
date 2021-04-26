from __future__ import print_function, division
from numpy.lib.utils import source
import sys
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
from data_generation.helpers import features_dataset,init_weights,train_resnet,create_feature_extractor,train_resnet_metric,print_learning,kmeans
from data_generation.center_loss import CenterLoss, slda
import argparse



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def make_parser():
    parser = argparse.ArgumentParser(description='Mozart Feature Generation Adversarial')
    parser.add_argument('--cuda', type=str, default='cuda:0', help="GPU Device ID")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of data loader workers")
    parser.add_argument('--bottleneck_dim', type=int, default=3, help="Bottleneck Dimension of Feature Extractor")
    parser.add_argument('--lr', type=float, default=0.001, help="Feature Extractor and Classifier LR")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size of Train loaders")
    parser.add_argument('--num_epochs', type=int, default=100, help="Training Epochs")
    parser.add_argument('--eval_epoch', type=int, default=5, help="Evaluation Cycle")
    parser.add_argument('--mode', type=str, default="supervised", help="Network Mode" )
    parser.add_argument('--path', type=str, default="/home/raabc/Jukebox", help="Network Mode" )

    return parser.parse_args()

def train_representation(args):

    data = np.load(args.path+'/classicalmusic_data.npy')
    label = np.load(args.path+'/classicalmusic_labels.npy')

    l = data.shape[0]

    train_dataset = TensorDataset(torch.tensor(data,dtype=torch.float),torch.tensor(label)) # create your datset

    train_loader = DataLoader(train_dataset,shuffle=True,num_workers=args.num_workers,batch_size=args.batch_size)
    extract_loader = DataLoader(train_dataset,shuffle=False,num_workers=args.num_workers,batch_size=args.batch_size)

    train_loader_size,extract_loader_size = len(train_loader),len(extract_loader)
    train_dataset_size = len(train_dataset)
    num_classes = len(torch.unique(train_dataset.tensors[1]))

    features_extractor = nn.Sequential(create_feature_extractor(),nn.Flatten(),nn.Linear(2048,args.bottleneck_dim),nn.ReLU()).to(args.cuda)
    classifier = nn.Sequential(nn.Linear(args.bottleneck_dim,num_classes)).to(args.cuda)
    classifier.apply(init_weights)
    metricl = losses.NCALoss()

    features_extractor[-1].apply(init_weights)
    optimizer = optim.Adam(
        [{'params': features_extractor[:-1].parameters(),"lr_mult":1,'decay_mult':2},
        {'params': features_extractor[-1:].parameters(),"lr_mult":10,'decay_mult':2},
        {'params': classifier.parameters(),"lr_mult":10,'decay_mult':2}],
        lr=args.lr,weight_decay=0.0005)

    features_extractor.train(),classifier.train()
    best_model = copy.deepcopy([features_extractor])
    best_acc = 0
    if args.mode != "unsupervised":
        for i in range(args.num_epochs):
            with torch.set_grad_enabled(True):
                avg_loss = avg_acc  = 0.0
                for (xs,ys) in train_loader:
                    
                    if args.mode == "metric":
                        optimizer,features_extractor,loss_func,avg_loss,avg_acc= train_resnet_metric(xs,ys,features_extractor,classifier,metricl,optimizer,avg_loss,avg_acc,args.cuda)
                    else:
                        optimizer,features_extractor,loss_func,avg_loss,avg_acc= train_resnet(xs,ys,features_extractor,classifier,optimizer,avg_loss,avg_acc,args.cuda)
            best_acc = print_learning(i,avg_acc,train_dataset_size,avg_loss,train_loader_size,best_acc)
            best_model = copy.deepcopy([features_extractor]) if avg_acc > best_acc else best_model
    else:
        for i in range(args.num_epochs):
            with torch.set_grad_enabled(True):
                avg_loss = avg_acc  = 0.0
                for (xs,ys) in train_loader:
                    with torch.no_grad():
                        yc,_ = kmeans(features_extractor(xs.float().to(args.cuda)),12)
                    optimizer,features_extractor,loss_func,avg_loss,avg_acc= train_resnet(xs,ys,features_extractor,classifier,optimizer,avg_loss,avg_acc,args.cuda)
                best_acc = print_learning(i,avg_acc,train_dataset_size,avg_loss,train_loader_size,best_acc)
                best_model = copy.deepcopy([features_extractor]) if avg_acc > best_acc else best_model

    print(str(best_acc))
    data = features_dataset(extract_loader,features_extractor,args)
    # data = (data - data.mean(0)) / data.std(0)
    data_max,_ = torch.max(data,0)
    data_min,_ = torch.min(data,0)
    data = (data-data_min) / (data_max - data_min)
    data = torch.nan_to_num(data,0)
    data.detach().cpu().numpy().tofile("../data_storage/resnet_"+args.mode+".bytes")


if __name__ == "__main__":

    args = make_parser()
    train_representation(args)