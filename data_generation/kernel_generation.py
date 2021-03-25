
import os
import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time
from sklearn import datasets
import scipy.io
import glob
from PIL import Image
import cv2
import re
import io
import base64

def read_name_from_directory(dir_name):
    file_names = glob.glob(dir_name+"/**/*.[MID|mid]*",recursive=True)
    file_names = [f.replace("\\","/").replace(dir_name,"").replace(".mid","").replace("mov","") for f in file_names]
    meta = [[f.split("/")[0],f.split("/")[1]] for f in file_names]
    labels = [m[0] for m in meta]
    names = [m[1].split(".")[0] for m in meta]
    return labels,names


   
def read_midi_kernel(path):
    df = pd.read_csv(path,header=None)
    return df.values.astype(np.float32)

path_data = os.getcwd() + '/data_generation/'

board_data = 'oss_data/'
labels_path = "mozart_labels.tsv"
data_path ="mozart.bytes"


## Create Metdadata
labels,names = read_name_from_directory(path_data+"/midi/classic/")
metadata = [[n,l] for n,l in zip(names,labels)]
pd.DataFrame(metadata).to_csv(header=["Songname","Artist"],sep="\t",index=False,path_or_buf=board_data+labels_path)

features = read_midi_kernel(path_data+"midi_d.csv")
features = MinMaxScaler().fit_transform(features)
features = PCA(n_components=50).fit_transform(features)
features.tofile(board_data+data_path)

