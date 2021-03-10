
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
    file_names = [f.replace("\\","/").replace("mov","") for f in file_names]
    meta = [[f.split("/")[2],f.split("/")[-1]] for f in file_names]
    labels = [m[0] for m in meta]
    names = [m[1].split(".")[0] for m in meta]
    return labels,names

def do_pca(features):
    # # Generating PCA
    # If PCA is needed beforehand
    pca = PCA(n_components=50,
             random_state = 123,
             svd_solver = 'auto'
             )
    return pca.fit_transform(features)

def read_midi_kernel(path):
    df = pd.read_csv(path,header=None)
    return df.values

features = read_midi_kernel("midi_d.csv")


### Feature preprocessing
features =do_pca(features)
sc_X = MinMaxScaler()
features = sc_X.fit_transform(features)

### Set Tensorboard paths and create metadata
PATH = os.getcwd()
LOG_DIR = PATH + '/log/'
metadata_path = "labels.csv"

## Create Metdadata
if os.name =="nt":
    labels,names = read_name_from_directory("midi/classic/")
    metadata = [[n,l] for n,l in zip(names,labels)]
    pd.DataFrame(metadata).to_csv(header=["Name","Artist"],sep="\t",index=False,path_or_buf=LOG_DIR+metadata_path)


### Setup Tensorbard and variables
embeddings = tf.Variable(features, name='Musik')
CHECKPOINT_FILE = LOG_DIR + '/model.ckpt'
ckpt = tf.train.Checkpoint(embeddings=embeddings)
ckpt.save(CHECKPOINT_FILE)

reader = tf.train.load_checkpoint(LOG_DIR)
map = reader.get_variable_to_shape_map()
key_to_use = ""
for key in map:
    if "Musik" in key:
        key_to_use = key

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = key_to_use
embedding.metadata_path = metadata_path

writer = tf.summary.create_file_writer(LOG_DIR)
projector.visualize_embeddings(LOG_DIR,config)

#Standalone:
#python -m http.server 8080 --bind 194.95.221.186

# OLD ==========================================================================
# Start on Bix Server:
# sudo /home/bix/anaconda3/envs/mozartai/bin/tensorboard --logdir /home/bix/Christoph/owncloud/mozartai/code/log/ --host 194.95.221.31 --port 8080
# Start on Mai Server
# sudo /home/raab/miniconda3/envs/mozart/bin/tensorboard --logdir /home/raab/jukebox/log/ --port 80 --host 194.95.221.186

# Data and Package Locations  ==================================================
# Data Location
# Find-Site-Package location python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
# Site-Package location ~/anaconda3/envs/mozartai/lib/python3.8/site-packages/tensorboard/