from librosa import feature
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
import librosa
import eyed3
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import os

import cv2
    
def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

def get_melspectrogram_db(file_path, sr=None, n_fft=8192, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db

def feature_exctraction(path):

    data_raw = []
    data = []
    labels = []
    metadata = []

    for file in glob.glob(path+ '/**/*.mp3', recursive=True):
        music = librosa.load(file,offset=45.0, duration=5.0,sr=None)
        data_raw.append(music[0])
        data.append(spec_to_image(get_melspectrogram_db(file))[np.newaxis,...])

        tag = eyed3.load(file).tag
        artist = tag.artist.split("/")[0]
        metadata.append([artist,tag.title.replace("\"",""),tag.album.replace("\"","")])
        labels.append(artist)


    le = LabelEncoder()
    le.fit(labels)
    a = le.classes_
    binarized_labels = le.transform(labels)
    data = np.array(data)
    data_raw = np.array(data_raw)
    np.save('classicalmusic_data_raw.npy', data_raw)
    np.save('classicalmusic_data.npy', data)
    np.save('classicalmusic_labels.npy', binarized_labels)

    df = pd.DataFrame(metadata)
    df1 = pd.DataFrame(binarized_labels)
    df = pd.concat([df1,df],axis=1)
    df.to_csv(header=["Label","Artist","Songname","Album"],sep="\t",index=False,path_or_buf='metadata.tsv')


def create_sprite(data):
    """
    Tile images into sprite image. 
    Add any necessary padding
    """
    
    # For B&W or greyscale images
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    
    # Tile images into sprite
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    # print(data.shape) => (n, image_height, n, image_width, 3)
    
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # print(data.shape) => (n * image_height, n * image_width, 3) 
    return data

def sprite(path):

    labels = []
    metadata = []
    images = []
    for file in glob.glob(path+"spotify/spotify/"+ '/**/*.mp3', recursive=True):

        tag = eyed3.load(file).tag
        artist = tag.artist.split("/")[0]
        metadata.append([artist])
        labels.append(artist)
        input_img = cv2.imread(path+"artist_image/"+artist.split(" ")[-1].lower()+".jpg")
        input_img_resize = cv2.resize(input_img, (32,32)) 
        images.append(input_img_resize)
    img_data = np.array(images)
    sprite = create_sprite(img_data)
    cv2.imwrite("sprite.png", sprite)

if __name__ == '__main__':

  # feature_exctraction('/home/bix/Christoph/owncloud/mozartai/jukebox/data_generation/spotify/spotify/')

  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler
  data = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_data_raw.npy')
  
  data = StandardScaler().fit_transform(data)
  data = PCA(50).fit_transform(data)
  data.tofile("data_generation/pca.bytes")

  # sprite('/home/bix/Christoph/owncloud/mozartai/jukebox/data_generation/')