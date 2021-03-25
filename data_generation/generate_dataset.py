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


if __name__ == '__main__':

  # feature_exctraction('/home/bix/Christoph/owncloud/mozartai/jukebox/data_generation/spotify/spotify/')

  # from sklearn.decomposition import PCA
  # data = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_data_raw.npy')
  # data = PCA(50).fit_transform(data)
  # data.tofile("data_generation/pca.bytes")

  o = np.fromfile("/home/bix/Christoph/owncloud/mozartai/jukebox/data_storage/resnet.bytes")
  p = np.fromfile("/home/bix/Christoph/owncloud/mozartai/jukebox/data_storage/resnet_metric.bytes")

  print(str((o-p).sum()))