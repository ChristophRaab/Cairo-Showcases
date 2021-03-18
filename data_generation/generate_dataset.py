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
from sklearn.preprocessing import MultiLabelBinarizer
import os

class ClassicalMusic(Dataset):
  def __init__(self, path=None):
 
    self.data = np.load(path+'classicalmusic_data.npy',allow_pickle=True)
    self.labels = np.load(path+'classicalmusic_label.npy', allow_pickle=True)

  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]


    
def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
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
        # music = librosa.load(file,offset=45.0, duration=5.0,sr=None)
        # data_raw.append(music[0])
        # data.append(spec_to_image(get_melspectrogram_db(file))[np.newaxis,...])

        tag = eyed3.load(file).tag
        artist = tag.artist.split("/")[0]
        metadata.append([artist,tag.title,tag.album])
        labels.append(artist)


    mbl = MultiLabelBinarizer()
    mbl.fit([labels])
    binarized_labels = mbl.transform(labels)
    # data = np.array(data)
    # data_raw = np.array(data_raw)
    # # np.save('classicalmusic_data_raw.npy', data_raw)
    # np.save('classicalmusic_data.npy', data)
    np.save('classicalmusic_labels.npy', binarized_labels)

    pd.DataFrame(mbl.classes_).to_csv("classicalmusic_labelencoding.csv")
    df = pd.DataFrame(metadata)
    
    df.to_csv(header=["Artist","Songname","Album"],sep="\t",index=False,path_or_buf='metadata.tsv')


if __name__ == '__main__':

    feature_exctraction('/home/bix/Christoph/owncloud/mozartai/jukebox/data_generation/spotify/spotify/')