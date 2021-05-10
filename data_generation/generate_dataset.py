from os import path
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
import requests
import urllib
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import spotipy
import argparse
from spotipy.oauth2 import SpotifyClientCredentials
import urllib.request
from requests import get 
from pathlib import Path 


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

def sprite():
    labels = []
    metadata = []
    images = []
    for file in glob.glob("preview/mp3/"+ '/**/*.mp3', recursive=True):

        tag = eyed3.load(file).tag
        artist = tag.artist.split("/")[0]
        metadata.append([artist])
        labels.append(artist)
        input_img = cv2.imread("artist_image/"+artist.split(" ")[-1].lower()+".jpg")
        input_img_resize = cv2.resize(input_img, (32,32)) 
        images.append(input_img_resize)
    img_data = np.array(images)
    sprite = create_sprite(img_data)
    cv2.imwrite("sprite.png", sprite)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

def download_track(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def get_melspectrogram_db(wav, sr, n_fft=8192, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    if wav.shape[0]<5*sr:
        wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:5*sr]
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def song_meta_data(track_obj,artist_name,album_name):
    
    track = track_obj
    artists = [ n["name"] for n in track["artists"] if n["name"] == artist_name]

    if len(artists) > 0:
        artist = artists[0].replace("/"," ")
        name = track["name"].replace("/"," ")
        album = track["album"]["name"].replace("/"," ") if album_name == None else album_name
        preview_url = track["preview_url"]
        external_url = track["external_urls"]["spotify"]
        return [artist,name,album,external_url],preview_url
    else:
        return (None,None)

def extract_playlist(id,name,ptype,sp):
    ply_raw = []
    ply_data = []
    ply_label = []
    ply_metadata =[]
    
    if ptype == "playlist":
        results = sp.playlist(id) 
        album_name = None
    else:
        results = sp.album(id)
        album_name = results["name"] 

    for track_obj in results["tracks"]["items"]:
        if ptype == "playlist":
            song_meta,download_url = song_meta_data(track_obj["track"],name,album_name)
        else:
            song_meta,download_url = song_meta_data(track_obj,name,album_name)

        if download_url != None and song_meta != None:
            ply_metadata.append(song_meta)
            ply_label.append(song_meta[0])

            file = "preview/mp3/"+song_meta[0]+"/"+song_meta[1]+".mp3"
            Path("preview/mp3/"+song_meta[0]).mkdir(parents=True, exist_ok=True)

            if not os.path.isfile(file):
                download_track(download_url,file)

            music,sr = librosa.load(file,offset=0.0, duration=5.0,sr=None)
            ply_raw.append(music[0])
            song_spec = spec_to_image(get_melspectrogram_db(music,sr))
            ply_data.append(song_spec[np.newaxis,...])

            subdir = file.split("mp3/")[0]+"images/"+song_meta[0]
            Path(subdir).mkdir(parents=True, exist_ok=True)

            img_path = subdir+"/"+file.split("/")[-1].split(".mp3")[0]+".png"
            
            plt.imshow(song_spec)
            plt.savefig(img_path)

    return ply_raw, ply_data,ply_label,ply_metadata

def feature_extraction(path_playlists,sp,args):

    data_raw =[]
    data=[]
    labels=[]
    metadata =[]

    playlists = pd.read_csv(path_playlists,sep=",",header=None,index_col=False)

    for i,name,ptype,url in playlists.itertuples():
        ply_raw, ply_data,ply_label,ply_metadata = extract_playlist(url,name,ptype,sp)
        data_raw.append(ply_raw),data.append(ply_data),labels.append(ply_label),metadata.append(ply_metadata)

    le = LabelEncoder()
    le.fit(labels)
    binarized_labels = le.transform(labels)
    data = np.array(data)
    data_raw = np.array(data_raw)
    np.save('classicalmusic_data_raw.npy', data_raw)
    np.save('classicalmusic_data.npy', data)
    np.save('classicalmusic_labels.npy', binarized_labels)

    df = pd.DataFrame(metadata)
    df.to_csv(header=["Artist","Songname","Album","EmbUrl"],sep="\t",index=False,path_or_buf='metadata.tsv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extractor Music Mozart')
    parser.add_argument('--id',  default="", type=str, help="Client")
    parser.add_argument('--secret',  default="", type=str, help="Client")
    parser.add_argument('--download',  default=True, type=boolean_string, help="Download preview mp3")
    args =  parser.parse_args()

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=args.id,
                                                           client_secret=args.secret))
    sp.country_codes = ["DE"]
    feature_extraction("playlists.csv",sp,args)
  # sprite('/home/raabc/jukebox/data_generation/')

  # from sklearn.decomposition import PCA
  # from sklearn.preprocessing import StandardScaler
  # data = np.load('/home/bix/Christoph/owncloud/mozartai/jukebox/classicalmusic_data_raw.npy')
  
  # data = StandardScaler().fit_transform(data)
  # data = PCA(50).fit_transform(data)
  # data.tofile("data_generation/pca.bytes")

