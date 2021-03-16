
#%%

import librosa, librosa.display
import os,glob
import numpy as np
import os, sys
from acrcloud.recognizer import ACRCloudRecognizer
os.getcwd()
# %%

a = librosa.load("/home/raab/jukebox/spotify/Beethoven/Ludwig van Beethoven - Eroica Dance (Arr. for Piano from Symphony No. 3 in E-Flat Major, Op. 55 by Martin Stadtfeld ).mp3")

a[0].shape
# %%


schubert https://open.spotify.com/playlist/37i9dQZF1DWY3VlkBR4Jhb?si=R1eQkIpbRimvQk1f7P4_Jg