# Cairo Showcases

This repository contains the code for the AI use cases of the Center of Artificial Intelligence and Robotics (Cairo)  

## Apps 
- [Jukebox](https://christophraab.github.io/Cairo-Showcases/jukebox/) (Music Embedding, 492 songs 15 artists)
- [Twitter](https://christophraab.github.io/Cairo-Showcases/twitter/) (Twitter Dataset Embedding ~60k Tweets from biggest Nasdaq companies.)
- [Style Transfer](https://christophraab.github.io/Cairo-Showcases/styletransfer/) (Transfer style to arbitrary image.)


## Documentation 
- [ Docs Jukebox](https://christophraab.github.io/Cairo-Showcases/jukebox/docs/) 
- [Docs Style Transfer](https://christophraab.github.io/Cairo-Showcases/styletransfer/docs/)

## Inhalt

    ├── Jukebox                 # Mozart Jukebox app to visualize music embeddings. See:
    │   ├── docs                # Explanations for jukebox
    │   ├── index.html          # Main Webapp File
    │   ├── data_storage        # Storage of trained embedding for mozart
    │   ├── deploy.sh           # deployment file on the showcase server
    │   └── others              # Favicons, licences etc.
    ├── StyleTransfer           # Style Transfer application. See:
    │   ├── docs                # Explanations for jukebox
    │   ├── index.html          # Main Webapp File
    │   ├── data_storage        # Storage of trained embedding for mozart
    │   ├── others              # Development files, favicons etcs.
    │   ├── deploy.sh           # deployment file on the showcase server
    │   └── saved_model_*       # Saved Tensorflow.Js models
    ├── Twitter                 # Twitter Embedding to visualize our Twitter dataset embeddings. See:
    │   ├── docs                # Explanations for jukebox
    │   ├── index.html          # Main Webapp File
    │   ├── data_storage        # Storage of trained embedding for mozart
    │   ├── deploy.sh           # deployment file on the showcase server
    └── └── others              # Favicons, licences etc. 

# Deployment 
All applications can be hosted via a simple webserver. In the `deploy.sh` is shown how to deploy to apache webserver. 

## Data generation for embeddings:
To save data comaptible with the tensorboard used at Jukebox or Twitter your data has to be float32 an in `*.bytes` format.
This can achieved by:

```python
# Save data
features.dtype=np.float32
features.tofile(board_data+data_path)
``` 

# Credits
- Twitter app by https://github.com/reiinakano 