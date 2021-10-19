# AI Showcases - Center for Aritifical Intellgience and Robotics ([CAIRO](https://www.fhws.de/forschung/institute/idee/center/cairo/))


This repository contains the code for the AI use cases of the Center of Artificial Intelligence and Robotics ([CAIRO](https://www.fhws.de/forschung/institute/idee/center/cairo/))

## Apps 
- [Jukebox](https://christophraab.github.io/Cairo-Showcases/mozart/) (Music Embedding, 492 songs 15 artists)
- [Twitter](https://christophraab.github.io/Cairo-Showcases/twitter/) (Twitter Dataset Embedding ~60k Tweets from biggest Nasdaq companies.)
- [Style Transfer](https://christophraab.github.io/Cairo-Showcases/style/) (Transfer style to arbitrary image.)


## Documentation 
- [ Docs Jukebox](https://christophraab.github.io/Cairo-Showcases/mozart/docs/) 
- [Docs Style Transfer](https://christophraab.github.io/Cairo-Showcases/style/docs/)

## Content

    ├── Jukebox                 # Mozart Jukebox app to visualize music embeddings. See:
    │   ├── docs                # Explanations for jukebox
    │   ├── index.html          # Main Webapp File
    │   ├── data_storage        # Storage of trained embedding for mozart
    │   ├── deploy.sh           # deployment file on the showcase server
    │   └── others              # Favicons, licences etc.
    ├── style           # Style Transfer application. See:
    │   ├── docs                # Explanations for jukebox
    │   ├── index.html          # Main Webapp File
    │   ├── data_storage        # Storage of trained embedding for mozart
    │   ├── others              # Development files, favicons etcs.
    │   ├── deploy.sh           # deployment file on the showcase server
    │   └── saved_model_*       # Saved Tensorflow.Js models
    ├── Twitter                 # Twitter Embedding to visualize our Twitter dataset embeddings. See:
    │   ├── index.html          # Main Webapp File
    │   ├── data_storage        # Storage of trained embedding for mozart
    │   ├── deploy.sh           # deployment file on the showcase server
    └── └── others              # Favicons, licences etc. 

# Development Twitter & Mozart Jukebox

The Embeddings for Mozart Jukebox and Twitter are stored under `<app>/data_storage`. New embeddings must also be stored there. Finally, the new embedding must be registered in `<app>/data_storage/projector_config.json`. How is done is self-explaining in the file itself.

## Data generation for embeddings:
To save data comaptible with the tensorboard used at Jukebox or Twitter your data has to be float32 an in `*.bytes` format.
This can achieved by:

```python
# Save data
features.dtype=np.float32
features.tofile(board_data+data_path)
``` 
# Deployment 
All applications can be hosted via a simple webserver. In the `deploy.sh` is shown how to deploy to apache webserver. 

## Twitter build
The Twitter webapp is a yarn application. To build the app use:
```
yarn run prep
yarn run build
```
The build applications can be inspected for development via: 
```
yarn run start
```


# Credits
- Style app by https://github.com/reiinakano 
- Twitter Embedding trained by Julian Hanke, Christian Niekler, Trong Duc Tran
