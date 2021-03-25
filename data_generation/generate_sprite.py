
def sprite(path):

    data_raw = []
    data = []
    labels = []
    metadata = []

    for file in glob.glob(path+ '/**/*.mp3', recursive=True):


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


feature_exctraction('/home/bix/Christoph/owncloud/mozartai/jukebox/data_generation/spotify/spotify/')