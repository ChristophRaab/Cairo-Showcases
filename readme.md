

# Save data
# features.dtype=np.float32
# features.tofile(board_data+data_path)

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