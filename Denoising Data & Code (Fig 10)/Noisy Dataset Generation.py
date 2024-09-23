# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:58:47 2020

@author: Sz-PC
"""

# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_and_scale_images(path, size=(256,256)):
	src_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img = pixels[:, :]
		src_list.append(sat_img)		
	return [asarray(src_list)]

#%%
# training dataset
# dataset path
path = 'noise_source/DIC/'
# load DIC dataset - small
[src_images] = load_and_scale_images(path)
print('Loaded: ', src_images.shape)
# save as compressed numpy array
filename = 'dic_256_noisy.npz'
savez_compressed(filename, src_images)
print('Saved dataset: ', filename)

#%%
# dataset path
pathb = 'noise_source/PCM/'
# load PCM dataset - small
[src_imagesb] = load_and_scale_images(pathb)
print('Loaded: ', src_imagesb.shape)
# save as compressed numpy array
filenameb = 'pcm_256_noisy.npz'
savez_compressed(filenameb, src_imagesb)
print('Saved dataset: ', filenameb)