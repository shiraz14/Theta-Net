# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:17:22 2020

@author: Sz-PC
"""
# Cell 0
# example of loading a pix2pix model and using it for image to image translation
from keras.models import load_model
from numpy import load
import time
from matplotlib import pyplot

#%% Cell 1 - load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	
    # unpack arrays
	X1 = data['arr_0']
	#X1, X2 = data['arr_0'], data['arr_1']
    
    # scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	#X2 = (X2 - 127.5) / 127.5
    
    # return output
	return [X1]
	#return [X1, X2]

#%% Cell 2 - plot & save source, generated and target images as a single file
def plot_image(src_img):	
	# scale from [-1,1] to [0,1]
	image = (src_img + 1) / 2.0	
	#titles = ['Source', 'Generated']
	# plot images row by row
	for i in range(len(image)):
		# turn off axis
		pyplot.axis('off')		
		# plot raw pixel data
		pyplot.imshow(image[i])		
	pyplot.show()
	pyplot.close()
    
#%% Cell 3 - load noisy DIC dataset
[X1] = load_real_samples('dic_256_noisy.npz')
[X2] = load_real_samples('pcm_256_noisy.npz')
print('Loaded', X1.shape, X2.shape)

#%% Cell 4 - load model
#model = load_model('Colab_O-Net PCM (120 ep) Model.h5')
model = load_model('7-layer O-Net DIC (160ep) Model_631040.h5')
#modela = load_model('model_473280.h5')
modela = load_model('7-layer_dic_model_473280.h5')
modelb = load_model('TL3-model_631040.h5')

#%% Cell 5 (for use with Cell 3) - set index value
ix = [0] # index for Validation datasets

#%% Cell 6 - defining source & GT (if available) datasets
src_imaged = X1[ix] # for Diatom dataset
src_imagep = X2[ix] # for Diatom dataset

#%% Cell 7 - generate images
gen_image = model.predict(src_imaged)
gen_imagea = modela.predict(gen_image)
gen_imageb = modelb.predict(gen_imagea)

#%% Cell 8 - plot individual images & save
plot_image(src_imaged)
plot_image(gen_image)
plot_image(gen_imageb)