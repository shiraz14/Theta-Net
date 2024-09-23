# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:17:22 2023

@author: Sz-PC
"""

# example of loading a pix2pix model and using it for image to image translation
from keras.models import load_model
from numpy import load
from matplotlib import pyplot
from numpy.random import randint
import numpy as np

#%% load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

#%% plot & save source, generated and target images as a single file
def plot_images(src_img, gen_img, tar_img, n_samples=3):	
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (src_img + 1) / 2.0
	X_realB = (tar_img + 1) / 2.0
	X_fakeB = (gen_img + 1) / 2.0
	X_errorB = abs(X_fakeB - X_realB)
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(4, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.title('Source Image')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(4, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.title('Generated Image')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(4, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.title('Expected Image')
		pyplot.imshow(X_realB[i])
	# plot error image
	for i in range(n_samples):
		pyplot.subplot(4, n_samples, 1 + n_samples*3 + i)
		#X_errorB_val = np.mean(np.squeeze(X_errorB[i]))
		pyplot.axis('off')
		pyplot.title('Error Image')
		pyplot.imshow(X_errorB[i])		
    # save plot to file
	#filename1 = 'plot_O (Val-trained).png'
	filename1 = 'plot_Theta (Val-trained).png'
	pyplot.savefig(filename1)
	pyplot.show()
	pyplot.close()
	for i in range(n_samples):
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
# 		filename2 = 'T-Generated Image (Val-untrained)-%01d.png' % (i + 1)
# 		pyplot.savefig(filename2)
		pyplot.show()
		pyplot.close()
# 		pyplot.axis('off')
# 		pyplot.imshow(X_realB[i])
# 		filename3 = 'Expected Image (Val-trained)-%01d.png' % (i + 1)
# 		pyplot.savefig(filename3)
# 		pyplot.close
# 		pyplot.axis('off')
# 		pyplot.imshow(X_realA[i])
# 		filename4 = 'Source Image (Val-trained)-%01d.png' % (i + 1)
# 		pyplot.savefig(filename4)
# 		pyplot.close		
		pyplot.axis('off')
		pyplot.imshow(X_errorB[i])
# 		filename5 = 'T-Error Image (Val-untrained)-%01d.png' % (i + 1)
# 		pyplot.savefig(filename5)
		pyplot.show()
		pyplot.close()
	print('>Completed')
    
#%% load dataset
[X1, X2] = load_real_samples('dic_256_train (for Fig 3, SF11).npz')
# [X1, X2] = load_real_samples('dic_256_untrain (for Fig 4, SF12).npz')
# [X1, X2] = load_real_samples('pcm_256_train (for Fig 6, SF13).npz')
# [X1, X2] = load_real_samples('pcm_256_untrain (for Fig 7, SF14).npz')

print('Loaded', X1.shape, X2.shape)

#%% load U-/O-Net model (101 ep - 398344; 120 ep - 473280)
model = load_model('o-dic_model_398344.h5')
# model = load_model('dic_model_398344.h5')
# model = load_model('o-pcm_model_398344.h5')
# model = load_model('pcm_model_398344.h5')
# model = load_model('o-pcm_model_473280.h5')

#%% load Theta-Net models
# for DIC
model = load_model('7-layer O-Net DIC (160ep) Model_631040.h5')
modela = load_model('7-layer_dic_model_473280.h5')
modelb = load_model('TL3-model_631040.h5')

# for PCM
# model = load_model('Colab_O-Net PCM (120 ep) Model.h5')
# modela = load_model('model_473280.h5')
# modelb = load_model('TL3-model_631040.h5')

#%% select random example
ix = randint(0, len(X1), 3)

# index for dataset
ix[0] = 0 # for N1
ix[1] = 1 # for N2
ix[2] = 2 # for N3

#%% defining source & GT datasets
src_image, tar_image = X1[ix], X2[ix]

#%% For U-/O-Net - generate image from source
gen_image = model.predict(src_image)

#%% For Theta-Net - generate image from source (*N.B.: Run this code line-by-line, NOT all at once!)
g_image = model.predict(src_image)
# Used in Figure 4 - Optional & ONLY for DIC - scale to [-1,1], then to [-1, 0.99215686] (caused by max value being 254, not 255)
# g_image = ((((g_image - np.min(g_image)) / (np.max(g_image) - np.min(g_image))) * 254) - 127.5) / 127.5

ge_image = modela.predict(g_image)
# Used in Figure 4 - Optional & ONLY for DIC - scale to [-1,1]
# ge_image = (((ge_image - np.min(ge_image)) / (np.max(ge_image) - np.min(ge_image))) * 2) - 1

# Optional & ONLY for PCM - scale to [-1,1], then to [-1, 0.99215686] (caused by max value being 254, not 255)
# ge_image = ((((ge_image - np.min(ge_image)) / (np.max(ge_image) - np.min(ge_image))) * 254) - 127.5) / 127.5

gen_image = modelb.predict(ge_image)
# scale to [-1,0.85] [change brightness (optional - use only when required, especially for PCM)]
# used in Figure 4 also
# gen_image = (((gen_image - np.min(gen_image)) / (np.max(gen_image) - np.min(gen_image))) * 1.85) - 1

#%% Checking min & max value of each np.ndarray
print('g_image:')
print(np.min(g_image))
print(np.max(g_image))
print()
print('ge_image:')
print(np.min(ge_image))
print(np.max(ge_image))
print()
print('gen_image:')
print(np.min(gen_image))
print(np.max(gen_image))

#%% plot & save all images
plot_images(src_image, gen_image, tar_image)
