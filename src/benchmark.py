import numpy as np
import matplotlib.pyplot as plt
import scipy

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc 
import matplotlib.pylab as pylab



def compressed_dct(image):
	"""
	Perform dct image compression

	:param image: grayscale matrix of the image
	:return : matrix of the compressed image
	"""

	def dct2(image):
		return scipy.fftpack.dct( scipy.fftpack.dct( image, axis=0, norm='ortho' ), axis=1, norm='ortho')


	imag_size = image.shape
	dct = np.zeros(imag_size)
	# Do 8x8 DCT on image (in-place)
	for i in r_[:imag_size[0]:8]:
		for j in r_[:imag_size[1]:8]:
			dct[i:(i+8),j:(j+8)] = dct2( image[i:(i+8),j:(j+8)] )
	return decompressed_dct(dct)

def decompressed_dct(image):
	image_size = image.shape
	def idct2(image):
		return scipy.fftpack.idct( scipy.fftpack.idct( image, axis=0 , norm='ortho'), axis=1 , norm='ortho')

	threshold = 0.012
	dct_thresh = image * (abs(image) > (threshold*np.max(image)))
	image_dct = np.zeros(image_size)

	for i in r_[:image_size[0]:8]:
		for j in r_[:image_size[1]:8]:
			image_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )
	return image_dct

def compressed_dft(image):
	imsize = image.shape
	dft = zeros(imsize,dtype='complex');
	im_dft = zeros(imsize,dtype='complex');
	# 8x8 DFT
	for i in r_[:imsize[0]:8]:
		for j in r_[:imsize[1]:8]:
			dft[i:(i+8),j:(j+8)] = np.fft.fft2( image[i:(i+8),j:(j+8)] )
	# Thresh
	thresh = 0.013
	dft_thresh = dft * (abs(dft) > (thresh*np.max(abs(dft))))

	percent_nonzeros_dft = np.sum( dft_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)
	# 8x8 iDFT
	for i in r_[:imsize[0]:8]:
		for j in r_[:imsize[1]:8]:
			im_dft[i:(i+8),j:(j+8)] = np.fft.ifft2( dft_thresh[i:(i+8),j:(j+8)] )
	return abs(im_dft)