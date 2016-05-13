import numpy as np
import random
from constants import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

class Image:
	def __init__(self, filename):
		self.rgb = cv2.imread(filename)
		self.H = self.rgb.shape[0]
		self.W = self.rgb.shape[1] 
		self.reduced = []
		self.pieces = []
		#self.lab = cv2.cvtColor(self.rgb, cv.CV_BGR2Lab)

class Square:
	def __init__(self, pixels, r, c, count, rot_idx):
		self.r = r
		self.c = c
		self.idx = count
		self.rot_idx = rot_idx
		self.pix = pixels
		self.pix90 = np.rot90(pixels, 1)
		self.pix180 = np.rot90(pixels, 2)
		self.pix270 = np.rot90(pixels, 3)

		grads = np.zeros(4)


def scramble(img, type2=True):
	num_squares_H = img.H/P
	num_squares_W = img.W/P
	img.reduced = img.rgb[0:P*num_squares_H, 0:P*num_squares_W, :]
	pieces = []
	count = 0
	for i in xrange(num_squares_H):
		for j in xrange(num_squares_W):
			pixels = img.reduced[i*P:(i+1)*P, j*P:(j+1)*P, :]
			rot_idx = 0
			if type2:
				rot_idx = random.randint(0, 3)
				pixels = np.rot90(pixels, rot_idx)
			sq = Square(pixels, i, j, count, rot_idx)
			count += 1
			pieces.append(sq)
	random.shuffle(pieces)
	img.pieces = np.array(pieces)

def assemble_image(img):
	H, W, _ = img.reduced.shape
	pieces = np.reshape(img.pieces, (H/P, W/P))
	reconstruct = np.zeros([H, W, 3])
	for i in xrange(H/P):
		for j in xrange(W/P):
			reconstruct[i*P:(i+1)*P, j*P:(j+1)*P, :] = pieces[i, j].pix

	cv2.imwrite("./Images/reconstruct.jpg", reconstruct)

