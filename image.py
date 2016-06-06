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

	def square_has_neighbor(self, square, rot_idx):
		H = self.H/P
		W = self.W/P
		if square.r == 0 and rot_idx == 3:
			return False
		elif square.c == 0 and rot_idx == 2:
			return False
		elif square.r == H - 1 and rot_idx == 1:
			return False
		elif square.c == W - 1 and rot_idx == 0:
			return False
		return True

class Square:
	def __init__(self, pixels, r, c, count, rot_idx):
		self.r = r
		self.c = c
		self.idx = count
		self.rot_idx = rot_idx
		self.pix = pixels

		self.mean = np.zeros(4)
		self.covar_inv = np.zeros(4)

	def get_rotated_pixels(self, rot=0):
		return np.rot90(self.pix, rot)

	def compute_mean_and_covar_inv(self):
		means = []
		covars_inv = []

		dummy_grad = np.array([ [0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1] ])

		for rot in xrange(4):
			pixels = np.rot90(self.pix, rot)
			GL = pixels[:, P-1, :] - pixels[:, P-2, :]
			mu = np.mean(GL, axis=0)
			GL_plus_dummy = np.concatenate((GL, dummy_grad))
			cov_inv = np.linalg.inv(np.cov(np.transpose(GL_plus_dummy)))

			means.append(mu)
			covars_inv.append(cov_inv)

		self.mean = np.array(means)
		self.covar_inv = np.array(covars_inv)

	def get_left_mean(self, rot=0):
		return self.mean[rot]

	def get_right_mean(self, rot=0):
		return self.mean[(rot+2)%4]

	def get_left_covar_inv(self, rot=0):
		return self.covar_inv[rot]

	def get_right_covar_inv(self, rot=0):
		return self.covar_inv[(rot+2)%4]



def scramble(img, type2=True):
	num_squares_H = img.H/P
	num_squares_W = img.W/P
	img.reduced = img.rgb[0:P*num_squares_H, 0:P*num_squares_W, :]
	pieces = []
	count = 0
	for i in xrange(num_squares_H):
		for j in xrange(num_squares_W):
			pixels = img.reduced[i*P:(i+1)*P, j*P:(j+1)*P, :]
			pixels = pixels.astype(np.int32)
			rot_idx = 0
			if type2:
				rot_idx = random.randint(0, 3)
				pixels = np.rot90(pixels, rot_idx)
			sq = Square(pixels, i, j, count, rot_idx)
			count += 1
			pieces.append(sq)
	random.shuffle(pieces)
	img.pieces = np.array(pieces)

def assemble_image(img, filename):
	H, W, _ = img.reduced.shape
	pieces = np.reshape(img.pieces, (H/P, W/P))
	reconstruct = np.zeros([H, W, 3])
	for i in xrange(H/P):
		for j in xrange(W/P):
			reconstruct[i*P:(i+1)*P, j*P:(j+1)*P, :] = pieces[i, j].pix

	cv2.imwrite(filename, reconstruct)

def save_squares(square, match1, match2, s_rot, m1_rot, m2_rot, filename):
	pixels = np.zeros([2*P, 2*P, 3])
	pixels[0:P, 0:P, :] = square.get_rotated_pixels(s_rot)
	pixels[0:P, P:2*P, :] = match1.get_rotated_pixels(m1_rot)
	pixels[P:2*P, 0:P, :] = square.get_rotated_pixels(s_rot)
	pixels[P:2*P, P:2*P, :] = match2.get_rotated_pixels(m2_rot)

	cv2.imwrite(filename, pixels)

def direct_metric_helper(img, index_grid, rot):
	true_H = img.H/P
	true_W = img.W/P
	index_grid2 = np.rot90(index_grid[:, :, 0], rot)
	index_grid3 = np.rot90(index_grid[:, :, 1], rot)
	[H, W] = index_grid2.shape
	if true_H != H or true_W != W:
		return 0
	correct = 0
	for x in xrange(W):
		for y in xrange(H):
			piece_index = index_grid2[y, x]
			piece_rot = index_grid3[y, x]
			sq = img.pieces[piece_index]
			if y == H - sq.r - 1 and x == sq.c and (sq.rot_idx + piece_rot - rot) % 4 == 0:
				correct += 1
	return correct

def direct_metric(img, index_grid):
	list_correct = []
	for rot in xrange(4):
		correct = direct_metric_helper(img, index_grid, rot)
		list_correct.append(correct)
	return max(list_correct)

def neighbor_metric_helper(img, index_grid, img_map):
	true_H = img.H/P
	true_W = img.W/P
	#index_grid2 = np.rot90(index_grid[:, :, 0], 0)
	#index_grid3 = np.rot90(index_grid[:, :, 1], 0)
	[H, W, _] = index_grid.shape
	#if true_H != H or true_W != W:
		#return 0
	correct = 0
	print index_grid[:, :, 0]
	print index_grid[:, :, 1]
	for x in xrange(W):
		for y in xrange(H):
			piece_index = index_grid[y, x, 0]
			piece_rot = index_grid[y, x, 1]
			sq = img.pieces[piece_index]
			print ''
			print x, y, sq.r, sq.c, (piece_rot + sq.rot_idx) % 4
			if (piece_rot + sq.rot_idx) % 4 == 0:
				print 'case 0'
				right_r = sq.r
				right_c = sq.c + 1
				bot_r = sq.r - 1
				bot_c = sq.c
			elif (piece_rot + sq.rot_idx) % 4 == 1:
				print 'case 1'
				right_r = sq.r - 1 
				right_c = sq.c
				bot_r = sq.r
				bot_c = sq.c - 1
			elif (piece_rot + sq.rot_idx) % 4 == 2:
				print 'case 2'
				right_r = sq.r
				right_c = sq.c - 1
				bot_r = sq.r + 1
				bot_c = sq.c
			else:
				print 'case 3'
				right_r = sq.r + 1 
				right_c = sq.c
				bot_r = sq.r
				bot_c = sq.c + 1

			print right_r, right_c, bot_r, bot_c

			if right_r >=0 and right_r < true_H and right_c >= 0 and right_c < true_W and x < W-1:
				right_index = img_map[(right_r, right_c)]
				right_rot = img.pieces[right_index].rot_idx
				right_sq = img.pieces[right_index]
				print 'right', right_sq.r, right_sq.c, right_index, right_rot
				if index_grid[y, x+1, 0] == right_index :#and ((piece_rot + sq.rot_idx) - (right_rot + right_sq.rot_idx)) % 4 == 0:
					correct += 1

			if bot_r >=0 and bot_r < true_H and bot_c >= 0 and bot_c < true_W and y < H-1:
				bot_index = img_map[(bot_r, bot_c)]
				bot_rot = img.pieces[bot_index].rot_idx
				bot_sq = img.pieces[bot_index]
				print 'bot', bot_sq.r, bot_sq.c, bot_index, bot_rot
				if index_grid[y+1, x, 0] == bot_index :#and ((piece_rot + sq.rot_idx) - (bot_rot + bot_sq.rot_idx)) % 4 == 0:
					correct += 1
	return correct



def neighbor_metric(img, index_grid):
	img_map = {}
	for piece in img.pieces:
		r = piece.r
		c = piece.c
		idx = piece.idx
		img_map[(r, c)] = idx

	return neighbor_metric_helper(img, index_grid, img_map)
