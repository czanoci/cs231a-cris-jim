import cv2
import cv
import numpy as np
from constants import *
from image import *
import matplotlib.pyplot as plt

def get_nonsym_compat(G, mu, S):
	compat = 0
	G -= mu
	S_inv = np.linalg.inv(S)
	return np.sum(np.dot(G, S_inv) * G)

def get_compat(s1, s2, r1, r2):
	p1 = s1.get_rotated_pixels(r1)
	p2 = s2.get_rotated_pixels(r2)
	G = p2[:, 0, :] - p1[:, P - 1, :]
	compat = get_nonsym_compat(G, s1.get_left_mean(r1), s1.get_left_covar(r1)) + \
		  get_nonsym_compat(-G, s2.get_right_mean(r2), s2.get_right_covar(r2))
	return compat


img = Image(img_filename)
scramble(img)
for sq in img.pieces:
	sq.compute_mean_and_covar()

K = len(img.pieces)

dists = np.zeros([K, 4, K, 4])

print K
for i in xrange(K):
	print i
	for ri in xrange(4):
		for j in range(i+1, K):
			for rj in xrange(4):
				dists[i, ri, j, rj] = get_compat(img.pieces[i], img.pieces[j], ri, rj)
				dists[j, (ri + 2)%4, i, (rj + 2)%4] = dists[i, ri, j, rj]


correct_pieces = 0
correct_rots = 0
for i in xrange(K):
	print i
	square = img.pieces[i]
	for ri in xrange(4):
		min_dist = float("inf");
		min_ind = 0;
		min_rot = 0;
		for j in xrange(K):
			if i == j:
				continue
			for rj in xrange(4):
				if dists[i, ri, j, rj] < min_dist:
					min_dist = dists[i, ri, j, rj]
					min_ind = j
					min_rot = rj
		# check whether the closets found piece is the correct piece
		min_s = img.pieces[min_ind]
		if abs(square.r - min_s.r) + abs(square.c - min_s.c) == 1:
			correct_pieces += 1
			# check whether the closets found piece (which we know is the right piece) is matched on the right edge
			if (square.rot_idx + ri) % 4 == (min_s.rot_idx + min_rot) % 4:
				correct_rots += 1
print K, correct_pieces, correct_rots