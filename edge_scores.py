import cv2
import cv
import numpy as np
from constants import *
from image import *
import matplotlib.pyplot as plt
from sets import Set

def get_nonsym_compat(G, mu, S_inv):
	compat = 0
	G -= mu
	#S_inv = np.linalg.inv(S)
	return np.sum(np.dot(G, S_inv) * G)

def get_compat(left_mean, left_covar_inv, right_mean, right_covar_inv, p_i, p_j):
	G = p_j[:, 0, :] - p_i[:, P - 1, :]
	compat = get_nonsym_compat(G, left_mean, left_covar_inv) + \
		  get_nonsym_compat(-G, right_mean, right_covar_inv)
	return compat

def get_ssd_compat(p_i, p_j):
	diffs = p_i[:, P-1, :] - p_j[:, 0, :]
	return np.sum(np.square(diffs))

def is_correct_neighbor(sq, sq_rot, nbr, nbr_rot):
	correct_piece = False
	if sq_rot == 0:
		correct_piece = nbr.r == sq.r and nbr.c == sq.c + 1
	elif sq_rot == 1:
		correct_piece = nbr.r == sq.r + 1 and nbr.c == sq.c
	elif sq_rot == 2:
		correct_piece = nbr.r == sq.r and nbr.c == sq.c - 1
	elif sq_rot == 3:
		correct_piece = nbr.r == sq.r - 1 and nbr.c == sq.c
	return correct_piece, correct_piece and sq_rot == nbr_rot

def get_edge_cost(hole_index, index_grid, extra, extra_rot, neighbor_flag, img):
	n_x = hole_index[0]
	n_y = hole_index[1]
	[H, W, _] = index_grid.shape
	if neighbor_flag == 0:
		n_x += 1
		if n_x >= W:
			return 0
	elif neighbor_flag == 1:
		n_y -= 1
		if n_y < 0:
			return 0
	elif neighbor_flag == 2:
		n_x -= 1
		if n_x < 0:
			return 0
	elif neighbor_flag == 3:
		n_y += 1
		if n_y >= H:
			return 0

	if index_grid[n_y, n_x, 0] < 0:
		return 0

	n_rot = (index_grid[n_y, n_x, 1] + neighbor_flag) % 4
	e_rot = (extra_rot + neighbor_flag) % 4

	sq_e = img.pieces[extra]
	sq_n = img.pieces[index_grid[n_y, n_x, 0]]

	left_mean = sq_e.get_left_mean(e_rot)
	right_mean = sq_n.get_right_mean(n_rot)

	left_covar_inv = sq_e.get_left_covar_inv(e_rot)
	right_covar_inv = sq_n.get_right_covar_inv(n_rot)

	p_e = sq_e.get_rotated_pixels(e_rot)
	p_n = sq_n.get_rotated_pixels(n_rot)

	dist = get_compat(left_mean, left_covar_inv, right_mean, right_covar_inv, p_e, p_n)
	return dist


def compute_edge_dist(img, type2=True, ssd=False, divideBySecond=True):
	K = len(img.pieces)

	buckets = [[[Set([]) for k in xrange(num_buckets)] for j in xrange(num_buckets)] for i in xrange(num_buckets)]
	bucket_size = 256 * P / num_buckets

	for i in xrange(K):
		sq = img.pieces[i]
		for rot in xrange(4):
			pixels = sq.get_rotated_pixels(rot)
			color_sums = np.sum(pixels[:, P - 1, :], axis = 0)
			red = color_sums[0] / bucket_size
			green = color_sums[1] / bucket_size
			blue = color_sums[2] / bucket_size
			print red, green, blue
			buckets[red][green][blue].add((i, rot))

	for i in xrange(num_buckets):
		for j in xrange(num_buckets):
			for k in xrange(num_buckets):
				if len(buckets[i][j][k]) != 0:
					print len(buckets[i][j][k])



	dists = np.zeros([K, 4, K, 4])
	for i in xrange(K):
		sq_i = img.pieces[i]
		if i % 10 == 0:
			print i, '/', K
		for ri in xrange(4):
			p_i = sq_i.get_rotated_pixels(ri)
			left_mean = sq_i.get_left_mean(ri)
			left_covar_inv = sq_i.get_left_covar_inv(ri)
			for j in range(i, K):
				sq_j = img.pieces[j]
				for rj in xrange(4):
					p_j = sq_j.get_rotated_pixels(rj)

					if i == j or ((not type2) and ri != rj):
						dists[i, ri, j, rj] = float("inf")
					elif not ssd:
						right_mean = sq_j.get_right_mean(rj)
						right_covar_inv = sq_j.get_right_covar_inv(rj)
						dists[i, ri, j, rj] = get_compat(left_mean, left_covar_inv, right_mean, right_covar_inv, p_i, p_j)
					else:
						dists[i, ri, j, rj] = get_ssd_compat(p_i, p_j)

					dists[j, (rj + 2)%4, i, (ri + 2)%4] = dists[i, ri, j, rj]

	if divideBySecond:
		eps = 0.000001
		dists += eps

		for i in xrange(K):
			for ri in xrange(4):
				second_smallest = np.partition(dists[i, ri, :, :], 1, axis=None)[1]
				dists[i, ri, :, :] /= second_smallest

	return dists


def count_correct_matches(img, dists, type2=True):
	K = len(img.pieces)
	correct_pieces = 0
	correct_rots = 0
	total_neighbors = 0;

	for i in xrange(K):
		square = img.pieces[i]
		for ri in xrange(4):
			square_total_rot = (square.rot_idx + ri) % 4

			# only consider the edges which actually have a match in the image
			if not img.square_has_neighbor(square, square_total_rot):
				continue

			total_neighbors = total_neighbors + 1

			min_ind, min_rot, _ = get_nearest_edge(i, ri, dists, K, type2)
			
			min_s = img.pieces[min_ind]
			min_s_total_rot = (min_s.rot_idx + min_rot) % 4

			# check whether the closets found piece is the correct piece
			cor_piece, cor_rot = is_correct_neighbor(square, square_total_rot, min_s, min_s_total_rot)
			if cor_piece:
				correct_pieces += 1
				# check whether the closets found piece (which we know is the right piece) is matched on the right edge
				if cor_rot:
					correct_rots += 1
					
	return correct_pieces/(1.0*total_neighbors), correct_rots/(1.0*total_neighbors)

def get_nearest_edge(i, ri, dists, K, type2=True):
	min_dist = float("inf");
	min_ind = 0;
	min_rot = 0;
	for j in xrange(K):
		if i == j:
			continue
		for rj in xrange(4):
			if (type2 or ri == rj) and dists[i, ri, j, rj] < min_dist:
				min_dist = dists[i, ri, j, rj]
				min_ind = j
				min_rot = rj
	return min_ind, min_rot, min_dist