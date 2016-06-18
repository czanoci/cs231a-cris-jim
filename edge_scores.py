import cv2
import cv
import numpy as np
from constants import *
from image import *
import matplotlib.pyplot as plt
from sets import Set

def get_nonsym_compat(G, mu, S_inv):
	compat = 0
	G -= mu.astype(G.dtype)
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

def getOffsetNeighbors(buckets, i, j, k, dist):
	results = Set([])

	for i1 in range(max(0, i - dist), min(num_buckets, i + dist + 1)):
		for j1 in range(max(0, j - dist), min(num_buckets, j + dist + 1)):
			for k1 in range(max(0, k - dist), min(num_buckets, k + dist + 1)):
				if max(abs(i1 - i), abs(j1 - j), abs(k1 - k)) != dist:
					continue
				results |= buckets[i1][j1][k1]

	return results


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
			buckets[red][green][blue].add((i, rot))

	dists_list = []

	# loop over each of the buckets
	print 'computing edge dists...'
	for i in xrange(num_buckets):
		print 'i', i, '/', num_buckets
		for j in xrange(num_buckets):
			print '  j', j, '/', num_buckets
			for k in xrange(num_buckets):
				bucket_center = buckets[i][j][k]

				neighbors = bucket_center.copy()
				dist = 0
				# expand outward in levels until we get enough neighbors
				while len(neighbors) < min_edges:
					dist += 1
					#TODO: check syntax
					neighbors |= getOffsetNeighbors(buckets, i, j, k, dist)

				# loop over each edge in the current bucket
				for e in bucket_center:
					sq_ind = e[0]
					sq_rot = e[1]
					sq = img.pieces[sq_ind]
					pix = sq.get_rotated_pixels(sq_rot)
					left_mean = sq.get_left_mean(sq_rot)
					left_covar_inv = sq.get_left_covar_inv(sq_rot)

					real_neighbor_count = 0
					neighbor_dists = np.zeros([len(neighbors), 3])

					# loop over each of the neighbors
					for e2 in neighbors:
						sq2_ind = e2[0]
						sq2_rot = e2[1]
						# make sure the neighbor isn't the same piece (and has the right rotation if type 1)
						if sq2_ind == sq_ind or ((not type2) and sq_rot != sq2_rot):
							continue

						# compute the distance for this pair of edges
						sq2 = img.pieces[sq2_ind]
						pix2 = sq2.get_rotated_pixels(sq2_rot)
						if not ssd:
							right_mean = sq2.get_right_mean(sq2_rot)
							right_covar_inv = sq2.get_right_covar_inv(sq2_rot)
							d = get_compat(left_mean, left_covar_inv, right_mean, right_covar_inv, pix, pix2)
						else:
							d = get_ssd_compat(pix, pix2)

						neighbor_dists[real_neighbor_count, 0] = sq2_ind
						neighbor_dists[real_neighbor_count, 1] = sq2_rot
						neighbor_dists[real_neighbor_count, 2] = d
						real_neighbor_count += 1

					#divide by the second smallest for that edge
					neighbor_dists = neighbor_dists[:real_neighbor_count, :]
					if divideBySecond:
						neighbor_dists[:, 2] += eps
						second_smallest = np.partition(neighbor_dists[:, 2], 1, axis=None)[1]
						neighbor_dists[:, 2] /= second_smallest

					# add all the edges to the list
					for n_ind in xrange(real_neighbor_count):
						#TODO: add in for both directions
						dists_list.append( (neighbor_dists[n_ind, 2], sq_ind, neighbor_dists[n_ind, 0].astype(int), sq_rot, neighbor_dists[n_ind, 1].astype(int)) )

	return dists_list





	# dists = np.zeros([K, 4, K, 4])
	# for i in xrange(K):
	# 	sq_i = img.pieces[i]
	# 	if i % 10 == 0:
	# 		print i, '/', K
	# 	for ri in xrange(4):
	# 		p_i = sq_i.get_rotated_pixels(ri)
	# 		left_mean = sq_i.get_left_mean(ri)
	# 		left_covar_inv = sq_i.get_left_covar_inv(ri)
	# 		for j in range(i, K):
	# 			sq_j = img.pieces[j]
	# 			for rj in xrange(4):
	# 				p_j = sq_j.get_rotated_pixels(rj)

	# 				if i == j or ((not type2) and ri != rj):
	# 					dists[i, ri, j, rj] = float("inf")
	# 				elif not ssd:
	# 					right_mean = sq_j.get_right_mean(rj)
	# 					right_covar_inv = sq_j.get_right_covar_inv(rj)
	# 					dists[i, ri, j, rj] = get_compat(left_mean, left_covar_inv, right_mean, right_covar_inv, p_i, p_j)
	# 				else:
	# 					dists[i, ri, j, rj] = get_ssd_compat(p_i, p_j)

	# 				dists[j, (rj + 2)%4, i, (ri + 2)%4] = dists[i, ri, j, rj]

	# if divideBySecond:
	# 	eps = 0.000001
	# 	dists += eps

	# 	for i in xrange(K):
	# 		for ri in xrange(4):
	# 			second_smallest = np.partition(dists[i, ri, :, :], 1, axis=None)[1]
	# 			dists[i, ri, :, :] /= second_smallest

	# return dists


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