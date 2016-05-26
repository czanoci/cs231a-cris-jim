import cv2
import cv
import numpy as np
from constants import *
from image import *
import matplotlib.pyplot as plt
import os
from heapq import *
from edge_scores import *
from DSFNode import *

numNodes = 2000
forest = DisjointSetForest(numNodes)

edgeHeap = []
for i in xrange(numNodes):
	for j in xrange(numNodes):
		if i != j:
			edgeHeap.append((random.random(), i, j, 0, 0))
heapify(edgeHeap)

while forest.get_num_clusters() > 1:
	edge = heappop(edgeHeap)
	i = edge[1]
	j = edge[2]
	forest.union(i, j)

rep_0 = forest.find(0)
rep_last = forest.find(numNodes - 1)
print rep_0.get_order(), rep_last.get_order(), rep_0.get_data(), rep_last.get_data()




'''
type2 = True
ssd = False

correct_pieces_list = []
correct_rots_list = []

for f in os.listdir(img_folder):
	print f
	img = Image(img_folder + f)
	scramble(img, type2)
	for sq in img.pieces:
		sq.compute_mean_and_covar_inv()
	dists = compute_edge_dist(img, type2, ssd)
	correct_pieces, correct_rots = count_correct_matches(img, dists, type2)
	correct_pieces_list.append(correct_pieces)
	correct_rots_list.append(correct_rots)
	print correct_pieces_list
	print correct_rots_list

print np.mean(correct_pieces_list)
print np.mean(correct_rots_list)


random.seed(1122334455667)
img = Image(img_filename)
scramble(img, type2)
print len(img.pieces)

for sq in img.pieces:
	sq.compute_mean_and_covar_inv()

dists_mgc = compute_edge_dist(img, type2)
dists_ssd = compute_edge_dist(img, type2, ssd=True)
np.save('Data/dists_mgc', dists_mgc)
np.save('Data/dists_ssd', dists_ssd)


dists_mgc = np.load('Data/dists_mgc.npy')
dists_ssd = np.load('Data/dists_ssd.npy')


K = len(img.pieces)
correct_pieces = 0
correct_rots = 0

count = 0

for i in xrange(K):
	square = img.pieces[i]
	for ri in xrange(4):
		square_total_rot = (square.rot_idx + ri) % 4

		# only consider the pieces which actually have a match in the picture
		if not img.square_has_neighbor(square, square_total_rot):
			continue

		min_ind_mgc, min_rot_mgc, _ = get_nearest_edge(i, ri, dists_mgc, K, type2)
		min_s_mgc = img.pieces[min_ind_mgc]
		min_s_total_rot_mgc = (min_s_mgc.rot_idx + min_rot_mgc) % 4
		cor_piece_mgc, cor_rot_mgc = is_correct_neighbor(square, square_total_rot, min_s_mgc, min_s_total_rot_mgc)
		if not (cor_piece_mgc and cor_rot_mgc):
			continue

		min_ind_ssd, min_rot_ssd, _ = get_nearest_edge(i, ri, dists_ssd, K, type2)
		min_s_ssd = img.pieces[min_ind_ssd]
		min_s_total_rot_ssd = (min_s_ssd.rot_idx + min_rot_ssd) % 4
		cor_piece_ssd, cor_rot_ssd = is_correct_neighbor(square, square_total_rot, min_s_ssd, min_s_total_rot_ssd)
		if cor_piece_ssd:
			continue

		save_squares(square, min_s_mgc, min_s_ssd, ri, min_rot_mgc, min_rot_ssd, 
			'./Images/differences/diff' + str(count) + '.jpg')
		count += 1


#img = Image(img_filename)
#scramble(img)
#assemble_image(img)


output_img = np.zeros([X, Y, 3])

for i in xrange(X):
	for j in xrange(Y):
		output_img[i, j, 1:3] = train.centroids[sampled_colors[test_labels[i, j]]]
		output_img[i, j, 0] = output_img_l[i, j]


plt.figure(1)
plt.imshow(cv2.cvtColor(np.uint8(output_img), cv.CV_Lab2RGB))
plt.axis('off')
plt.savefig('./output3.png')
'''