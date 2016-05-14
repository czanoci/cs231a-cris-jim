import cv2
import cv
import numpy as np
from constants import *
from image import *
import matplotlib.pyplot as plt
import os
from edge_scores import *

type2 = False

correct_pieces_list = []
correct_rots_list = []

for f in os.listdir(img_folder):
	print f
	img = Image(img_folder + f)
	scramble(img, type2)
	for sq in img.pieces:
		sq.compute_mean_and_covar()
	dists = compute_edge_dist(img, type2)
	correct_pieces, correct_rots = count_correct_matches(img, dists, type2)
	correct_pieces_list.append(correct_pieces)
	correct_rots_list.append(correct_rots)

print correct_pieces_list
print correct_rots_list

print np.mean(correct_pieces_list)
print np.mean(correct_rots_list)


#img = Image(img_filename)
#scramble(img)
#assemble_image(img)

'''
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