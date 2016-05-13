import cv2
import cv
import numpy as np
from constants import *
from image import *
import matplotlib.pyplot as plt

img = Image(img_filename)
scramble(img)
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