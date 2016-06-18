img_filename = './Images/elephants/elephant.jpg'
# img_filename = './Images/imData/8.png'
img_filename2 = './Images/imData/13.png'

# ./Images/imData/ for the MIT dataset and ./Images/elephants/ for the elephant picture
img_folder = './Images/imData/'

# the side length (in pixels) of the nxn square puzzle piece
# 64 for the elephant and 28 for the MIT data
P = 28

# epsilon factor to add to distances so that no dividing by zero
eps = 0.000001

# the number of buckets to put edges into (per color channel)
num_buckets = 32

# the minimum number of edge weights that will be checked per square.
# Should be at least 5 since it can use four for the edges of that same square
# not super safe for type1, could potentially throw out everything if they all have the wrong rotation
min_edges = 250